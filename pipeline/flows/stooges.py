"""
Stooges Channels flow — build multi-channel CRT TV content for OBS composite.

Each channel is a sequence of shader-processed segments interleaved with
TV static bursts:

    glitched_seg → static → glitched_seg → static → ... → glitched_seg

Output: N video files (one per OBS CRT TV source), all at source resolution.

Usage:
    from pipeline.flows.stooges import stooges_channels
    from pathlib import Path

    channels = stooges_channels(
        src=Path("source/tooth_will_out.mp4"),
        n_channels=4,
        static_duration=0.3,
        seed=42,
    )
    # channels → [Path("output/channels/channel_00.mp4"), ...]

CLI:
    python -m pipeline.flows.stooges \\
        source/tooth_will_out.mp4 \\
        --n-channels 4 \\
        --static-duration 0.3 \\
        --seed 42
"""

from __future__ import annotations

import random
import subprocess
from pathlib import Path
from typing import Optional, Union

from prefect import flow, task
from prefect.task_runners import ConcurrentTaskRunner

from ..config import Config
from ..ffmpeg import probe, extract_segment as _extract_segment
from ..isf import load_shader_dir
from ..tasks import (
    apply_shader_stack,
    generate_static,
    bitrate_crush,
    normalize_levels,
)


@task(name="extract-segment")
def _extract_seg_task(src: Path, dst: Path,
                      start: float, duration: float,
                      fps: float, cfg=None) -> Path:
    """Task-wrapped segment extraction for Prefect visibility."""
    _extract_segment(src, dst, start, duration, fps=fps, cfg=cfg)
    return dst


@task(name="concat-channel")
def _concat_channel(pieces: list[Path], ch_out: Path,
                    cfg=None) -> Path:
    """Re-encoding concat as a tracked task."""
    c = cfg or Config()
    concat_list = ch_out.with_suffix(".concat.txt")
    with open(concat_list, "w") as f:
        for p in pieces:
            f.write(f"file '{p.resolve()}'\n")
    subprocess.run([
        c.ffmpeg_bin, "-y", "-loglevel", c.ffmpeg_loglevel,
        "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
        *c.encode_args(),
        "-g", "48",
        "-bf", "0",
        "-movflags", "+faststart",
        "-an",
        str(ch_out),
    ], check=True)
    concat_list.unlink(missing_ok=True)
    return ch_out


@flow(name="stooges-channels", log_prints=True,
      task_runner=ConcurrentTaskRunner(max_workers=4))
def stooges_channels(
    src: Path,
    shader_dir: Optional[Path] = None,
    n_channels: int = 4,
    static_duration: float = 0.3,
    shuffle: bool = True,
    n_shaders: int = 3,
    segment_counts: Union[int, list[int]] = 8,
    min_dur: float = 1.0,
    max_dur: float = 30.0,
    crush_amount: float = 0.95,
    seed: Optional[int] = None,
    output_dir: Optional[Path] = None,
    cfg: Optional[Config] = None,
) -> list[Path]:
    """
    Build N channels of glitched footage for the OBS CRT TV grid.

    Each channel:
      1. Draws random segments from the source (random start, random duration)
      2. Applies crush sandwich to each segment:
         crush → N shaders → crush → N shaders
      3. Optionally shuffles the segment order (each channel shuffled differently)
      4. Interleaves the processed segments with TV static bursts

    Result per channel:  seg → static → seg → static → seg

    Parameters
    ----------
    src             : source footage
    shader_dir      : ISF shader directory (defaults to cfg.shader_dir)
    n_channels      : number of output files (default 4)
    static_duration : length of each static burst in seconds (default 0.3)
    shuffle         : shuffle segment order per channel (default True)
    n_shaders       : shaders per stack in the crush sandwich (default 3)
    segment_counts  : segments per channel — int (same for all) or list
                      (one per channel). Controls output duration.
    min_dur         : minimum segment duration in seconds (default 1.0)
    max_dur         : maximum segment duration in seconds (default 30.0)
    crush_amount    : crush intensity 0.0–1.0 (default 0.95)
    seed            : master random seed (None = unseeded)
    output_dir      : where to write channel files
    cfg             : pipeline Config (uses defaults if None)
    """
    c = cfg or Config()
    c.ensure_dirs()
    s_dir = shader_dir or c.shader_dir

    out_dir = output_dir or c.output_dir / "channels"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Probe source ─────────────────────────────────────────────────────────

    info = probe(src, c)
    print(
        f"Source: {src.name}  "
        f"{info.width}x{info.height} @ {info.fps:.2f}fps  "
        f"{info.duration:.1f}s  {info.bitrate}kbps"
    )
    if c.default_video_bitrate is None and info.bitrate > 0:
        c.default_video_bitrate = info.bitrate

    # ── Load shader library (once; shared read-only across threads) ───────────

    all_shaders = load_shader_dir(s_dir)
    if not all_shaders:
        raise ValueError(f"No .fs shaders found in {s_dir}")
    shader_names = list(all_shaders.keys())
    print(f"Shader library: {len(all_shaders)} shaders in {s_dir}")

    # ── Resolve per-channel segment counts ───────────────────────────────────

    if isinstance(segment_counts, int):
        counts = [segment_counts] * n_channels
    else:
        counts = list(segment_counts)
        if len(counts) < n_channels:
            # Pad with the last value if list is too short
            counts += [counts[-1]] * (n_channels - len(counts))

    # ── Shader picker helper ─────────────────────────────────────────────────

    def _pick_shaders(seg_rng):
        """Pick n_shaders random shaders with randomised float params."""
        n = min(n_shaders, len(shader_names))
        chosen_names = seg_rng.sample(shader_names, n)
        chosen_shaders = [all_shaders[name] for name in chosen_names]
        paths = [s.path for s in chosen_shaders]
        overrides: dict[str, dict[str, float]] = {}
        for shader in chosen_shaders:
            sp = {}
            for inp in shader.param_inputs:
                if (inp.type == "float"
                        and inp.min is not None
                        and inp.max is not None):
                    sp[inp.name] = (
                        inp.min + seg_rng.random() * (inp.max - inp.min)
                    )
            if sp:
                overrides[shader.path.stem] = sp
        return paths, overrides

    # ── Phase 1: Submit all work across all channels ─────────────────────────

    total_segs = sum(counts)
    print(
        f"\nSubmitting {n_channels} channels ({total_segs} segments total)  "
        f"(static={static_duration}s, "
        f"{n_shaders} shaders/pass, crush={crush_amount}, "
        f"max_workers=4)..."
    )

    channel_submissions = []

    for ch_idx in range(n_channels):
        ch_seed = (seed + ch_idx) if seed is not None else None
        rng = random.Random(ch_seed)

        ch_work = c.work_dir / f"ch{ch_idx:02d}"
        ch_work.mkdir(parents=True, exist_ok=True)

        n_segs = counts[ch_idx]

        seg_dir = ch_work / "segments"
        seg_dir.mkdir(parents=True, exist_ok=True)

        # Submit extract + crush sandwich for each segment
        norm_futures = []
        for i in range(n_segs):
            dur = min_dur + rng.random() * (max_dur - min_dur)
            dur += rng.randint(0, 999) / 1000.0
            max_start = max(0, info.duration - dur - 2.0)
            start = rng.random() * max_start

            dst = seg_dir / f"seg_{i:04d}.mp4"
            seg_f = _extract_seg_task.submit(
                src, dst, start, dur, fps=info.fps, cfg=c)

            seg_rng = random.Random(rng.randint(0, 2 ** 31))

            # Crush sandwich: crush → shaders → crush → shaders → normalize
            p1 = ch_work / f"seg_{i:04d}_crush1.mp4"
            c1_f = bitrate_crush.submit(
                seg_f, p1, crush=crush_amount, cfg=c)

            paths_a, overrides_a = _pick_shaders(seg_rng)
            p2 = ch_work / f"seg_{i:04d}_shade1.mp4"
            s1_f = apply_shader_stack.submit(
                c1_f, p2, paths_a,
                param_overrides=overrides_a, cfg=c)

            p3 = ch_work / f"seg_{i:04d}_crush2.mp4"
            c2_f = bitrate_crush.submit(
                s1_f, p3, crush=crush_amount, cfg=c)

            paths_b, overrides_b = _pick_shaders(seg_rng)
            p4 = ch_work / f"seg_{i:04d}_shade2.mp4"
            s2_f = apply_shader_stack.submit(
                c2_f, p4, paths_b,
                param_overrides=overrides_b, cfg=c)

            p5 = ch_work / f"seg_{i:04d}_norm.mp4"
            n_f = normalize_levels.submit(s2_f, p5, cfg=c)

            norm_futures.append(n_f)

        static_path = ch_work / "static.mp4"
        static_f = generate_static.submit(
            static_path, static_duration,
            width=info.width, height=info.height,
            fps=info.fps, cfg=c,
        )

        ch_out = out_dir / f"channel_{ch_idx:02d}.mp4"
        channel_submissions.append(
            (ch_idx, norm_futures, static_f, rng, ch_out))

        print(f"  ch{ch_idx:02d}: {n_segs} segments submitted "
              f"(seed={ch_seed})")

    # ── Phase 2: Resolve futures, shuffle, concat per channel ────────────────

    print(f"\nAll {total_segs} segment pipelines submitted. "
          f"Waiting for results...")

    channel_files: list[Path] = []
    concat_futures = []

    for ch_idx, norm_futures, static_f, rng, ch_out in channel_submissions:
        processed = [f.result() for f in norm_futures]
        static_result = static_f.result()

        if shuffle:
            rng.shuffle(processed)

        pieces: list[Path] = []
        for i, seg in enumerate(processed):
            pieces.append(seg)
            if i < len(processed) - 1:
                pieces.append(static_result)

        concat_f = _concat_channel.submit(pieces, ch_out, cfg=c)
        concat_futures.append((ch_idx, ch_out, concat_f))

    for ch_idx, ch_out, concat_f in concat_futures:
        concat_f.result()
        print(f"  ch{ch_idx:02d}: done → {ch_out.name}")
        channel_files.append(ch_out)

    print(f"\n{n_channels} channels written to {out_dir}/")
    return channel_files


# ── CLI ──────────────────────────────────────────────────────────────────────

def _cli():
    import argparse

    parser = argparse.ArgumentParser(
        description="Build stooges CRT TV channel files for OBS composite.",
    )
    parser.add_argument("src", type=Path,
                        help="Source footage (e.g. tooth_will_out.mp4)")
    parser.add_argument("--shader-dir", type=Path, default=None)
    parser.add_argument("-n", "--n-channels", type=int, default=4,
                        help="Number of channel files to produce (default: 4)")
    parser.add_argument("--static-duration", type=float, default=0.3,
                        help="Static burst length in seconds (default: 0.3)")
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false",
                        default=True,
                        help="Disable per-channel segment shuffling")
    parser.add_argument("--n-shaders", type=int, default=3,
                        help="Shaders per pass in crush sandwich (default: 3)")
    parser.add_argument("--segment-counts", type=str, default="8",
                        help="Segments per channel: single int or comma-separated "
                             "list (e.g. '8,10,12,14,16')")
    parser.add_argument("--min-dur", type=float, default=1.0,
                        help="Minimum segment duration in seconds (default: 1.0)")
    parser.add_argument("--max-dur", type=float, default=30.0,
                        help="Maximum segment duration in seconds (default: 30.0)")
    parser.add_argument("--crush", type=float, default=0.95,
                        help="Crush intensity 0.0–1.0 (default: 0.95)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Master random seed for reproducibility")
    parser.add_argument("-o", "--output-dir", type=Path, default=None)

    args = parser.parse_args()
    cfg = Config()
    cfg.ensure_dirs()

    channels = stooges_channels(
        src=args.src,
        shader_dir=args.shader_dir,
        n_channels=args.n_channels,
        static_duration=args.static_duration,
        shuffle=args.shuffle,
        n_shaders=args.n_shaders,
        segment_counts=[int(x) for x in args.segment_counts.split(",")],
        min_dur=args.min_dur,
        max_dur=args.max_dur,
        crush_amount=args.crush,
        seed=args.seed,
        output_dir=args.output_dir,
        cfg=cfg,
    )

    for ch in channels:
        print(ch)


if __name__ == "__main__":
    _cli()
