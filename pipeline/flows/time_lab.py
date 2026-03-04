"""
Time Lab — temporal manipulation experiments.

Effects:
  scrub    — virtual playhead wanders through source at varying speeds
  drift    — fixed-length loop window slides slowly through source
  pingpong — short subsegment plays forward-backward continuously
  echo     — ghostly motion trails via frame accumulation
  patch    — temporal patchwork, random crops reveal 'now' over frozen past

Usage (Python):
    from pipeline.flows.time_lab import time_lab_scrub, time_lab_drift, time_lab_pingpong, time_lab_echo, time_lab_patch
    time_lab_scrub(Path("src.mp4"), smoothness=2.0, intensity=0.5, seed=42)
    time_lab_drift(Path("src.mp4"), loop_dur=0.5, seed=1)
    time_lab_pingpong(Path("src.mp4"), window=0.5, seed=1)
    time_lab_echo(Path("src.mp4"), delay=0.5, trail=0.8)
    time_lab_patch(Path("src.mp4"), patch_min=0.05, patch_max=0.4, seed=42)

CLI:
    python -m pipeline.flows.time_lab scrub src.mp4 --intensity 0.5 --seed 42
    python -m pipeline.flows.time_lab drift src.mp4 --loop-dur 0.5 --seed 1
    python -m pipeline.flows.time_lab pingpong src.mp4 --window 0.5 --seed 1
    python -m pipeline.flows.time_lab echo src.mp4 --delay 0.5 --trail 0.8
    python -m pipeline.flows.time_lab patch src.mp4 --patch-min 0.05 --patch-max 0.4 --seed 42
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from prefect import flow

from ..config import Config
from ..tasks.time import (
    drift_loop, echo_trail, ping_pong, time_patch, time_scrub, temporal_fft,
    temporal_gradient, axis_swap,
    temporal_morph, depth_slice, temporal_equalize,
    temporal_displace, spectral_remix, phase_scramble,
    datamosh, frame_quantize,
)


# ─── Flows ────────────────────────────────────────────────────────────────────

@flow(name="time-lab-scrub", log_prints=True)
def time_lab_scrub(
    src: Path,
    output: Optional[Path] = None,
    smoothness: float = 2.0,
    intensity: float = 0.5,
    seed: Optional[int] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """Random temporal scrubbing — playhead wanders at varying speeds."""
    c = cfg or Config()
    c.ensure_dirs()
    out = output or c.output_dir / "time_scrub.mp4"
    return time_scrub(src, out, smoothness=smoothness, intensity=intensity,
                      seed=seed, cfg=c)


@flow(name="time-lab-drift", log_prints=True)
def time_lab_drift(
    src: Path,
    output: Optional[Path] = None,
    loop_dur: float = 0.5,
    drift: Optional[float] = None,
    seed: Optional[int] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """Drift loop — looping window slides through the source."""
    c = cfg or Config()
    c.ensure_dirs()
    out = output or c.output_dir / "drift_loop.mp4"
    return drift_loop(src, out, loop_dur=loop_dur, drift=drift,
                      seed=seed, cfg=c)


@flow(name="time-lab-pingpong", log_prints=True)
def time_lab_pingpong(
    src: Path,
    output: Optional[Path] = None,
    window: float = 0.5,
    seed: Optional[int] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """Ping-pong — short subsegment breathes forward-backward."""
    c = cfg or Config()
    c.ensure_dirs()
    out = output or c.output_dir / "ping_pong.mp4"
    return ping_pong(src, out, window=window, seed=seed, cfg=c)


@flow(name="time-lab-echo", log_prints=True)
def time_lab_echo(
    src: Path,
    output: Optional[Path] = None,
    delay: float = 0.0,
    trail: float = 0.8,
    cfg: Optional[Config] = None,
) -> Path:
    """Echo trails — ghostly temporal echoes with configurable delay."""
    c = cfg or Config()
    c.ensure_dirs()
    out = output or c.output_dir / "echo_trail.mp4"
    return echo_trail(src, out, delay=delay, trail=trail, cfg=c)


@flow(name="time-lab-patch", log_prints=True)
def time_lab_patch(
    src: Path,
    output: Optional[Path] = None,
    patch_min: float = 0.05,
    patch_max: float = 0.4,
    seed: Optional[int] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """Temporal patchwork — random crops reveal 'now', rest frozen in past."""
    c = cfg or Config()
    c.ensure_dirs()
    out = output or c.output_dir / "time_patch.mp4"
    return time_patch(src, out, patch_min=patch_min, patch_max=patch_max,
                      seed=seed, cfg=c)


@flow(name="time-lab-temporal-fft", log_prints=True)
def time_lab_temporal_fft(
    src: Path,
    output: Optional[Path] = None,
    filter_type: str = "low_pass",
    cutoff_low: float = 0.1,
    cutoff_high: float = 0.5,
    preserve_dc: bool = True,
    seed: Optional[int] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """Temporal FFT filtering — manipulate frequency content along time axis."""
    c = cfg or Config()
    c.ensure_dirs()
    out = output or c.output_dir / "temporal_fft.mp4"
    return temporal_fft(src, out, filter_type=filter_type, cutoff_low=cutoff_low,
                        cutoff_high=cutoff_high, preserve_dc=preserve_dc,
                        seed=seed, cfg=c)


@flow(name="time-lab-temporal-gradient", log_prints=True)
def time_lab_temporal_gradient(
    src: Path,
    output: Optional[Path] = None,
    order: int = 1,
    seed: Optional[int] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """Temporal gradient — per-pixel temporal derivative."""
    c = cfg or Config()
    c.ensure_dirs()
    out = output or c.output_dir / "temporal_gradient.mp4"
    return temporal_gradient(src, out, order=order, seed=seed, cfg=c)



@flow(name="time-lab-axis-swap", log_prints=True)
def time_lab_axis_swap(
    src: Path,
    output: Optional[Path] = None,
    axis: str = "horizontal",
    seed: Optional[int] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """Axis swap — view the frame volume from the side."""
    c = cfg or Config()
    c.ensure_dirs()
    out = output or c.output_dir / "axis_swap.mp4"
    return axis_swap(src, out, axis=axis, seed=seed, cfg=c)


@flow(name="time-lab-temporal-morph", log_prints=True)
def time_lab_temporal_morph(
    src: Path,
    output: Optional[Path] = None,
    operation: str = "dilate",
    window: int = 5,
    seed: Optional[int] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """Temporal morphology — min/max along time axis."""
    c = cfg or Config()
    c.ensure_dirs()
    out = output or c.output_dir / "temporal_morph.mp4"
    return temporal_morph(src, out, operation=operation, window=window, seed=seed, cfg=c)


@flow(name="time-lab-depth-slice", log_prints=True)
def time_lab_depth_slice(
    src: Path,
    output: Optional[Path] = None,
    angle: float = 45.0,
    axis: str = "horizontal",
    seed: Optional[int] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """Depth slice — angled scan plane through spacetime."""
    c = cfg or Config()
    c.ensure_dirs()
    out = output or c.output_dir / "depth_slice.mp4"
    return depth_slice(src, out, angle=angle, axis=axis, seed=seed, cfg=c)


@flow(name="time-lab-temporal-equalize", log_prints=True)
def time_lab_temporal_equalize(
    src: Path,
    output: Optional[Path] = None,
    strength: float = 1.0,
    seed: Optional[int] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """Temporal equalize — per-pixel histogram equalization along time."""
    c = cfg or Config()
    c.ensure_dirs()
    out = output or c.output_dir / "temporal_equalize.mp4"
    return temporal_equalize(src, out, strength=strength, seed=seed, cfg=c)


@flow(name="time-lab-temporal-displace", log_prints=True)
def time_lab_temporal_displace(
    src: Path,
    output: Optional[Path] = None,
    amount: float = 0.5,
    channel: str = "luma",
    seed: Optional[int] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """Temporal displace — brightness drives time index lookup."""
    c = cfg or Config()
    c.ensure_dirs()
    out = output or c.output_dir / "temporal_displace.mp4"
    return temporal_displace(src, out, amount=amount, channel=channel, seed=seed, cfg=c)


@flow(name="time-lab-spectral-remix", log_prints=True)
def time_lab_spectral_remix(
    src: Path,
    output: Optional[Path] = None,
    mode: str = "swap",
    amount: float = 0.3,
    seed: Optional[int] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """Spectral remix — rearrange FFT frequency bins."""
    c = cfg or Config()
    c.ensure_dirs()
    out = output or c.output_dir / "spectral_remix.mp4"
    return spectral_remix(src, out, mode=mode, amount=amount, seed=seed, cfg=c)


@flow(name="time-lab-phase-scramble", log_prints=True)
def time_lab_phase_scramble(
    src: Path,
    output: Optional[Path] = None,
    amount: float = 1.0,
    seed: Optional[int] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """Phase scramble — randomize FFT phases, preserve magnitudes."""
    c = cfg or Config()
    c.ensure_dirs()
    out = output or c.output_dir / "phase_scramble.mp4"
    return phase_scramble(src, out, amount=amount, seed=seed, cfg=c)


@flow(name="time-lab-datamosh", log_prints=True)
def time_lab_datamosh(
    src: Path,
    output: Optional[Path] = None,
    refresh_interval: int = 30,
    blend: float = 0.0,
    seed: Optional[int] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """Simulated datamosh — freeze reference, warp with optical flow."""
    c = cfg or Config()
    c.ensure_dirs()
    out = output or c.output_dir / "datamosh.mp4"
    return datamosh(src, out, refresh_interval=refresh_interval, blend=blend, seed=seed, cfg=c)


@flow(name="time-lab-frame-quantize", log_prints=True)
def time_lab_frame_quantize(
    src: Path,
    output: Optional[Path] = None,
    n_levels: int = 8,
    mode: str = "luminance",
    seed: Optional[int] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """Frame quantize — reduce to K representative frames."""
    c = cfg or Config()
    c.ensure_dirs()
    out = output or c.output_dir / "frame_quantize.mp4"
    return frame_quantize(src, out, n_levels=n_levels, mode=mode, seed=seed, cfg=c)



# Keep backward-compatible alias
time_lab = time_lab_scrub


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _cli():
    import argparse

    parser = argparse.ArgumentParser(
        description="Time Lab — temporal manipulation experiments.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- scrub ---
    p_scrub = sub.add_parser("scrub",
                             help="Random temporal scrubbing")
    p_scrub.add_argument("src", type=Path,
                         help="Source video file")
    p_scrub.add_argument("-o", "--output", type=Path, default=None,
                         help="Output path (default: output/time_scrub.mp4)")
    p_scrub.add_argument("--smoothness", type=float, default=2.0,
                         help="Temporal scale of speed changes in seconds (default: 2.0)")
    p_scrub.add_argument("--intensity", type=float, default=0.5,
                         help="Speed variation magnitude, 0=normal 1=wild (default: 0.5)")
    p_scrub.add_argument("--seed", type=int, default=None,
                         help="Random seed for reproducibility")

    # --- drift ---
    p_drift = sub.add_parser("drift",
                             help="Drift loop — sliding loop window")
    p_drift.add_argument("src", type=Path,
                         help="Source video file")
    p_drift.add_argument("-o", "--output", type=Path, default=None,
                         help="Output path (default: output/drift_loop.mp4)")
    p_drift.add_argument("--loop-dur", type=float, default=0.5,
                         help="Seconds per loop cycle (default: 0.5)")
    p_drift.add_argument("--drift", type=float, default=None,
                         help="Seconds of drift per cycle; negative=backward (default: auto)")
    p_drift.add_argument("--seed", type=int, default=None,
                         help="Random seed (start position + auto-drift direction)")

    # --- pingpong ---
    p_pp = sub.add_parser("pingpong",
                          help="Ping-pong — breathing forward-backward loop")
    p_pp.add_argument("src", type=Path,
                      help="Source video file")
    p_pp.add_argument("-o", "--output", type=Path, default=None,
                      help="Output path (default: output/ping_pong.mp4)")
    p_pp.add_argument("--window", type=float, default=0.5,
                      help="Seconds of source to ping-pong (default: 0.5)")
    p_pp.add_argument("--seed", type=int, default=None,
                      help="Random seed (controls which subsegment)")

    # --- echo ---
    p_echo = sub.add_parser("echo",
                            help="Echo trails — ghostly motion smear")
    p_echo.add_argument("src", type=Path,
                        help="Source video file")
    p_echo.add_argument("-o", "--output", type=Path, default=None,
                        help="Output path (default: output/echo_trail.mp4)")
    p_echo.add_argument("--delay", type=float, default=0.0,
                        help="Echo delay in seconds; 0=motion blur (default: 0.0)")
    p_echo.add_argument("--trail", type=float, default=0.8,
                        help="Echo strength/feedback 0-1, higher=more ghost (default: 0.8)")

    # --- patch ---
    p_patch = sub.add_parser("patch",
                             help="Temporal patchwork — random crop reveals")
    p_patch.add_argument("src", type=Path,
                         help="Source video file")
    p_patch.add_argument("-o", "--output", type=Path, default=None,
                         help="Output path (default: output/time_patch.mp4)")
    p_patch.add_argument("--patch-min", type=float, default=0.05,
                         help="Min patch size as fraction of frame (default: 0.05)")
    p_patch.add_argument("--patch-max", type=float, default=0.4,
                         help="Max patch size as fraction of frame (default: 0.4)")
    p_patch.add_argument("--seed", type=int, default=None,
                         help="Random seed for reproducibility")

    # --- temporal-fft ---
    p_fft = sub.add_parser("temporal-fft",
                           help="Temporal FFT filtering")
    p_fft.add_argument("src", type=Path,
                       help="Source video file")
    p_fft.add_argument("-o", "--output", type=Path, default=None,
                       help="Output path (default: output/temporal_fft.mp4)")
    p_fft.add_argument("--filter-type", type=str, default="low_pass",
                       choices=["low_pass", "high_pass", "band_pass", "notch"],
                       help="Filter mode (default: low_pass)")
    p_fft.add_argument("--cutoff-low", type=float, default=0.1,
                       help="Low cutoff freq 0-1 (default: 0.1)")
    p_fft.add_argument("--cutoff-high", type=float, default=0.5,
                       help="High cutoff freq 0-1 (default: 0.5)")
    p_fft.add_argument("--no-preserve-dc", action="store_true",
                       help="Don't preserve DC bin (allows brightness shift)")
    p_fft.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")

    # --- temporal-gradient ---
    p_grad = sub.add_parser("temporal-gradient",
                            help="Temporal gradient — motion-only output")
    p_grad.add_argument("src", type=Path,
                        help="Source video file")
    p_grad.add_argument("-o", "--output", type=Path, default=None,
                        help="Output path (default: output/temporal_gradient.mp4)")
    p_grad.add_argument("--order", type=int, default=1, choices=[1, 2],
                        help="Derivative order: 1=velocity, 2=acceleration (default: 1)")
    p_grad.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")

    # --- axis-swap ---
    p_swap = sub.add_parser("axis-swap",
                            help="Axis swap — view volume from the side")
    p_swap.add_argument("src", type=Path,
                        help="Source video file")
    p_swap.add_argument("-o", "--output", type=Path, default=None,
                        help="Output path (default: output/axis_swap.mp4)")
    p_swap.add_argument("--axis", type=str, default="horizontal",
                        choices=["horizontal", "vertical"],
                        help="Which axis to swap with time (default: horizontal)")
    p_swap.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")

    # --- temporal-morph ---
    p_morph = sub.add_parser("temporal-morph",
                              help="Temporal morphology (dilate/erode/open/close)")
    p_morph.add_argument("src", type=Path, help="Source video file")
    p_morph.add_argument("-o", "--output", type=Path, default=None)
    p_morph.add_argument("--operation", type=str, default="dilate",
                          choices=["dilate", "erode", "open", "close"])
    p_morph.add_argument("--window", type=int, default=5,
                          help="Kernel size in frames (default: 5)")
    p_morph.add_argument("--seed", type=int, default=None)

    # --- depth-slice ---
    p_dslice = sub.add_parser("depth-slice",
                               help="Angled scan plane through spacetime")
    p_dslice.add_argument("src", type=Path, help="Source video file")
    p_dslice.add_argument("-o", "--output", type=Path, default=None)
    p_dslice.add_argument("--angle", type=float, default=45.0,
                           help="Slice angle in degrees (default: 45)")
    p_dslice.add_argument("--axis", type=str, default="horizontal",
                           choices=["horizontal", "vertical"])
    p_dslice.add_argument("--seed", type=int, default=None)

    # --- temporal-equalize ---
    p_teq = sub.add_parser("temporal-equalize",
                            help="Per-pixel histogram equalization along time")
    p_teq.add_argument("src", type=Path, help="Source video file")
    p_teq.add_argument("-o", "--output", type=Path, default=None)
    p_teq.add_argument("--strength", type=float, default=1.0,
                        help="Blend 0=original 1=full equalize (default: 1.0)")
    p_teq.add_argument("--seed", type=int, default=None)

    # --- temporal-displace ---
    p_tdisp = sub.add_parser("temporal-displace",
                              help="Brightness-driven time index warp")
    p_tdisp.add_argument("src", type=Path, help="Source video file")
    p_tdisp.add_argument("-o", "--output", type=Path, default=None)
    p_tdisp.add_argument("--amount", type=float, default=0.5,
                          help="Displacement strength 0-1 (default: 0.5)")
    p_tdisp.add_argument("--channel", type=str, default="luma",
                          choices=["luma", "r", "g", "b"])
    p_tdisp.add_argument("--seed", type=int, default=None)

    # --- spectral-remix ---
    p_sremix = sub.add_parser("spectral-remix",
                               help="Rearrange FFT frequency bins")
    p_sremix.add_argument("src", type=Path, help="Source video file")
    p_sremix.add_argument("-o", "--output", type=Path, default=None)
    p_sremix.add_argument("--mode", type=str, default="swap",
                           choices=["swap", "reverse", "rotate", "shuffle"])
    p_sremix.add_argument("--amount", type=float, default=0.3,
                           help="Blend strength 0=original 1=full rearrangement (default: 0.3)")
    p_sremix.add_argument("--seed", type=int, default=None)

    # --- phase-scramble ---
    p_pscr = sub.add_parser("phase-scramble",
                             help="Randomize FFT phases, keep magnitudes")
    p_pscr.add_argument("src", type=Path, help="Source video file")
    p_pscr.add_argument("-o", "--output", type=Path, default=None)
    p_pscr.add_argument("--amount", type=float, default=1.0,
                         help="Scramble strength 0-1 (default: 1.0)")
    p_pscr.add_argument("--seed", type=int, default=None)

    # --- datamosh ---
    p_dm = sub.add_parser("datamosh",
                           help="Simulated datamosh — freeze reference, warp with flow")
    p_dm.add_argument("src", type=Path, help="Source video file")
    p_dm.add_argument("-o", "--output", type=Path, default=None)
    p_dm.add_argument("--refresh-interval", type=int, default=30,
                       help="Frames between reference refresh (default: 30)")
    p_dm.add_argument("--blend", type=float, default=0.0,
                       help="Blend real pixels in 0-1 (default: 0.0)")
    p_dm.add_argument("--seed", type=int, default=None)

    # --- frame-quantize ---
    p_fq = sub.add_parser("frame-quantize",
                           help="Reduce to K representative frames")
    p_fq.add_argument("src", type=Path, help="Source video file")
    p_fq.add_argument("-o", "--output", type=Path, default=None)
    p_fq.add_argument("--n-levels", type=int, default=8,
                       help="Number of representative frames (default: 8)")
    p_fq.add_argument("--mode", type=str, default="luminance",
                       choices=["luminance", "color"],
                       help="Clustering metric (default: luminance)")
    p_fq.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()
    cfg = Config()
    cfg.ensure_dirs()

    if args.command == "scrub":
        time_lab_scrub(
            src=args.src, output=args.output,
            smoothness=args.smoothness, intensity=args.intensity,
            seed=args.seed, cfg=cfg,
        )
    elif args.command == "drift":
        time_lab_drift(
            src=args.src, output=args.output,
            loop_dur=args.loop_dur, drift=args.drift,
            seed=args.seed, cfg=cfg,
        )
    elif args.command == "pingpong":
        time_lab_pingpong(
            src=args.src, output=args.output,
            window=args.window, seed=args.seed,
            cfg=cfg,
        )
    elif args.command == "echo":
        time_lab_echo(
            src=args.src, output=args.output,
            delay=args.delay, trail=args.trail, cfg=cfg,
        )
    elif args.command == "patch":
        time_lab_patch(
            src=args.src, output=args.output,
            patch_min=args.patch_min, patch_max=args.patch_max,
            seed=args.seed, cfg=cfg,
        )
    elif args.command == "temporal-fft":
        time_lab_temporal_fft(
            src=args.src, output=args.output,
            filter_type=args.filter_type,
            cutoff_low=args.cutoff_low, cutoff_high=args.cutoff_high,
            preserve_dc=not args.no_preserve_dc,
            seed=args.seed, cfg=cfg,
        )
    elif args.command == "temporal-gradient":
        time_lab_temporal_gradient(
            src=args.src, output=args.output,
            order=args.order, seed=args.seed, cfg=cfg,
        )
    elif args.command == "axis-swap":
        time_lab_axis_swap(
            src=args.src, output=args.output,
            axis=args.axis, seed=args.seed, cfg=cfg,
        )
    elif args.command == "temporal-morph":
        time_lab_temporal_morph(
            src=args.src, output=args.output,
            operation=args.operation, window=args.window,
            seed=args.seed, cfg=cfg,
        )
    elif args.command == "depth-slice":
        time_lab_depth_slice(
            src=args.src, output=args.output,
            angle=args.angle, axis=args.axis,
            seed=args.seed, cfg=cfg,
        )
    elif args.command == "temporal-equalize":
        time_lab_temporal_equalize(
            src=args.src, output=args.output,
            strength=args.strength, seed=args.seed, cfg=cfg,
        )
    elif args.command == "temporal-displace":
        time_lab_temporal_displace(
            src=args.src, output=args.output,
            amount=args.amount, channel=args.channel,
            seed=args.seed, cfg=cfg,
        )
    elif args.command == "spectral-remix":
        time_lab_spectral_remix(
            src=args.src, output=args.output,
            mode=args.mode, amount=args.amount,
            seed=args.seed, cfg=cfg,
        )
    elif args.command == "phase-scramble":
        time_lab_phase_scramble(
            src=args.src, output=args.output,
            amount=args.amount, seed=args.seed, cfg=cfg,
        )
    elif args.command == "datamosh":
        time_lab_datamosh(
            src=args.src, output=args.output,
            refresh_interval=args.refresh_interval, blend=args.blend,
            seed=args.seed, cfg=cfg,
        )
    elif args.command == "frame-quantize":
        time_lab_frame_quantize(
            src=args.src, output=args.output,
            n_levels=args.n_levels, mode=args.mode,
            seed=args.seed, cfg=cfg,
        )


if __name__ == "__main__":
    _cli()
