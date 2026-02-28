"""
Show reel flow — channel-surfing through heterogeneous generator "shows".

Each show is a short (5–10s) generator clip rendered at a random complexity
level, so some are raw warped generators and others have crush, shaders,
time effects, etc. Shows are joined with random per-pair transitions.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

from prefect import flow
from prefect.task_runners import ConcurrentTaskRunner

from ..config import Config
from ..recipe import (
    random_recipe, hash_recipe, print_recipe,
    GeneratorSource, StaticSource,
)
from ..tasks.transition import transition_sequence
from .brain_wipe import brain_wipe


@flow(name="show-reel", log_prints=True,
      task_runner=ConcurrentTaskRunner(max_workers=4))
def show_reel(
    n_shows: int = 20,
    min_dur: float = 5.0,
    max_dur: float = 10.0,
    min_complexity: float = 0.1,
    max_complexity: float = 0.6,
    transition_dur: float = 0.5,
    width: int = 1280,
    height: int = 720,
    src: Optional[Path] = None,
    footage_ratio: float = 0.4,
    seed: Optional[int] = None,
    output: Optional[Path] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """
    Generate a show reel: N short clips at varying complexity,
    joined with random transitions.

    n_shows:        number of shows (segments)
    min_dur/max_dur: duration range for each show (seconds)
    min_complexity:  lowest complexity for a show
    max_complexity:  highest complexity for a show
    transition_dur:  transition duration between shows (seconds)
    src:            optional source footage — when provided, some shows use it
    footage_ratio:  probability each show uses footage vs generator (0.0–1.0)
    """
    c = cfg or Config()
    c.ensure_dirs()
    rng = random.Random(seed)

    reel_seed = seed or rng.randint(0, 2**31)
    reel_tag = f"reel_{reel_seed}"
    work = c.work_dir / reel_tag
    work.mkdir(parents=True, exist_ok=True)

    # When source footage is provided, match its resolution so all shows
    # (footage and generator) share the same dimensions for transitions
    if src:
        from ..ffmpeg import probe as ffprobe
        src_info = ffprobe(src, c)
        width = src_info.width
        height = src_info.height

    print(f"═══ Show Reel (seed={reel_seed}) ═══")
    print(f"    {n_shows} shows, {min_dur}–{max_dur}s each")
    print(f"    complexity {min_complexity}–{max_complexity}")
    if src:
        print(f"    source: {src.name} ({footage_ratio:.0%} footage)")
    else:
        print(f"    generators only (no source footage)")
    print(f"    {width}×{height} @ 30fps")
    print(f"    random transitions ({transition_dur}s)\n")

    # Pre-assign footage vs generator per show — guarantees the ratio
    # instead of independent coin flips which can streak badly at small N
    if src:
        n_footage = round(n_shows * footage_ratio)
        n_footage = max(1, min(n_footage, n_shows - 1))  # at least 1 of each
        show_is_footage = [True] * n_footage + [False] * (n_shows - n_footage)
        rng.shuffle(show_is_footage)
    else:
        show_is_footage = [False] * n_shows

    # Generate all recipes upfront, then render sequentially as subflows
    show_configs = []
    for i in range(n_shows):
        show_seed = rng.randint(0, 2**31)
        complexity = rng.uniform(min_complexity, max_complexity)
        dur = rng.uniform(min_dur, max_dur)

        use_footage = show_is_footage[i]

        # Footage needs higher complexity to look interesting — at low
        # complexity it's barely processed (1 mild crush + 1 shader).
        # Floor footage at 0.4 so it always gets a proper crush sandwich.
        show_complexity = max(complexity, 0.4) if use_footage else complexity

        recipe = random_recipe(
            src=src if use_footage else None,
            complexity=show_complexity,
            target_dur=dur,
            use_generators=True,
            n_segments=1,
            use_transitions=False,
            seed=show_seed,
        )
        recipe.width = width
        recipe.height = height

        # Reject StaticSource in any lane — TV noise is boring. Replace
        # with GeneratorSource. Also clamp durations to the show's target.
        for lane in recipe.lanes:
            if isinstance(lane.source, StaticSource):
                lane.source = GeneratorSource(
                    min_dur=lane.source.min_dur,
                    max_dur=lane.source.max_dur,
                    n_warps=rng.randint(0, 2),
                )
            lane.source.max_dur = min(lane.source.max_dur, dur)
            lane.source.min_dur = min(lane.source.min_dur, dur)

        show_tag = hash_recipe(recipe)
        show_path = work / f"show_{i:03d}_{show_tag}.mp4"

        kind = "footage" if use_footage else "generator"
        n_lanes = len(recipe.lanes)
        print(f"  show {i:03d} (seed={show_seed}, {kind}, complexity={show_complexity:.2f}, "
              f"{dur:.1f}s, {n_lanes} lane{'s' if n_lanes > 1 else ''}):")
        for li, lane in enumerate(recipe.lanes):
            if n_lanes > 1:
                src_name = type(lane.source).__name__
                print(f"    lane {li} ({src_name}):")
            for step in lane.recipe:
                prefix = "      " if n_lanes > 1 else "    "
                print(f"{prefix}{step}")

        show_configs.append((recipe, show_path))

    # Render each show as a subflow
    show_clips = []
    for i, (recipe, show_path) in enumerate(show_configs):
        print(f"\n  rendering show {i:03d}...")
        result = brain_wipe(recipe, output=show_path, cfg=c, cleanup=True)
        show_clips.append(result)

    print(f"\n  all {n_shows} shows rendered")

    # Join with random transitions
    out = output or c.output_dir / f"show_reel_{reel_seed}.mp4"
    print(f"\n  joining {len(show_clips)} shows with random transitions...")
    transition_sequence(
        show_clips, out,
        transition_type="random",
        duration=transition_dur,
        seed=rng.randint(0, 2**31),
        cfg=c,
    )

    # Cleanup show clips from work dir
    import shutil
    if work.exists():
        total_mb = sum(
            f.stat().st_size for f in work.rglob("*") if f.is_file()
        ) / (1024 * 1024)
        n_files = sum(1 for f in work.rglob("*") if f.is_file())
        shutil.rmtree(work)
        print(f"  cleanup: removed {n_files} work items ({total_mb:.0f} MB)")

    print(f"\nOutput: {out}")
    return out


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Show reel generator")
    parser.add_argument("-n", "--n-shows", type=int, default=20,
                        help="Number of shows")
    parser.add_argument("--min-dur", type=float, default=5.0,
                        help="Minimum show duration (seconds)")
    parser.add_argument("--max-dur", type=float, default=10.0,
                        help="Maximum show duration (seconds)")
    parser.add_argument("--min-complexity", type=float, default=0.1,
                        help="Minimum complexity per show")
    parser.add_argument("--max-complexity", type=float, default=0.6,
                        help="Maximum complexity per show")
    parser.add_argument("--transition-dur", type=float, default=0.5,
                        help="Transition duration (seconds)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--src", type=str, default=None,
                        help="Source footage (optional — some shows will use it)")
    parser.add_argument("--footage-ratio", type=float, default=0.4,
                        help="Fraction of shows that use footage (0.0–1.0)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("-o", "--output", type=str, default=None)
    args = parser.parse_args()

    out = Path(args.output) if args.output else None
    show_reel(
        n_shows=args.n_shows,
        min_dur=args.min_dur,
        max_dur=args.max_dur,
        min_complexity=args.min_complexity,
        max_complexity=args.max_complexity,
        transition_dur=args.transition_dur,
        width=args.width,
        height=args.height,
        src=Path(args.src) if args.src else None,
        footage_ratio=args.footage_ratio,
        seed=args.seed,
        output=out,
    )


if __name__ == "__main__":
    main()
