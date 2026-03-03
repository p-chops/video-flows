"""
Show reel flow — channel-surfing through heterogeneous generator "shows".

Each show is a short (5–10s) generator clip rendered at a random complexity
level, so some are raw warped generators and others have crush, shaders,
time effects, etc. Shows are joined with random per-pair transitions.

Supports a human-in-the-loop workflow via CLI subcommands:

    # Preview recipes and save a manifest (no rendering)
    python -m pipeline.flows.show_reel plan -n 8 --seed 2222 ...

    # Edit the manifest, then render it
    python -m pipeline.flows.show_reel render work/reel_2222_manifest.json

    # One-shot: plan + render (the original behaviour)
    python -m pipeline.flows.show_reel run -n 8 --seed 2222 ...
"""

from __future__ import annotations

import contextvars
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from prefect import flow
from prefect.task_runners import ConcurrentTaskRunner

from ..config import Config
from ..recipe import (
    random_recipe, hash_recipe, recipe_to_dict, recipe_from_dict,
    FootageSource, GeneratorSource, StaticSource,
)
from ..tasks.transition import transition_sequence
from .brain_wipe import brain_wipe


def _fixup_recipe_sources(recipe, dur: float, rng: random.Random) -> None:
    """Apply source fixups to a recipe's lanes (in-place).

    Shared between _plan_shows and reroll to ensure consistent behavior:
    - Reject StaticSource → replace with GeneratorSource
    - Force method="random" for footage (scene detection ignores dur bounds)
    - Clamp per-segment duration to budget
    """
    for lane in recipe.lanes:
        if isinstance(lane.source, StaticSource):
            lane.source = GeneratorSource(
                min_dur=lane.source.min_dur,
                max_dur=lane.source.max_dur,
                n_warps=rng.randint(0, 2),
            )
        if isinstance(lane.source, FootageSource):
            lane.source.method = "random"
        seg_budget = dur / max(lane.n_segments, 1)
        lane.source.max_dur = min(lane.source.max_dur, seg_budget)
        lane.source.min_dur = min(lane.source.min_dur, seg_budget)


def _resolve_src(rng: random.Random, src: Optional[Path]) -> Optional[Path]:
    """If src is a directory, pick a random .mp4 from it. Otherwise return as-is."""
    if src is None:
        return None
    if src.is_dir():
        files = sorted(src.glob("*.mp4"))
        if not files:
            raise ValueError(f"No .mp4 files found in {src}")
        return rng.choice(files)
    return src


def _resolve_n_shows(n_shows: Optional[int], reel_dur: Optional[float],
                     min_dur: float, max_dur: float) -> int:
    """Resolve n_shows from explicit value or reel_dur target.

    Priority: explicit n_shows > computed from reel_dur > default 20.
    """
    if n_shows is not None:
        return n_shows
    if reel_dur is not None:
        avg_dur = (min_dur + max_dur) / 2
        return max(1, round(reel_dur / avg_dur))
    return 20


# ── Plan phase ───────────────────────────────────────────────────────────────

def _plan_shows(
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
    archetype: Optional[str] = None,
    output: Optional[Path] = None,
    cfg: Optional[Config] = None,
) -> dict:
    """Generate show recipes and return a manifest dict (no rendering)."""
    c = cfg or Config()
    c.ensure_dirs()
    rng = random.Random(seed)

    reel_seed = seed or rng.randint(0, 2**31)

    is_src_dir = src is not None and src.is_dir()

    print(f"═══ Show Reel (seed={reel_seed}) ═══")
    print(f"    {n_shows} shows, {min_dur}–{max_dur}s each")
    print(f"    complexity {min_complexity}–{max_complexity}")
    if src:
        if is_src_dir:
            n_files = len(list(src.glob("*.mp4")))
            print(f"    source: {src.name}/ ({n_files} files, {footage_ratio:.0%} footage)")
        else:
            print(f"    source: {src.name} ({footage_ratio:.0%} footage)")
    else:
        print(f"    generators only (no source footage)")
    print(f"    {width}×{height} @ 30fps")
    print(f"    random transitions ({transition_dur}s)\n")

    # Pre-assign footage vs generator per show — guarantees the ratio
    # instead of independent coin flips which can streak badly at small N
    if src:
        n_footage = round(n_shows * footage_ratio)
        # At extremes (0.0 or 1.0) honour the request exactly;
        # otherwise guarantee at least 1 of each kind.
        if footage_ratio > 0.0 and footage_ratio < 1.0:
            n_footage = max(1, min(n_footage, n_shows - 1))
        show_is_footage = [True] * n_footage + [False] * (n_shows - n_footage)
        rng.shuffle(show_is_footage)
    else:
        show_is_footage = [False] * n_shows

    shows = []
    for i in range(n_shows):
        show_seed = rng.randint(0, 2**31)
        complexity = rng.uniform(min_complexity, max_complexity)
        dur = rng.uniform(min_dur, max_dur)

        use_footage = show_is_footage[i]

        # Footage complexity band: floor 0.4 (enough processing to be
        # interesting) and cap 0.55 (preserve recognisable source).
        # Generators can go as high as the user wants.
        if use_footage:
            show_complexity = max(0.4, min(0.55, complexity))
        else:
            show_complexity = complexity

        show_src = _resolve_src(rng, src) if use_footage else None

        recipe = random_recipe(
            src=show_src,
            complexity=show_complexity,
            target_dur=dur,
            use_generators=False if use_footage else True,
            n_segments=1,
            use_transitions=False,
            seed=show_seed,
            archetype=archetype,
            width=width,
            height=height,
            packs=c.packs,
        )

        _fixup_recipe_sources(recipe, dur, rng)

        kind = "footage" if use_footage else "generator"
        src_tag = f" [{show_src.name}]" if (use_footage and is_src_dir and show_src) else ""
        n_lanes = len(recipe.lanes)
        print(f"  show {i:03d} (seed={show_seed}, {kind}{src_tag}, complexity={show_complexity:.2f}, "
              f"{dur:.1f}s, {n_lanes} lane{'s' if n_lanes > 1 else ''}):")
        for li, lane in enumerate(recipe.lanes):
            if n_lanes > 1:
                src_name = type(lane.source).__name__
                print(f"    lane {li} ({src_name}):")
            for step in lane.recipe:
                prefix = "      " if n_lanes > 1 else "    "
                print(f"{prefix}{step}")

        shows.append({
            "index": i,
            "seed": show_seed,
            "kind": kind,
            "complexity": round(show_complexity, 4),
            "duration": round(dur, 2),
            "recipe": recipe_to_dict(recipe),
        })

    out_path = output or c.output_dir / f"show_reel_{reel_seed}.mp4"

    manifest = {
        "seed": reel_seed,
        "transition_dur": transition_dur,
        "width": width,
        "height": height,
        "output": str(out_path),
        "shows": shows,
    }
    if archetype:
        manifest["archetype"] = archetype
    return manifest


# ── Render phase ─────────────────────────────────────────────────────────────

@flow(name="show-reel-render", log_prints=True,
      task_runner=ConcurrentTaskRunner(max_workers=4))
def show_reel_render(
    manifest: dict,
    output: Optional[Path] = None,
    cfg: Optional[Config] = None,
    cleanup: bool = True,
    motion_floor: float = 0.005,
    max_reroll: int = 2,
) -> Path:
    """Render a show reel from a manifest dict (or loaded JSON).

    motion_floor: minimum motion score (0–1). Shows below this are
                  considered stasis and rerolled with a new seed.
                  0.0 disables the check.
    max_reroll:   max reroll attempts per show before accepting.
    """
    c = cfg or Config()
    c.ensure_dirs()

    reel_seed = manifest["seed"]
    transition_dur = manifest["transition_dur"]
    width = manifest.get("width", 1280)
    height = manifest.get("height", 720)
    archetype_hint = manifest.get("archetype")
    shows = manifest["shows"]
    out = output or Path(manifest["output"])

    reel_tag = f"reel_{reel_seed}"
    work = c.work_dir / reel_tag
    work.mkdir(parents=True, exist_ok=True)

    rng = random.Random(reel_seed)

    print(f"═══ Rendering Show Reel (seed={reel_seed}, {len(shows)} shows) ═══")

    from ..tasks.color import probe_quality, should_reroll, QualityThresholds
    thresholds = QualityThresholds(motion_floor=motion_floor)
    quality_log = c.work_dir / "quality_log.jsonl"

    print(f"    quality gate: motion_floor={thresholds.motion_floor}, "
          f"max_reroll={max_reroll}")

    def _log_quality(
        show: dict, report, outcome: str, reason: str, attempt: int,
    ) -> None:
        """Append one JSONL row to the quality log for training data."""
        import json as _json
        row = {
            "seed": show["seed"],
            "index": show["index"],
            "kind": show.get("kind", "unknown"),
            "complexity": show.get("complexity", 0),
            "archetype": show.get("archetype") or archetype_hint,
            "features": report.to_dict(),
            "outcome": outcome,
            "reason": reason,
            "attempt": attempt,
        }
        with open(quality_log, "a") as f:
            f.write(_json.dumps(row) + "\n")

    def _render_show(show: dict, attempt: int = 0) -> Path:
        idx = show["index"]
        recipe = recipe_from_dict(show["recipe"])
        show_tag = hash_recipe(recipe)
        show_path = work / f"show_{idx:03d}_{show_tag}.mp4"

        attempt_msg = f" (attempt {attempt + 1})" if attempt > 0 else ""
        print(f"\n  rendering show {idx:03d}{attempt_msg} "
              f"(seed={show['seed']}, {show['kind']}, "
              f"complexity={show['complexity']:.2f}, {show['duration']:.1f}s)...")
        result = brain_wipe(recipe, output=show_path, cfg=c, cleanup=True)

        # Unified quality gate — single decode pass for all metrics.
        report = probe_quality(result, c)
        reroll, reason, fixable = should_reroll(report, thresholds)
        print(f"    quality: {report.summary()}")

        if reroll and attempt < max_reroll:
            _log_quality(show, report, "rerolled", reason, attempt)
            print(f"    REROLL: {reason}")

            if result.exists():
                result.unlink()

            # Reconstruct source path for recipe generation
            show_src = None
            if show["kind"] == "footage":
                lanes = show["recipe"].get("lanes", [])
                if lanes:
                    src_dict = lanes[0].get("source", {})
                    src_path = src_dict.get("path")
                    if src_path:
                        show_src = Path(src_path)

            new_seed = show["seed"] + attempt + 1
            reroll_rng = random.Random(new_seed)
            new_recipe = random_recipe(
                src=show_src,
                complexity=show["complexity"],
                target_dur=show["duration"],
                use_generators=(show["kind"] != "footage"),
                n_segments=1,
                use_transitions=False,
                seed=new_seed,
                archetype=archetype_hint,
                width=width,
                height=height,
                packs=c.packs,
            )
            _fixup_recipe_sources(new_recipe, show["duration"], reroll_rng)

            new_show = dict(
                show,
                seed=new_seed,
                recipe=recipe_to_dict(new_recipe),
            )
            return _render_show(new_show, attempt=attempt + 1)
        elif reroll:
            _log_quality(show, report, "kept_exhausted", reason, attempt)
            print(f"    quality issue ({reason}) but max rerolls reached, keeping")
        else:
            # Fixable — apply auto_levels brightness stretch
            if fixable:
                import subprocess
                lifted = work / f"show_{idx:03d}_lifted.mp4"
                subprocess.run([
                    c.ffmpeg_bin, "-y", "-loglevel", c.ffmpeg_loglevel,
                    "-i", str(result),
                    "-vf", "normalize=independence=0:smoothing=10:strength=0.8",
                    "-an",
                    *c.encode_args(),
                    str(lifted),
                ], check=True)
                result.unlink()
                lifted.rename(result)
                result = show_path
                _log_quality(show, report, "fixed", reason, attempt)
                print(f"    brightness fix applied (was {report.brightness:.3f})")
            else:
                _log_quality(show, report, "kept", reason, attempt)

        return result

    max_show_workers = min(c.max_parallel_shows, len(shows))
    if max_show_workers <= 1:
        show_clips = [_render_show(show) for show in shows]
    else:
        with ThreadPoolExecutor(max_workers=max_show_workers) as pool:
            # Each thread gets its own copy of the Prefect flow context
            # so brain_wipe subflows register as children of this flow.
            futures = {}
            for show in shows:
                ctx = contextvars.copy_context()
                futures[pool.submit(ctx.run, _render_show, show)] = show["index"]
            show_clips_map = {}
            for future in as_completed(futures):
                idx = futures[future]
                show_clips_map[idx] = future.result()
                print(f"  show {idx:03d} complete")
            show_clips = [show_clips_map[show["index"]] for show in shows]

    print(f"\n  all {len(shows)} shows rendered")

    print(f"\n  joining {len(show_clips)} shows with random transitions...")
    transition_sequence(
        show_clips, out,
        transition_type="random",
        duration=transition_dur,
        seed=rng.randint(0, 2**31),
        cfg=c,
    )

    # Cleanup show clips from work dir
    if cleanup:
        import shutil
        if work.exists():
            total_mb = sum(
                f.stat().st_size for f in work.rglob("*") if f.is_file()
            ) / (1024 * 1024)
            n_files = sum(1 for f in work.rglob("*") if f.is_file())
            shutil.rmtree(work)
            print(f"  cleanup: removed {n_files} work items ({total_mb:.0f} MB)")
    else:
        print(f"  intermediate shows retained in {work}/")

    print(f"\nOutput: {out}")
    return out


# ── One-shot flow (plan + render) ────────────────────────────────────────────

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
    archetype: Optional[str] = None,
    output: Optional[Path] = None,
    cleanup: bool = True,
    motion_floor: float = 0.005,
    max_reroll: int = 2,
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
    cleanup:        remove intermediate show clips after joining (default True)
    archetype:      force all shows to use this archetype (e.g. "deep_time")
    motion_floor:   stasis detection threshold (0.0 disables)
    max_reroll:     max reroll attempts per stasis show
    """
    c = cfg or Config()
    manifest = _plan_shows(
        n_shows=n_shows, min_dur=min_dur, max_dur=max_dur,
        min_complexity=min_complexity, max_complexity=max_complexity,
        transition_dur=transition_dur, width=width, height=height,
        src=src, footage_ratio=footage_ratio, seed=seed,
        archetype=archetype, output=output, cfg=c,
    )
    # Save manifest alongside the run for reproducibility
    manifest_path = c.work_dir / f"reel_{seed}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(f"  manifest saved: {manifest_path}")
    return show_reel_render(
        manifest, cfg=c, cleanup=cleanup,
        motion_floor=motion_floor, max_reroll=max_reroll,
    )


# ── Batch flow ───────────────────────────────────────────────────────────────

@flow(name="show-reel-batch", log_prints=True,
      task_runner=ConcurrentTaskRunner(max_workers=4))
def show_reel_batch(
    n_reels: int = 10,
    n_shows: int = 8,
    min_dur: float = 5.0,
    max_dur: float = 10.0,
    min_complexity: float = 0.4,
    max_complexity: float = 0.9,
    transition_dur: float = 0.5,
    width: int = 1280,
    height: int = 720,
    src: Optional[Path] = None,
    footage_ratio: float = 0.5,
    seed: Optional[int] = None,
    archetype: Optional[str] = None,
    cleanup: bool = True,
    motion_floor: float = 0.005,
    max_reroll: int = 2,
    cfg: Optional[Config] = None,
) -> list[Path]:
    """Generate multiple random show reels for curation.

    Each reel gets a unique random seed. Output files are named
    show_reel_batch_<batch_seed>_<index>.mp4.
    """
    c = cfg or Config()
    c.ensure_dirs()

    batch_rng = random.Random(seed)
    batch_seed = seed or batch_rng.randint(0, 2**31)
    batch_rng = random.Random(batch_seed)

    reel_seeds = [batch_rng.randint(0, 2**31) for _ in range(n_reels)]

    print(f"═══ Show Reel Batch (seed={batch_seed}, {n_reels} reels) ═══\n")

    results = []
    for i, reel_seed in enumerate(reel_seeds):
        out_path = c.output_dir / f"show_reel_batch_{batch_seed}_{i:03d}.mp4"
        print(f"──── Reel {i+1}/{n_reels} (seed={reel_seed}) → {out_path.name} ────")

        manifest = _plan_shows(
            n_shows=n_shows, min_dur=min_dur, max_dur=max_dur,
            min_complexity=min_complexity, max_complexity=max_complexity,
            transition_dur=transition_dur, width=width, height=height,
            src=src, footage_ratio=footage_ratio, seed=reel_seed,
            archetype=archetype, output=out_path, cfg=c,
        )
        if archetype:
            manifest["archetype"] = archetype
        result = show_reel_render(
            manifest, cfg=c, cleanup=cleanup,
            motion_floor=motion_floor, max_reroll=max_reroll,
        )
        results.append(result)
        print(f"  ✓ reel {i+1}/{n_reels} done: {result.name}\n")

    print(f"\n═══ Batch complete: {len(results)} reels in {c.output_dir}/ ═══")
    for r in results:
        print(f"  {r.name}")

    return results


# ── CLI ──────────────────────────────────────────────────────────────────────

def _add_plan_args(parser):
    """Add the shared show-reel arguments to a parser."""
    parser.add_argument("-n", "--n-shows", type=int, default=None,
                        help="Number of shows (auto-calculated from --reel-dur if omitted)")
    parser.add_argument("--reel-dur", type=float, default=None,
                        help="Target reel duration in seconds (auto-calculates n_shows)")
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
    parser.add_argument("--pack", action="append", dest="packs",
                        help="Restrict to specific shader packs (repeatable)")
    parser.add_argument("--archetype", type=str, default=None,
                        help="Force a specific archetype for all shows (e.g. deep_time)")
    parser.add_argument("-o", "--output", type=str, default=None)
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Retain intermediate show clips in work dir")
    parser.add_argument("--motion-floor", type=float, default=0.005,
                        help="Stasis detection threshold (0.0 disables)")
    parser.add_argument("--max-reroll", type=int, default=2,
                        help="Max reroll attempts per stasis show")


def main():
    import argparse
    import sys

    # If the first positional arg isn't a known subcommand, assume "run"
    _COMMANDS = {"plan", "render", "run", "batch"}
    if len(sys.argv) > 1 and sys.argv[1] not in _COMMANDS:
        sys.argv.insert(1, "run")

    parser = argparse.ArgumentParser(
        description="Show reel generator",
        usage="python -m pipeline.flows.show_reel {plan,render,run} ...",
    )
    sub = parser.add_subparsers(dest="command")

    # plan — generate recipes, save manifest, no rendering
    p_plan = sub.add_parser("plan", help="Preview recipes and save a manifest (no rendering)")
    _add_plan_args(p_plan)

    # render — load manifest and render
    p_render = sub.add_parser("render", help="Render a show reel from a manifest JSON")
    p_render.add_argument("manifest", type=str, help="Path to manifest JSON file")
    p_render.add_argument("-o", "--output", type=str, default=None,
                          help="Override output path from manifest")
    p_render.add_argument("--no-cleanup", action="store_true",
                          help="Retain intermediate show clips in work dir")
    p_render.add_argument("--motion-floor", type=float, default=0.005,
                          help="Stasis detection threshold (0.0 disables)")
    p_render.add_argument("--max-reroll", type=int, default=2,
                          help="Max reroll attempts per stasis show")
    p_render.add_argument("--pack", action="append", dest="packs",
                          help="Restrict to specific shader packs (repeatable)")

    # run — one-shot plan + render (original behaviour)
    p_run = sub.add_parser("run", help="Plan and render in one shot (default)")
    _add_plan_args(p_run)

    # batch — multiple random reels for curation
    p_batch = sub.add_parser("batch", help="Generate multiple random reels for curation")
    p_batch.add_argument("n_reels", type=int, help="Number of reels to generate")
    _add_plan_args(p_batch)

    args = parser.parse_args()

    packs = getattr(args, "packs", None)

    if args.command == "plan":
        cfg = Config(packs=packs)
        cfg.ensure_dirs()
        n_shows = _resolve_n_shows(args.n_shows, args.reel_dur,
                                   args.min_dur, args.max_dur)
        manifest = _plan_shows(
            n_shows=n_shows,
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
            archetype=args.archetype,
            output=Path(args.output) if args.output else None,
            cfg=cfg,
        )
        # Save manifest
        manifest_path = cfg.work_dir / f"reel_{manifest['seed']}_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"\n  manifest saved: {manifest_path}")
        print(f"  edit it, then: python -m pipeline.flows.show_reel render {manifest_path}")

    elif args.command == "render":
        manifest = json.loads(Path(args.manifest).read_text())
        out = Path(args.output) if args.output else None
        show_reel_render(
            manifest, output=out, cleanup=not args.no_cleanup,
            motion_floor=args.motion_floor, max_reroll=args.max_reroll,
            cfg=Config(packs=packs),
        )

    elif args.command == "run":
        out = Path(args.output) if args.output else None
        n_shows = _resolve_n_shows(args.n_shows, args.reel_dur,
                                   args.min_dur, args.max_dur)
        show_reel(
            n_shows=n_shows,
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
            archetype=args.archetype,
            output=out,
            cleanup=not args.no_cleanup,
            motion_floor=args.motion_floor,
            max_reroll=args.max_reroll,
            cfg=Config(packs=packs),
        )

    elif args.command == "batch":
        n_shows = _resolve_n_shows(args.n_shows, args.reel_dur,
                                   args.min_dur, args.max_dur)
        show_reel_batch(
            n_reels=args.n_reels,
            n_shows=n_shows,
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
            archetype=args.archetype,
            cleanup=not args.no_cleanup,
            motion_floor=args.motion_floor,
            max_reroll=args.max_reroll,
            cfg=Config(packs=packs),
        )


if __name__ == "__main__":
    main()
