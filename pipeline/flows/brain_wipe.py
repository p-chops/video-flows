"""
Brain Wipe flows — apply warp and distortion shaders to video.

Two flows:

  warp_chain         Apply a chain of warp shaders to source footage.
                     Category-aware: filters library to "Warp" / "Brain Wipe"
                     shaders so glitch/color shaders stay separate.

  brain_wipe_render  Pre-render a long-form brain wipe sequence for streaming.
                     Generator shaders (plasma, tunnel, chladni, etc.)
                     synthesize content from scratch — no source footage needed.
                     Optional warp shaders can be chained after each generator.

Both flows accept an explicit shader_categories list for filtering, or fall
back to ["Warp", "Brain Wipe"] by default. Pass shader_categories=None to
use the full library (same behaviour as shader_lab / crush_lab).

Usage (Python):
    from pipeline.flows.brain_wipe import warp_chain, brain_wipe_render
    from pathlib import Path

    warp_chain(src=Path("source/footage.mp4"), n_shaders=2, seed=7)

    brain_wipe_render(
        n_segments=12,
        segment_dur=20.0,
        min_shaders=2,
        max_shaders=6,
        seed=42,
    )

CLI:
    python -m pipeline.flows.brain_wipe warp-chain source/footage.mp4 --n-shaders 2
    python -m pipeline.flows.brain_wipe brain-wipe-render -n 12 --seed 42
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Optional

from prefect import flow, task
from prefect.task_runners import ConcurrentTaskRunner

from ..config import Config
from ..ffmpeg import probe
from ..isf import ISFShader, load_shader_dir
from ..tasks import (
    apply_shader_stack,
    concat_clips,
    shuffle_clips,
    normalize_levels,
    generate_solid,
)


# ─── Utilities ───────────────────────────────────────────────────────────────

def filter_shaders(
    shaders: dict[str, ISFShader],
    categories: Optional[list[str]] = None,
    has_image_input: Optional[bool] = None,
) -> dict[str, ISFShader]:
    """
    Filter a shader dict by ISF category tags and/or image-input presence.

    Parameters
    ----------
    shaders         : dict from load_shader_dir
    categories      : keep only shaders whose CATEGORIES list shares at least
                      one entry with this list. None = no category filter.
    has_image_input : True  = keep only video warpers (have inputImage)
                      False = keep only generators (no inputImage)
                      None  = no filter
    """
    result = {}
    for name, shader in shaders.items():
        if categories is not None:
            shader_cats = {c.lower() for c in shader.categories}
            want_cats = {c.lower() for c in categories}
            if not shader_cats & want_cats:
                continue
        if has_image_input is not None:
            is_warper = len(shader.image_inputs) > 0
            if has_image_input != is_warper:
                continue
        result[name] = shader
    return result


def randomise_params(
    shader: ISFShader,
    rng: random.Random,
    pin_defaults: Optional[set[str]] = None,
) -> dict[str, float]:
    """Return a dict of param overrides with float params randomised.

    Parameters in pin_defaults are left at their shader-declared DEFAULT
    instead of being randomised. Useful for level-control params like
    brightness / desaturate that destroy output when randomised.
    """
    pinned = pin_defaults or set()
    params: dict[str, float] = {}
    for inp in shader.param_inputs:
        if inp.type == "float" and inp.min is not None and inp.max is not None:
            if inp.name in pinned and inp.default is not None:
                params[inp.name] = inp.default
            else:
                params[inp.name] = round(
                    inp.min + rng.random() * (inp.max - inp.min), 3
                )
    return params


# Params that control output levels rather than creative shape.
# Left at shader defaults to preserve brightness through chains.
LEVEL_PARAMS = {"brightness", "bg_brightness", "desaturate"}


def pick_shader_stack(
    shader_pool: dict[str, ISFShader],
    n: int,
    rng: random.Random,
    pin_defaults: Optional[set[str]] = None,
) -> tuple[list[Path], dict[str, dict[str, float]]]:
    """
    Pick n shaders from pool, randomise their float params.
    Returns (shader_paths, param_overrides).
    """
    count = min(n, len(shader_pool))
    chosen_names = rng.sample(list(shader_pool.keys()), count)
    chosen = [shader_pool[name] for name in chosen_names]
    paths = [s.path for s in chosen]
    overrides: dict[str, dict[str, float]] = {}
    for shader in chosen:
        p = randomise_params(shader, rng, pin_defaults=pin_defaults)
        if p:
            overrides[shader.path.stem] = p
    return paths, overrides


def print_stack(shaders: list[Path],
                 overrides: dict[str, dict[str, float]],
                 indent: str = "    ") -> None:
    for i, path in enumerate(shaders):
        params = overrides.get(path.stem, {})
        param_str = "  ".join(f"{k}={v}" for k, v in params.items())
        print(f"{indent}{i+1}. {path.stem:<28s} {param_str}")


# ─── Flow 1: warp_chain ───────────────────────────────────────────────────────

@flow(name="warp-chain", log_prints=True,
      task_runner=ConcurrentTaskRunner(max_workers=4))
def warp_chain(
    src: Path,
    shader_dir: Optional[Path] = None,
    shader_paths: Optional[list[Path]] = None,
    shader_categories: Optional[list[str]] = None,
    n_shaders: int = 2,
    warpers_only: bool = True,
    output: Optional[Path] = None,
    normalize: bool = True,
    seed: Optional[int] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """
    Apply a chain of warp/distortion shaders to source footage.

    By default, filters the shader library to shaders tagged "Warp" or
    "Brain Wipe", and further restricts to video warpers (shaders with an
    inputImage input) so generators (plasma, tunnel, chladni) are excluded.

    Parameters
    ----------
    src               : source footage
    shader_dir        : ISF shader directory (defaults to cfg.shader_dir)
    shader_paths      : explicit list of shader paths — skips library loading
                        and random selection entirely
    shader_categories : category filter (default: ["Warp", "Brain Wipe"])
    n_shaders         : how many shaders to pick randomly (ignored if
                        shader_paths is provided)
    warpers_only      : if True, exclude generator shaders (no inputImage)
    output            : output path (default: output/warp_chain.mp4)
    normalize         : normalize output levels (default: True)
    seed              : random seed for reproducibility
    cfg               : pipeline Config
    """
    c = cfg or Config()
    c.ensure_dirs()
    out = output or c.output_dir / "warp_chain.mp4"
    s_dir = shader_dir or c.shader_dir

    info = probe(src, c)
    print(
        f"Source: {src.name}  "
        f"{info.width}x{info.height} @ {info.fps:.2f}fps  "
        f"{info.duration:.1f}s"
    )
    if c.default_video_bitrate is None and info.bitrate > 0:
        c.default_video_bitrate = info.bitrate

    # ── Resolve shader stack ──────────────────────────────────────────────

    if shader_paths is not None:
        # Explicit list — use as-is, no randomisation
        paths = shader_paths
        overrides: dict[str, dict[str, float]] = {}
        print(f"\nExplicit shader stack ({len(paths)} shaders):")
        print_stack(paths, overrides)

    else:
        # Load library and filter
        cats = shader_categories if shader_categories is not None \
               else ["Warp", "Brain Wipe"]
        all_shaders = load_shader_dir(s_dir)
        pool = filter_shaders(
            all_shaders,
            categories=cats if cats else None,
            has_image_input=True if warpers_only else None,
        )

        if not pool:
            # Fall back to full library if filter yields nothing
            print(
                f"Warning: no shaders matched categories={cats} in {s_dir}. "
                f"Using full library ({len(all_shaders)} shaders)."
            )
            pool = all_shaders

        print(
            f"Shader pool: {len(pool)} shaders "
            f"(categories={cats}, warpers_only={warpers_only})"
        )

        rng = random.Random(seed)
        paths, overrides = pick_shader_stack(pool, n_shaders, rng)
        print(f"\nSelected stack ({len(paths)} shaders, seed={seed}):")
        print_stack(paths, overrides)

    # ── Apply shader stack ────────────────────────────────────────────────

    work_path = c.work_dir / "warp_chain_raw.mp4"
    apply_shader_stack(src, work_path, paths,
                       param_overrides=overrides, cfg=c)

    # ── Normalize ─────────────────────────────────────────────────────────

    if normalize:
        print("Normalizing levels...")
        normalize_levels(work_path, out, cfg=c)
    else:
        import shutil
        shutil.copy2(work_path, out)

    print(f"\nOutput: {out}")
    return out


# ─── Flow 2: brain_wipe_render ───────────────────────────────────────────────

@flow(name="brain-wipe-render", log_prints=True,
      task_runner=ConcurrentTaskRunner(max_workers=4))
def brain_wipe_render(
    shader_dir: Optional[Path] = None,
    shader_categories: Optional[list[str]] = None,
    n_segments: int = 8,
    segment_dur: float = 20.0,
    min_shaders: int = 2,
    max_shaders: int = 6,
    width: int = 1920,
    height: int = 1080,
    fps: float = 30.0,
    shuffle: bool = True,
    normalize: bool = True,
    seed: Optional[int] = None,
    output: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """
    Pre-render a long-form brain wipe sequence for streaming.

    Each segment is rendered through a randomly selected generator shader
    (plasma, tunnel, chladni, etc. — shaders with no inputImage that
    synthesize content from scratch). No source footage is needed —
    solid black placeholder clips drive the frame loop while generators
    synthesize all visual content.

    Optionally, warp shaders can be chained after the generator to further
    distort the synthesized imagery.

    Segments are normalized, optionally shuffled, and concatenated into a
    single output suitable for use as a stream asset or OBS media source.

    Parameters
    ----------
    shader_dir        : ISF shader directory (defaults to cfg.shader_dir)
    shader_categories : categories to filter from (default: ["Warp", "Brain Wipe"])
    n_segments        : number of segments to process (default: 8)
    segment_dur       : duration of each segment in seconds (default: 20.0)
    min_shaders       : minimum total shaders per segment (default: 2)
    max_shaders       : maximum total shaders per segment (default: 6)
    width             : output width in pixels (default: 1920)
    height            : output height in pixels (default: 1080)
    fps               : output frame rate (default: 30.0)
    shuffle           : shuffle segment order in final concat (default: True)
    normalize         : normalize levels on each segment (default: True)
    seed              : master random seed
    output            : final output path (default: output/brain_wipe_render.mp4)
    output_dir        : directory for per-segment previews (optional)
    cfg               : pipeline Config
    """
    c = cfg or Config()
    c.ensure_dirs()
    out = output or c.output_dir / "brain_wipe_render.mp4"
    s_dir = shader_dir or c.shader_dir
    cats = shader_categories if shader_categories is not None \
           else ["Warp", "Brain Wipe", "Generator"]

    print(f"Output format: {width}x{height} @ {fps:.2f}fps")

    # ── Load and partition shader library ────────────────────────────────

    all_shaders = load_shader_dir(s_dir)
    if not all_shaders:
        raise ValueError(f"No .fs shaders found in {s_dir}")

    # Warpers: video warp shaders (have inputImage)
    warpers = filter_shaders(
        all_shaders,
        categories=cats if cats else None,
        has_image_input=True,
    )
    # Generators: synthesize from scratch (no inputImage)
    generators = filter_shaders(
        all_shaders,
        categories=cats if cats else None,
        has_image_input=False,
    )

    print(
        f"Shader library: {len(all_shaders)} total  "
        f"warpers={len(warpers)}  generators={len(generators)}"
    )

    if not generators:
        raise ValueError(
            f"No generator shaders (no inputImage) found in {s_dir} "
            f"matching categories={cats}. "
            f"Generator shaders are required for brain_wipe_render."
        )

    # ── Generate placeholder segments ────────────────────────────────────

    clip_dir = c.work_dir / "brain_wipe_clips"
    clip_dir.mkdir(parents=True, exist_ok=True)
    segments = []
    for i in range(n_segments):
        seg_path = clip_dir / f"solid_{i:03d}.mp4"
        generate_solid(seg_path, segment_dur, width=width, height=height,
                       fps=fps, cfg=c)
        segments.append(seg_path)
    print(f"\n{len(segments)} placeholder segments generated ({segment_dur:.0f}s each)")

    # ── Preview dir for per-segment output ───────────────────────────────

    seg_out_dir = output_dir or (c.output_dir / "brain_wipe_segments")
    seg_out_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Submit shader stacks ────────────────────────────────────

    rng = random.Random(seed)
    futures = []
    raw_paths = []
    recipes = []

    print(f"\nSubmitting {len(segments)} shader stacks "
          f"({min_shaders}-{max_shaders} shaders per segment)...\n")

    for i, seg in enumerate(segments):
        seg_rng = random.Random(rng.randint(0, 2 ** 31))

        # Random total stack depth for this segment
        stack_depth = seg_rng.randint(min_shaders, max_shaders)
        n_warps = stack_depth - 1

        # Build shader chain: generator first, then warps
        chain_paths: list[Path] = []
        chain_overrides: dict[str, dict[str, float]] = {}

        # Always pick a generator — pin level params to defaults
        gen_paths, gen_overrides = pick_shader_stack(
            generators, 1, seg_rng, pin_defaults=LEVEL_PARAMS,
        )
        chain_paths.extend(gen_paths)
        chain_overrides.update(gen_overrides)

        # Warp shaders after the generator
        # Pin brightness/desaturate to defaults so warps don't crush levels
        if n_warps > 0 and warpers:
            warp_paths, warp_overrides = pick_shader_stack(
                warpers, n_warps, seg_rng,
                pin_defaults=LEVEL_PARAMS,
            )
            chain_paths.extend(warp_paths)
            chain_overrides.update(warp_overrides)

        # Recipe hash for filename
        recipe_str = ";".join(
            p.stem + ":" + ",".join(
                f"{k}={v}"
                for k, v in sorted(chain_overrides.get(p.stem, {}).items())
            )
            for p in chain_paths
        )
        tag = hashlib.sha1(recipe_str.encode()).hexdigest()[:8]

        print(f"  seg {i:03d} [{tag}]:")
        print_stack(chain_paths, chain_overrides, indent="    ")

        raw = c.work_dir / f"bw_{i:03d}_{tag}_raw.mp4"
        raw_paths.append(raw)
        recipes.append((i, tag, chain_paths, chain_overrides))

        future = apply_shader_stack.submit(
            seg, raw, chain_paths,
            param_overrides=chain_overrides, cfg=c,
        )
        futures.append(future)

    # ── Phase 2: Wait, normalize, copy previews ───────────────────────────

    print(f"\nWaiting for {len(futures)} shader stacks...")
    for future in futures:
        future.result()

    norm_futures = []
    normed_paths = []

    for i, (seg_idx, tag, _, _) in enumerate(recipes):
        raw = raw_paths[i]
        if normalize:
            normed = c.work_dir / f"bw_{seg_idx:03d}_{tag}_norm.mp4"
            nf = normalize_levels.submit(raw, normed, cfg=c)
            norm_futures.append(nf)
            normed_paths.append(normed)
        else:
            normed_paths.append(raw)

    if norm_futures:
        print("Normalizing levels...")
        for nf in norm_futures:
            nf.result()

    # Copy segment previews
    import shutil
    for i, (seg_idx, tag, _, _) in enumerate(recipes):
        preview = seg_out_dir / f"seg_{seg_idx:03d}_{tag}.mp4"
        shutil.copy2(normed_paths[i], preview)
        print(f"  preview → {preview.name}")

    # ── Phase 3: Shuffle and concat ──────────────────────────────────────

    print("\nConcatenating...")
    if shuffle:
        shuffle_clips(normed_paths, out, seed=seed, cfg=c)
    else:
        concat_clips(normed_paths, out, cfg=c)

    print(f"\n{len(normed_paths)} segments → {out}")
    return out


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _cli():
    import argparse

    parser = argparse.ArgumentParser(
        description="Brain Wipe flows — warp and distortion shader processing.",
    )
    sub = parser.add_subparsers(dest="flow", required=True)

    # ── warp-chain ──
    p = sub.add_parser("warp-chain",
                       help="Apply a chain of warp shaders to source footage")
    p.add_argument("src", type=Path)
    p.add_argument("--shader-dir", type=Path, default=None)
    p.add_argument("--shaders", type=Path, nargs="+", default=None,
                   dest="shader_paths",
                   help="Explicit shader list (skips random selection)")
    p.add_argument("--categories", nargs="+",
                   default=["Warp", "Brain Wipe"],
                   help="ISF category filter (default: Warp 'Brain Wipe')")
    p.add_argument("-n", "--n-shaders", type=int, default=2,
                   help="Number of shaders to chain (default: 2)")
    p.add_argument("--all-shaders", dest="warpers_only", action="store_false",
                   default=True,
                   help="Include generator shaders (no inputImage)")
    p.add_argument("--no-normalize", dest="normalize", action="store_false",
                   default=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("-o", "--output", type=Path, default=None)

    # ── brain-wipe-render ──
    p = sub.add_parser("brain-wipe-render",
                       help="Pre-render a long-form brain wipe sequence")
    p.add_argument("--shader-dir", type=Path, default=None)
    p.add_argument("--categories", nargs="+",
                   default=["Warp", "Brain Wipe", "Generator"])
    p.add_argument("-n", "--n-segments", type=int, default=8,
                   help="Number of segments (default: 8)")
    p.add_argument("--segment-dur", type=float, default=20.0,
                   help="Segment duration in seconds (default: 20)")
    p.add_argument("--min-shaders", type=int, default=2,
                   help="Min total shaders per segment (default: 2)")
    p.add_argument("--max-shaders", type=int, default=6,
                   help="Max total shaders per segment (default: 6)")
    p.add_argument("--width", type=int, default=1920,
                   help="Output width (default: 1920)")
    p.add_argument("--height", type=int, default=1080,
                   help="Output height (default: 1080)")
    p.add_argument("--fps", type=float, default=30.0,
                   help="Output frame rate (default: 30)")
    p.add_argument("--no-shuffle", dest="shuffle", action="store_false",
                   default=True)
    p.add_argument("--no-normalize", dest="normalize", action="store_false",
                   default=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("-o", "--output", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Directory for per-segment previews")

    args = parser.parse_args()
    cfg = Config()
    cfg.ensure_dirs()

    if args.flow == "warp-chain":
        out = warp_chain(
            src=args.src,
            shader_dir=args.shader_dir,
            shader_paths=args.shader_paths,
            shader_categories=args.categories,
            n_shaders=args.n_shaders,
            warpers_only=args.warpers_only,
            output=args.output,
            normalize=args.normalize,
            seed=args.seed,
            cfg=cfg,
        )
        print(f"\nOutput: {out}")

    elif args.flow == "brain-wipe-render":
        out = brain_wipe_render(
            shader_dir=args.shader_dir,
            shader_categories=args.categories,
            n_segments=args.n_segments,
            segment_dur=args.segment_dur,
            min_shaders=args.min_shaders,
            max_shaders=args.max_shaders,
            width=args.width,
            height=args.height,
            fps=args.fps,
            shuffle=args.shuffle,
            normalize=args.normalize,
            seed=args.seed,
            output=args.output,
            output_dir=args.output_dir,
            cfg=cfg,
        )
        print(f"\nOutput: {out}")


if __name__ == "__main__":
    _cli()
