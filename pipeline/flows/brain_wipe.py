"""
Brain Wipe flows — apply warp and distortion shaders to video.

Three flows:

  warp_chain         Apply a chain of warp shaders to source footage.
                     Category-aware: filters library to "Warp" / "Brain Wipe"
                     shaders so glitch/color shaders stay separate.

  brain_wipe_render  Pre-render a long-form brain wipe sequence for streaming.
                     Generator shaders (plasma, tunnel, chladni, etc.)
                     synthesize content from scratch — no source footage needed.
                     Optional warp shaders can be chained after each generator.

  brain_wipe         Recipe-driven meta-flow. Takes a BrainWipeRecipe dataclass
                     that declaratively describes lanes, per-segment processing
                     steps, sequencing, and compositing. Subsumes all other flows
                     — any complex pipeline can be expressed as a recipe.

Both legacy flows accept an explicit shader_categories list for filtering, or
fall back to ["Warp", "Brain Wipe"] by default. Pass shader_categories=None to
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

    # Meta-flow via recipe:
    from pipeline.recipe import crush_sandwich_recipe
    from pipeline.flows.brain_wipe import brain_wipe

    recipe = crush_sandwich_recipe(Path("source/footage.mp4"), seed=42)
    brain_wipe(recipe)

CLI:
    python -m pipeline.flows.brain_wipe warp-chain source/footage.mp4 --n-shaders 2
    python -m pipeline.flows.brain_wipe brain-wipe-render -n 12 --seed 42
    python -m pipeline.flows.brain_wipe brain-wipe --preset crush-sandwich source.mp4
"""

from __future__ import annotations

import hashlib
import random
import shutil
import subprocess

import numpy as np
from pathlib import Path
from typing import Optional

from prefect import flow, task
from prefect.task_runners import ConcurrentTaskRunner

from ..config import Config
from ..ffmpeg import probe
from ..isf import ISFShader, load_shader_dir
from ..recipe import (
    BrainWipeRecipe,
    FootageSource,
    GeneratorSource,
    StaticSource,
    SolidSource,
    CrushStep,
    ShaderStep,
    NormalizeStep,
    ScrubStep,
    DriftStep,
    PingPongStep,
    EchoStep,
    PatchStep,
    SlitScanStep,
    TemporalTileStep,
    SmearStep,
    BloomStep,
    StackStep,
    SlipStep,
    MirrorStep,
    ZoomStep,
    InvertStep,
    HueShiftStep,
    SaturateStep,
    FlowWarpStep,
    TemporalSortStep,
    ExtremaHoldStep,
    FeedbackTransformStep,
    QuadLoopStep,
    BlendComposite,
    MaskedComposite,
    RandomComposite,
    SplitComposite,
    print_recipe,
    hash_recipe,
)
from ..tasks import (
    apply_shader_stack,
    bitrate_crush,
    blend_layers,
    concat_clips,
    detect_cuts,
    drift_loop,
    echo_trail,
    edge_mask,
    generate_solid,
    generate_static,
    gradient_mask,
    luma_mask,
    masked_composite,
    motion_mask,
    normalize_levels,
    ping_pong,
    random_segments,
    segment_at_cuts,
    shuffle_clips,
    time_patch,
    time_scrub,
    transition_sequence,
    slit_scan,
    temporal_tile,
    quad_loop,
    smear,
    bloom,
    frame_stack,
    slip,
    mirror,
    zoom,
    invert,
    hue_shift,
    saturate,
    flow_warp,
    temporal_sort,
    extrema_hold,
    feedback_transform,
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
      task_runner=ConcurrentTaskRunner(max_workers=3))
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
        shutil.copy2(work_path, out)

    print(f"\nOutput: {out}")
    return out


# ─── Flow 2: brain_wipe_render ───────────────────────────────────────────────

@flow(name="brain-wipe-render", log_prints=True,
      task_runner=ConcurrentTaskRunner(max_workers=3))
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


# ─── Flow 3: brain_wipe (recipe-driven meta-flow) ────────────────────────────

def _resolve_shaders_for_step(
    step: ShaderStep,
    rng: random.Random,
    shader_cache: dict[str, dict[str, ISFShader]],
    recipe: BrainWipeRecipe,
    cfg: Config,
) -> tuple[list[Path], dict[str, dict[str, float]]]:
    """Resolve shader paths + param overrides for a ShaderStep."""
    if step.shader_paths:
        return step.shader_paths, step.param_overrides or {}

    s_dir = step.shader_dir or recipe.shader_dir or cfg.shader_dir
    key = str(s_dir)
    if key not in shader_cache:
        shader_cache[key] = load_shader_dir(s_dir)

    pool = shader_cache[key]
    if step.categories:
        pool = filter_shaders(pool, categories=step.categories)

    if not pool:
        pool = shader_cache[key]

    return pick_shader_stack(pool, step.n_shaders, rng, pin_defaults=LEVEL_PARAMS)


# ─── Filter chain merging ────────────────────────────────────────────────────

_MERGEABLE_TYPES = (
    MirrorStep, ZoomStep, InvertStep, HueShiftStep, SaturateStep, NormalizeStep,
)


def _group_steps(recipe_steps: list) -> list[list]:
    """Partition recipe steps into runs of consecutive mergeable steps.

    Non-mergeable steps become singleton groups. Consecutive mergeable steps
    are collected into a single group so they can be applied in one ffmpeg
    command instead of separate encode/decode cycles.
    """
    groups: list[list] = []
    current_run: list = []
    for step in recipe_steps:
        if isinstance(step, _MERGEABLE_TYPES):
            current_run.append(step)
        else:
            if current_run:
                groups.append(current_run)
                current_run = []
            groups.append([step])
    if current_run:
        groups.append(current_run)
    return groups


@task(name="filter-chain")
def _apply_filter_chain(
    src: Path,
    dst: Path,
    *,
    steps: list,
    cfg: Optional[Config] = None,
) -> Path:
    """Apply multiple ffmpeg video filters in a single encode pass.

    Eliminates intermediate encode/decode cycles when consecutive recipe
    steps are all pure ffmpeg filter operations.
    """
    c = cfg or Config()
    info = probe(src, c)

    filters: list[str] = []
    for step in steps:
        match step:
            case MirrorStep(axis=axis):
                filters.append("hflip" if axis == "horizontal" else "vflip")
            case ZoomStep(factor=factor, center_x=cx, center_y=cy):
                w, h = info.width, info.height
                cw = max(2, int(w / factor) & ~1)
                ch = max(2, int(h / factor) & ~1)
                cx_px = int((w - cw) * cx)
                cy_px = int((h - ch) * cy)
                filters.append(
                    f"crop={cw}:{ch}:{cx_px}:{cy_px},"
                    f"scale={w}:{h}:flags=lanczos"
                )
            case InvertStep():
                filters.append("negate")
            case HueShiftStep(degrees=degrees):
                filters.append(f"hue=h={degrees}")
            case SaturateStep(amount=amount):
                filters.append(f"hue=s={amount}")
            case NormalizeStep(black_point=bp, white_point=wp):
                filters.append(
                    f"colorlevels="
                    f"rimin={bp}:gimin={bp}:bimin={bp}:"
                    f"rimax={wp}:gimax={wp}:bimax={wp}"
                )

    vf = ",".join(filters)

    subprocess.run([
        c.ffmpeg_bin, "-y", "-loglevel", c.ffmpeg_loglevel,
        "-i", str(src),
        "-vf", vf,
        "-an",
        *c.encode_args(),
        str(dst),
    ], check=True)

    return dst


def _submit_step(
    step,
    src,
    dst: Path,
    rng: random.Random,
    shader_cache: dict[str, dict[str, ISFShader]],
    recipe: BrainWipeRecipe,
    cfg: Config,
):
    """Submit a single processing step, returning a Prefect future."""
    match step:
        case CrushStep(crush=crush, codec=codec, downscale=downscale):
            return bitrate_crush.submit(
                src, dst, crush=crush, codec=codec, downscale=downscale, cfg=cfg,
            )
        case ShaderStep():
            paths, overrides = _resolve_shaders_for_step(
                step, rng, shader_cache, recipe, cfg,
            )
            return apply_shader_stack.submit(
                src, dst, paths, param_overrides=overrides, cfg=cfg,
            )
        case NormalizeStep(black_point=bp, white_point=wp):
            return normalize_levels.submit(
                src, dst, black_point=bp, white_point=wp, cfg=cfg,
            )
        case ScrubStep(smoothness=smoothness, intensity=intensity):
            return time_scrub.submit(
                src, dst, smoothness=smoothness, intensity=intensity,
                seed=rng.randint(0, 2 ** 31), cfg=cfg,
            )
        case DriftStep(loop_dur=loop_dur, drift=drift):
            return drift_loop.submit(
                src, dst, loop_dur=loop_dur, drift=drift,
                seed=rng.randint(0, 2 ** 31), cfg=cfg,
            )
        case PingPongStep(window=window):
            return ping_pong.submit(
                src, dst, window=window,
                seed=rng.randint(0, 2 ** 31), cfg=cfg,
            )
        case EchoStep(delay=delay, trail=trail):
            return echo_trail.submit(
                src, dst, delay=delay, trail=trail, cfg=cfg,
            )
        case PatchStep(patch_min=patch_min, patch_max=patch_max):
            return time_patch.submit(
                src, dst, patch_min=patch_min, patch_max=patch_max,
                seed=rng.randint(0, 2 ** 31), cfg=cfg,
            )
        case SlitScanStep(axis=axis, scan_speed=scan_speed):
            return slit_scan.submit(
                src, dst, axis=axis, scan_speed=scan_speed,
                seed=rng.randint(0, 2 ** 31), cfg=cfg,
            )
        case TemporalTileStep(grid=grid, offset_scale=offset_scale):
            return temporal_tile.submit(
                src, dst, grid=grid, offset_scale=offset_scale,
                seed=rng.randint(0, 2 ** 31), cfg=cfg,
            )
        case QuadLoopStep(loop_dur=loop_dur, offset_scale=offset_scale, layout=layout):
            return quad_loop.submit(
                src, dst, loop_dur=loop_dur, offset_scale=offset_scale,
                layout=layout, seed=rng.randint(0, 2 ** 31), cfg=cfg,
            )
        case SmearStep(threshold=threshold):
            return smear.submit(
                src, dst, threshold=threshold, cfg=cfg,
            )
        case BloomStep(sensitivity=sensitivity):
            return bloom.submit(
                src, dst, sensitivity=sensitivity, cfg=cfg,
            )
        case StackStep(window=window, mode=mode):
            return frame_stack.submit(
                src, dst, window=window, mode=mode, cfg=cfg,
            )
        case SlipStep(n_bands=n_bands, max_slip=max_slip, axis=axis):
            return slip.submit(
                src, dst, n_bands=n_bands, max_slip=max_slip, axis=axis,
                seed=rng.randint(0, 2 ** 31), cfg=cfg,
            )
        case MirrorStep(axis=axis):
            return mirror.submit(src, dst, axis=axis, cfg=cfg)
        case ZoomStep(factor=factor, center_x=cx, center_y=cy):
            return zoom.submit(
                src, dst, factor=factor, center_x=cx, center_y=cy, cfg=cfg,
            )
        case InvertStep():
            return invert.submit(src, dst, cfg=cfg)
        case HueShiftStep(degrees=degrees):
            return hue_shift.submit(src, dst, degrees=degrees, cfg=cfg)
        case SaturateStep(amount=amount):
            return saturate.submit(src, dst, amount=amount, cfg=cfg)
        case FlowWarpStep(amplify=amplify, smooth=smooth):
            return flow_warp.submit(
                src, dst, amplify=amplify, smooth=smooth,
                seed=rng.randint(0, 2 ** 31), cfg=cfg,
            )
        case TemporalSortStep(mode=mode, direction=direction):
            return temporal_sort.submit(
                src, dst, mode=mode, direction=direction,
                seed=rng.randint(0, 2 ** 31), cfg=cfg,
            )
        case ExtremaHoldStep(mode=mode, decay=decay):
            return extrema_hold.submit(
                src, dst, mode=mode, decay=decay,
                seed=rng.randint(0, 2 ** 31), cfg=cfg,
            )
        case FeedbackTransformStep(transform=xform, amount=amount, mix=mix_val):
            return feedback_transform.submit(
                src, dst, transform=xform, amount=amount, mix=mix_val,
                seed=rng.randint(0, 2 ** 31), cfg=cfg,
            )
        case _:
            raise ValueError(f"Unknown step type: {type(step).__name__}")


def _materialize_generator_source(
    source: GeneratorSource,
    n_segments: int,
    lane_idx: int,
    rng: random.Random,
    shader_cache: dict[str, dict[str, ISFShader]],
    recipe: BrainWipeRecipe,
    recipe_tag: str,
    cfg: Config,
) -> list:
    """Generate segments via generator shaders. Returns list of futures."""
    work = cfg.work_dir / f"bw_{recipe_tag}_lane_{lane_idx:02d}"
    work.mkdir(parents=True, exist_ok=True)

    bw_dir = recipe.brain_wipe_dir
    key = str(bw_dir)
    if key not in shader_cache:
        shader_cache[key] = load_shader_dir(bw_dir)

    bw_shaders = shader_cache[key]
    generators = filter_shaders(bw_shaders, has_image_input=False)
    warpers = filter_shaders(
        bw_shaders,
        categories=source.warp_categories,
        has_image_input=True,
    )

    if not generators:
        raise ValueError(
            f"No generator shaders found in {bw_dir}. "
            f"Generator shaders (no inputImage) are required."
        )

    futures = []
    for i in range(n_segments):
        seg_rng = random.Random(rng.randint(0, 2 ** 31))

        # Solid placeholder — randomize duration per segment
        seg_dur = rng.uniform(source.min_dur, source.max_dur)
        solid_path = work / f"solid_{i:03d}.mp4"
        solid_f = generate_solid.submit(
            solid_path, seg_dur,
            width=recipe.width, height=recipe.height,
            fps=recipe.fps, cfg=cfg,
        )

        # Generator + warps
        gen_paths, gen_ov = pick_shader_stack(
            generators, 1, seg_rng, pin_defaults=LEVEL_PARAMS,
        )
        chain_paths = list(gen_paths)
        chain_ov = dict(gen_ov)

        if source.n_warps > 0 and warpers:
            w_paths, w_ov = pick_shader_stack(
                warpers, source.n_warps, seg_rng, pin_defaults=LEVEL_PARAMS,
            )
            chain_paths.extend(w_paths)
            chain_ov.update(w_ov)

        tag = hashlib.sha1(
            ";".join(p.stem for p in chain_paths).encode()
        ).hexdigest()[:8]

        print(f"    seg {i:03d} [{tag}]:")
        print_stack(chain_paths, chain_ov, indent="      ")

        rendered_path = work / f"gen_{i:03d}_{tag}.mp4"
        rendered_f = apply_shader_stack.submit(
            solid_f, rendered_path, chain_paths,
            param_overrides=chain_ov, cfg=cfg,
        )
        futures.append(rendered_f)

    return futures


def _scale_segments_to_recipe(
    segments: list[Path],
    recipe: BrainWipeRecipe,
    cfg: Config,
) -> list[Path]:
    """Scale footage segments to recipe dimensions if they don't match."""
    import subprocess
    target_w, target_h = recipe.width, recipe.height
    result = []
    for seg in segments:
        info = probe(seg, cfg)
        if info.width == target_w and info.height == target_h:
            result.append(seg)
            continue
        scaled = seg.parent / f"{seg.stem}_scaled{seg.suffix}"
        subprocess.run([
            cfg.ffmpeg_bin, "-y", "-loglevel", cfg.ffmpeg_loglevel,
            "-i", str(seg),
            "-vf", f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
                   f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2",
            "-an",
            *cfg.encode_args(),
            str(scaled),
        ], check=True)
        result.append(scaled)
    return result


def _materialize_source(
    lane,
    lane_idx: int,
    rng: random.Random,
    shader_cache: dict[str, dict[str, ISFShader]],
    recipe: BrainWipeRecipe,
    recipe_tag: str,
    cfg: Config,
) -> list:
    """
    Materialise segment sources for a lane.
    Returns a list of Paths or Prefect futures (for generator sources).
    """
    source = lane.source
    work = cfg.work_dir / f"bw_{recipe_tag}_lane_{lane_idx:02d}"
    work.mkdir(parents=True, exist_ok=True)

    match source:
        case FootageSource(path=path, method=method):
            if method == "scene":
                cuts = detect_cuts(path, cfg=cfg)
                segments = segment_at_cuts(
                    path, cuts, output_dir=work / "segments", cfg=cfg,
                )
                if lane.n_segments > 0 and lane.n_segments < len(segments):
                    segments = rng.sample(segments, lane.n_segments)
                print(f"    {len(segments)} scene segments from {path.name}")
                return _scale_segments_to_recipe(segments, recipe, cfg)
            else:
                segments = random_segments(
                    path, lane.n_segments,
                    min_dur=source.min_dur, max_dur=source.max_dur,
                    output_dir=work / "segments", cfg=cfg,
                )
                print(f"    {len(segments)} random segments from {path.name}")
                return _scale_segments_to_recipe(segments, recipe, cfg)

        case GeneratorSource():
            dur_label = (f"{source.min_dur:.0f}s" if source.min_dur == source.max_dur
                         else f"{source.min_dur:.0f}–{source.max_dur:.0f}s")
            print(f"    generating {lane.n_segments} segments "
                  f"({dur_label}, {source.n_warps} warps)")
            return _materialize_generator_source(
                source, lane.n_segments, lane_idx, rng,
                shader_cache, recipe, recipe_tag, cfg,
            )

        case StaticSource(min_dur=lo, max_dur=hi):
            paths = []
            for i in range(lane.n_segments):
                dur = rng.uniform(lo, hi)
                p = work / f"static_{i:03d}.mp4"
                generate_static(p, dur, width=recipe.width,
                                height=recipe.height, fps=recipe.fps, cfg=cfg)
                paths.append(p)
            dur_label = f"{lo:.0f}s" if lo == hi else f"{lo:.0f}–{hi:.0f}s"
            print(f"    {lane.n_segments} static segments ({dur_label})")
            return paths

        case SolidSource(min_dur=lo, max_dur=hi, color=color):
            paths = []
            for i in range(lane.n_segments):
                dur = rng.uniform(lo, hi)
                p = work / f"solid_{i:03d}.mp4"
                generate_solid(p, dur, color=color, width=recipe.width,
                               height=recipe.height, fps=recipe.fps, cfg=cfg)
                paths.append(p)
            dur_label = f"{lo:.0f}s" if lo == hi else f"{lo:.0f}–{hi:.0f}s"
            print(f"    {lane.n_segments} solid segments ({dur_label})")
            return paths

        case _:
            raise ValueError(f"Unknown source type: {type(source).__name__}")


def _process_lane(
    lane,
    lane_idx: int,
    source_items: list,
    rng: random.Random,
    shader_cache: dict[str, dict[str, ISFShader]],
    recipe: BrainWipeRecipe,
    recipe_tag: str,
    cfg: Config,
) -> list[Path]:
    """
    Process all segments in a lane through the recipe steps.
    Returns list of final output paths.
    """
    if not lane.recipe:
        # No processing — resolve source futures and return
        return [
            item.result() if hasattr(item, 'result') else item
            for item in source_items
        ]

    work = cfg.work_dir / f"bw_{recipe_tag}_lane_{lane_idx:02d}"
    work.mkdir(parents=True, exist_ok=True)

    groups = _group_steps(lane.recipe)

    final_futures = []
    for seg_idx, seg_source in enumerate(source_items):
        seg_rng = random.Random(rng.randint(0, 2 ** 31))
        current = seg_source  # Path or future

        output_idx = 0
        for group in groups:
            dst = work / f"seg_{seg_idx:03d}_s{output_idx}.mp4"
            if len(group) >= 2 and isinstance(group[0], _MERGEABLE_TYPES):
                # Merge consecutive ffmpeg filters into one encode pass
                current = _apply_filter_chain.submit(
                    current, dst, steps=group, cfg=cfg,
                )
            else:
                current = _submit_step(
                    group[0], current, dst, seg_rng,
                    shader_cache, recipe, cfg,
                )
            output_idx += 1

        final_futures.append(current)

    # Wait for all segment pipelines to complete
    results = []
    for f in final_futures:
        results.append(f.result() if hasattr(f, 'result') else f)
    return results


def _sequence_lane(
    lane,
    lane_idx: int,
    processed: list[Path],
    rng: random.Random,
    recipe: BrainWipeRecipe,
    recipe_tag: str,
    cfg: Config,
) -> Path:
    """Sequence processed segments into a single output per lane."""
    out = cfg.work_dir / f"bw_{recipe_tag}_lane_{lane_idx:02d}_seq.mp4"

    # Interleave static if requested (only without transitions)
    if lane.static_gap > 0 and not lane.transition:
        # Match static resolution to actual segment resolution (not recipe defaults)
        seg_info = probe(processed[0], cfg)
        static_path = cfg.work_dir / f"bw_{recipe_tag}_lane_{lane_idx:02d}_static.mp4"
        generate_static(
            static_path, lane.static_gap,
            width=seg_info.width, height=seg_info.height,
            fps=seg_info.fps, cfg=cfg,
        )
        pieces = []
        for i, seg in enumerate(processed):
            pieces.append(seg)
            if i < len(processed) - 1:
                pieces.append(static_path)
        processed = pieces

    if lane.transition:
        # Shuffle order if requested (before applying transitions)
        if lane.sequencing == "shuffle":
            seq_rng = random.Random(rng.randint(0, 2 ** 31))
            seq_rng.shuffle(processed)
        t = lane.transition
        kwargs: dict = {}
        if t.type == "luma_wipe":
            kwargs = dict(pattern=t.pattern, softness=t.softness, angle=t.angle)
        elif t.type == "whip_pan":
            kwargs = dict(direction=t.direction, blur_strength=t.blur_strength)
        elif t.type == "flash":
            kwargs = dict(decay=t.decay)
        transition_sequence(
            processed, out, transition_type=t.type,
            duration=t.duration, seed=rng.randint(0, 2 ** 31), cfg=cfg,
            **kwargs,
        )
    elif lane.sequencing == "shuffle":
        shuffle_clips(processed, out, seed=rng.randint(0, 2 ** 31), cfg=cfg)
    else:
        concat_clips(processed, out, cfg=cfg)

    print(f"    lane {lane_idx}: {len(processed)} clips → {out.name}")
    return out


def _equalize_lane_durations(
    lane_paths: list[Path],
    recipe_tag: str,
    cfg: Config,
) -> list[Path]:
    """Loop shorter lanes to match the longest, preventing freeze frames."""
    durations = [probe(p, cfg).duration for p in lane_paths]
    max_dur = max(durations)
    equalized = []
    for i, (path, dur) in enumerate(zip(lane_paths, durations)):
        if dur >= max_dur - 0.1:  # close enough
            equalized.append(path)
        else:
            looped = cfg.work_dir / f"bw_{recipe_tag}_lane_{i:02d}_looped.mp4"
            cmd = [
                cfg.ffmpeg_bin,
                "-loglevel", cfg.ffmpeg_loglevel,
                "-stream_loop", "-1",
                "-i", str(path),
                "-t", str(max_dur),
                "-an",
                *cfg.encode_args(),
                "-y", str(looped),
            ]
            subprocess.run(cmd, check=True)
            print(f"    lane {i}: looped {dur:.1f}s → {max_dur:.1f}s")
            equalized.append(looped)
    return equalized


def _composite_lanes(
    lane_paths: list[Path],
    recipe: BrainWipeRecipe,
    recipe_tag: str,
    cfg: Config,
) -> Path:
    """Composite multiple lane outputs according to the recipe's CompositeSpec."""
    lane_paths = _equalize_lane_durations(lane_paths, recipe_tag, cfg)
    out = cfg.work_dir / f"bw_{recipe_tag}_composited.mp4"

    match recipe.composite:
        case BlendComposite(mode=mode, opacity=opacity):
            # Fold: blend lane0+lane1, then result+lane2, etc.
            current = lane_paths[0]
            for i, overlay in enumerate(lane_paths[1:], 1):
                dst = cfg.work_dir / f"bw_{recipe_tag}_blend_{i}.mp4"
                blend_layers(current, overlay, dst,
                             mode=mode, opacity=opacity, cfg=cfg)
                current = dst
            shutil.copy2(current, out)

        case MaskedComposite(mask_type=mask_type, mask_params=params):
            # Generate mask from base lane, composite overlay onto base
            mask_path = cfg.work_dir / f"bw_{recipe_tag}_composite_mask.mp4"

            if mask_type == "gradient":
                # Gradient mask is synthetic — generated from dimensions,
                # not derived from source video
                base_info = probe(lane_paths[0], cfg)
                gradient_mask(
                    mask_path,
                    width=base_info.width,
                    height=base_info.height,
                    duration=base_info.duration,
                    fps=base_info.fps,
                    direction=params.get("direction", "horizontal"),
                    cfg=cfg,
                )
            else:
                mask_fns = {
                    "luma": luma_mask,
                    "edge": edge_mask,
                    "motion": motion_mask,
                }
                mask_fn = mask_fns.get(mask_type)
                if mask_fn is None:
                    raise ValueError(
                        f"Unsupported mask type for compositing: {mask_type}. "
                        f"Available: {list(mask_fns.keys()) + ['gradient']}"
                    )
                mask_fn(lane_paths[0], mask_path, **params, cfg=cfg)

            # Fold: composite lane0+lane1 via mask, then result+lane2, etc.
            current = lane_paths[0]
            for i, overlay in enumerate(lane_paths[1:], 1):
                dst = cfg.work_dir / f"bw_{recipe_tag}_masked_{i}.mp4"
                masked_composite(current, overlay, mask_path, dst, cfg=cfg)
                current = dst
            shutil.copy2(current, out)

        case SplitComposite(layout=layout):
            from ..ffmpeg import read_frames, FrameWriter
            base_info = probe(lane_paths[0], cfg)
            h, w = base_info.height, base_info.width
            n_lanes = len(lane_paths)

            # Compute band regions — crop each lane's corresponding region
            if layout == "horizontal":
                band_h = h // n_lanes
                regions = []
                for i in range(n_lanes):
                    y0 = i * band_h
                    bh = band_h if i < n_lanes - 1 else h - y0
                    regions.append((y0, 0, bh, w))
            else:
                band_w = w // n_lanes
                regions = []
                for i in range(n_lanes):
                    x0 = i * band_w
                    bw = band_w if i < n_lanes - 1 else w - x0
                    regions.append((0, x0, h, bw))

            # Crop each lane's band from its own frames (no resize)
            streams = [read_frames(lp, cfg) for lp in lane_paths]
            with FrameWriter(out, base_info, cfg=cfg) as writer:
                for frames in zip(*streams):
                    out_frame = np.empty((h, w, 3), dtype=np.uint8)
                    for i, (y0, x0, bh, bw) in enumerate(regions):
                        out_frame[y0:y0 + bh, x0:x0 + bw] = \
                            frames[i][y0:y0 + bh, x0:x0 + bw]
                    writer.write(out_frame)

        case RandomComposite():
            raise NotImplementedError(
                "RandomComposite requires compositing_lab integration — "
                "use compositing_lab directly for now."
            )

        case _:
            raise ValueError(
                f"Unknown composite type: {type(recipe.composite).__name__}"
            )

    print(f"  composited {len(lane_paths)} lanes → {out.name}")
    return out


def _cleanup_work(work_dir: Path, recipe_tag: str) -> None:
    """Remove all intermediate files for a recipe run."""
    count = 0
    freed = 0
    prefix = f"bw_{recipe_tag}"
    for p in sorted(work_dir.iterdir()):
        if p.name.startswith(prefix):
            if p.is_dir():
                size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                shutil.rmtree(p)
            else:
                size = p.stat().st_size
                p.unlink()
            freed += size
            count += 1
    if count:
        print(f"  cleanup: removed {count} work items ({freed / 1024 / 1024:.0f} MB)")


@flow(name="brain-wipe", log_prints=True,
      task_runner=ConcurrentTaskRunner(max_workers=3))
def brain_wipe(
    recipe: BrainWipeRecipe,
    output: Optional[Path] = None,
    cfg: Optional[Config] = None,
    cleanup: bool = True,
) -> Path | list[Path]:
    """
    Recipe-driven meta-flow for the brain wipe pipeline.

    Takes a BrainWipeRecipe that declaratively describes:
      - Lanes: parallel processing streams with source, recipe, sequencing
      - Compositing: optional blending/masking of lane outputs
      - Post-processing: final steps after compositing

    Returns a single Path (composited or single-lane) or list[Path]
    (multi-lane, no compositing).

    Build recipes directly or use the helpers in pipeline.recipe:
      crush_sandwich_recipe, stooges_recipe, generator_render_recipe,
      composite_recipe.
    """
    c = cfg or Config()
    c.ensure_dirs()
    rng = random.Random(recipe.seed)
    shader_cache: dict[str, dict[str, ISFShader]] = {}

    recipe_tag = hash_recipe(recipe)

    # ── Print recipe ──────────────────────────────────────────────────────
    print_recipe(recipe)
    print(f"  recipe hash: {recipe_tag}\n")

    # ── Process each lane ─────────────────────────────────────────────────

    lane_outputs: list[Path] = []

    for lane_idx, lane in enumerate(recipe.lanes):
        lane_rng = random.Random(rng.randint(0, 2 ** 31))

        print(f"─── Lane {lane_idx} ───")

        # Materialise source segments
        source_items = _materialize_source(
            lane, lane_idx, lane_rng, shader_cache, recipe, recipe_tag, c,
        )

        # Process through recipe steps
        processed = _process_lane(
            lane, lane_idx, source_items, lane_rng,
            shader_cache, recipe, recipe_tag, c,
        )

        # Sequence into single output
        sequenced = _sequence_lane(
            lane, lane_idx, processed, lane_rng, recipe, recipe_tag, c,
        )
        lane_outputs.append(sequenced)

    # ── Composite lanes ───────────────────────────────────────────────────

    if recipe.composite is not None and len(lane_outputs) > 1:
        result = _composite_lanes(lane_outputs, recipe, recipe_tag, c)
    elif len(lane_outputs) == 1:
        result = lane_outputs[0]
    else:
        # Multi-lane, no composite — apply post per lane, return list
        final_lanes = []
        for i, lane_path in enumerate(lane_outputs):
            current = lane_path
            if recipe.post:
                for step_idx, step in enumerate(recipe.post):
                    dst = c.work_dir / f"bw_{recipe_tag}_post_ch{i:02d}_s{step_idx}.mp4"
                    post_rng = random.Random(rng.randint(0, 2 ** 31))
                    f = _submit_step(
                        step, current, dst, post_rng, shader_cache, recipe, c,
                    )
                    current = f.result() if hasattr(f, 'result') else f
            out_path = c.output_dir / f"brain_wipe_{recipe_tag}_ch{i:02d}.mp4"
            shutil.copy2(current, out_path)
            final_lanes.append(out_path)
        print(f"\n{len(final_lanes)} channel outputs in {c.output_dir}")
        if cleanup:
            _cleanup_work(c.work_dir, recipe_tag)
        return final_lanes

    # ── Post-processing ───────────────────────────────────────────────────

    if recipe.post:
        current = result
        for step_idx, step in enumerate(recipe.post):
            dst = c.work_dir / f"bw_{recipe_tag}_post_s{step_idx}.mp4"
            post_rng = random.Random(rng.randint(0, 2 ** 31))
            f = _submit_step(
                step, current, dst, post_rng, shader_cache, recipe, c,
            )
            current = f.result() if hasattr(f, 'result') else f
        result = current

    # ── Final output ──────────────────────────────────────────────────────

    final = output or c.output_dir / f"brain_wipe_{recipe_tag}.mp4"
    shutil.copy2(result, final)
    print(f"\nOutput: {final}")
    if cleanup:
        _cleanup_work(c.work_dir, recipe_tag)
    return final


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

    # ── brain-wipe (recipe meta-flow) ──
    p = sub.add_parser("brain-wipe",
                       help="Recipe-driven meta-flow (use --preset for quick start)")
    p.add_argument("src", type=Path, nargs="?", default=None,
                   help="Source footage (required for footage-based presets)")
    p.add_argument("--preset", type=str, default="crush-sandwich",
                   choices=[
                       "crush-sandwich", "stooges", "generator-render",
                       "temporal-sandwich", "deep-time", "hybrid-composite",
                       "codec-spectrum", "breathing-wall", "erosion",
                       "palimpsest", "generator-stooges", "gradient-dissolve",
                       "accretion",
                   ],
                   help="Recipe preset (default: crush-sandwich)")
    p.add_argument("-n", "--n-segments", type=int, default=8)
    p.add_argument("--segment-dur", type=float, default=20.0,
                   help="Segment duration for generator preset (default: 20)")
    p.add_argument("--n-shaders", type=int, default=3,
                   help="Shaders per stack (default: 3)")
    p.add_argument("--crush", type=float, default=0.95,
                   help="Crush amount (default: 0.95)")
    p.add_argument("--segment-counts", type=str, default=None,
                   help="Comma-separated segment counts for stooges (e.g. 8,10,12)")
    p.add_argument("--static-gap", type=float, default=0.3,
                   help="Static gap duration for stooges (default: 0.3)")
    p.add_argument("--n-warps", type=int, default=2,
                   help="Warp shaders for generator preset (default: 2)")
    p.add_argument("--brain-wipe-dir", type=Path,
                   default=Path("brain-wipe-shaders"))
    p.add_argument("--no-normalize", dest="normalize", action="store_false",
                   default=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("-o", "--output", type=Path, default=None)

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

    elif args.flow == "brain-wipe":
        from ..recipe import (
            crush_sandwich_recipe,
            stooges_recipe,
            generator_render_recipe,
            temporal_sandwich_recipe,
            deep_time_recipe,
            hybrid_composite_recipe,
            codec_spectrum_recipe,
            breathing_wall_recipe,
            erosion_recipe,
            palimpsest_recipe,
            generator_stooges_recipe,
            gradient_dissolve_recipe,
            accretion_recipe,
        )

        preset = args.preset

        # -- presets that need source footage --
        _footage_presets = {
            "crush-sandwich", "stooges", "temporal-sandwich", "deep-time",
            "hybrid-composite", "codec-spectrum", "breathing-wall",
            "erosion", "palimpsest", "gradient-dissolve", "accretion",
        }
        if preset in _footage_presets and args.src is None:
            parser.error(f"{preset} preset requires source footage")

        if preset == "crush-sandwich":
            r = crush_sandwich_recipe(
                args.src,
                n_segments=args.n_segments,
                crush=args.crush,
                n_shaders=args.n_shaders,
                normalize=args.normalize,
                seed=args.seed,
            )
        elif preset == "stooges":
            counts = (
                [int(x) for x in args.segment_counts.split(",")]
                if args.segment_counts
                else args.n_segments
            )
            r = stooges_recipe(
                args.src,
                segment_counts=counts,
                crush=args.crush,
                n_shaders=args.n_shaders,
                static_gap=args.static_gap,
                seed=args.seed,
            )
        elif preset == "generator-render":
            r = generator_render_recipe(
                n_segments=args.n_segments,
                segment_dur=args.segment_dur,
                max_warps=args.n_warps,
                normalize=args.normalize,
                seed=args.seed,
                brain_wipe_dir=args.brain_wipe_dir,
            )
        elif preset == "temporal-sandwich":
            r = temporal_sandwich_recipe(
                args.src,
                n_segments=args.n_segments,
                n_shaders=args.n_shaders,
                seed=args.seed,
            )
        elif preset == "deep-time":
            r = deep_time_recipe(
                args.src,
                n_segments=args.n_segments,
                seed=args.seed,
            )
        elif preset == "hybrid-composite":
            r = hybrid_composite_recipe(
                args.src,
                n_segments=args.n_segments,
                crush=args.crush,
                n_shaders=args.n_shaders,
                segment_dur=args.segment_dur,
                n_warps=args.n_warps,
                seed=args.seed,
                brain_wipe_dir=args.brain_wipe_dir,
            )
        elif preset == "codec-spectrum":
            r = codec_spectrum_recipe(
                args.src,
                n_segments=args.n_segments,
                n_shaders=args.n_shaders,
                seed=args.seed,
            )
        elif preset == "breathing-wall":
            r = breathing_wall_recipe(
                args.src,
                n_shaders=args.n_shaders,
                seed=args.seed,
            )
        elif preset == "erosion":
            r = erosion_recipe(
                args.src,
                n_segments=args.n_segments,
                seed=args.seed,
            )
        elif preset == "palimpsest":
            r = palimpsest_recipe(
                args.src,
                n_segments=args.n_segments,
                seed=args.seed,
            )
        elif preset == "generator-stooges":
            counts = (
                [int(x) for x in args.segment_counts.split(",")]
                if args.segment_counts
                else [6, 8, 10, 8, 6]
            )
            r = generator_stooges_recipe(
                segment_counts=counts,
                segment_dur=args.segment_dur,
                n_warps=args.n_warps,
                crush=args.crush,
                n_shaders=args.n_shaders,
                static_gap=args.static_gap,
                seed=args.seed,
                brain_wipe_dir=args.brain_wipe_dir,
            )
        elif preset == "gradient-dissolve":
            r = gradient_dissolve_recipe(
                args.src,
                n_segments=args.n_segments,
                crush=args.crush,
                n_shaders=args.n_shaders,
                segment_dur=args.segment_dur,
                n_warps=args.n_warps,
                seed=args.seed,
                brain_wipe_dir=args.brain_wipe_dir,
            )
        elif preset == "accretion":
            r = accretion_recipe(
                args.src,
                n_segments=args.n_segments,
                seed=args.seed,
            )
        else:
            parser.error(f"Unknown preset: {preset}")

        out = brain_wipe(r, output=args.output, cfg=cfg)
        if isinstance(out, list):
            for p in out:
                print(f"Output: {p}")
        else:
            print(f"\nOutput: {out}")


if __name__ == "__main__":
    _cli()
