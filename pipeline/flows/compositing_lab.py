"""
Compositing Lab — brain wipe composite explorer.

Generates N independent composite samples by layering brain wipe renders
(generator shaders + optional warps) on top of each other through random
compositing operations. No source footage needed — all visual content is
synthesized by generator shaders from the brain-wipe-shaders directory.

Layer types: brain_wipe (generator + 0–N warps), static (TV noise), solid
(random colour). Compositing operations: blend modes, masked composites,
self-keyed overlays, chromakey, picture-in-picture, and multi-PIP.

Usage (Python):
    from pipeline.flows.compositing_lab import compositing_lab
    compositing_lab(n_samples=8, seed=42)

CLI:
    python -m pipeline.flows.compositing_lab \\
        -n 8 --seed 42 --segment-dur 10 \\
        --brain-wipe-dir brain-wipe-shaders \\
        -o output/comp_lab
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Optional

from prefect import flow

from ..config import Config
from ..isf import load_shader_dir
from ..tasks import (
    blend_layers,
    masked_composite,
    chromakey_composite,
    picture_in_picture,
    apply_shader_stack,
    apply_random_shader_stack,
    generate_static,
    generate_solid,
    normalize_levels,
    luma_mask,
    edge_mask,
    motion_mask,
    chroma_mask,
    gradient_mask,
)
from ..tasks.glitch import bitrate_crush
from .brain_wipe import filter_shaders, pick_shader_stack, LEVEL_PARAMS, print_stack


# ─── Blend modes available for random selection ─────────────────────────────

BLEND_MODES = ["screen", "add", "multiply", "overlay", "difference", "softlight"]

# Mask types and their parameter randomizers
MASK_TYPES = ["luma", "edge", "motion", "gradient"]

# Extended mask types including chroma (for self-keyed operations)
MASK_TYPES_EXTENDED = ["luma", "edge", "motion", "gradient", "chroma"]

# Warp-category shader stems from shaders/ dir (spatial distortion, not color)
WARP_SHADER_STEMS = [
    "warp", "lens_warp", "shear", "feedback_zoom",
    "stereo_project", "block_shift", "scan_tear",
]

# Hue targets for random chromakey (OpenCV 0-179 scale)
CHROMA_HUE_TARGETS = [0, 15, 30, 60, 90, 120, 150]

# Weighted operation pool — self_keyed and chromakey are favoured
# because content-dependent transparency is the core new capability.
OP_WEIGHTS = {
    "blend": 3,
    "masked": 2,
    "pip": 2,
    "self_keyed": 4,
    "chromakey": 3,
    "multi_pip": 2,
}


# ─── Layer generation ────────────────────────────────────────────────────────

def _make_layer_brain_wipe(
    dur: float, width: int, height: int, fps: float,
    work_dir: Path, tag: str, rng: random.Random,
    generators: dict, warpers: dict,
    min_warps: int, max_warps: int,
    cfg: Config,
) -> Path:
    """Render a brain wipe layer: generator + optional warp chain."""
    solid = work_dir / f"layer_{tag}_solid.mp4"
    generate_solid(solid, dur, width=width, height=height, fps=fps, cfg=cfg)

    # Pick generator (always 1), pin level params
    gen_paths, gen_overrides = pick_shader_stack(
        generators, 1, rng, pin_defaults=LEVEL_PARAMS,
    )
    chain_paths = list(gen_paths)
    chain_overrides = dict(gen_overrides)

    # Pick 0–N warps, pin level params
    n_warps = rng.randint(min_warps, max_warps)
    if n_warps > 0 and warpers:
        warp_paths, warp_overrides = pick_shader_stack(
            warpers, n_warps, rng, pin_defaults=LEVEL_PARAMS,
        )
        chain_paths.extend(warp_paths)
        chain_overrides.update(warp_overrides)

    print(f"  layer {tag}: brain_wipe")
    print_stack(chain_paths, chain_overrides, indent="      ")

    raw = work_dir / f"layer_{tag}_bw.mp4"
    apply_shader_stack(solid, raw, chain_paths,
                        param_overrides=chain_overrides, cfg=cfg)

    # 25% chance: crush the layer before compositing
    if rng.random() < 0.25:
        crush_level = 0.4 + rng.random() * 0.6
        crushed = work_dir / f"layer_{tag}_bw_crush.mp4"
        bitrate_crush(raw, crushed, crush=crush_level, cfg=cfg)
        print(f"      crush={crush_level:.2f}")
        return crushed

    return raw


def _make_layer_static(
    dur: float, width: int, height: int, fps: float,
    work_dir: Path, tag: str, cfg: Config,
) -> Path:
    """Generate TV static noise."""
    dst = work_dir / f"layer_{tag}_static.mp4"
    generate_static(dst, dur, width=width, height=height, fps=fps, cfg=cfg)
    return dst


def _make_layer_solid(
    dur: float, width: int, height: int, fps: float,
    work_dir: Path, tag: str, rng: random.Random, cfg: Config,
) -> Path:
    """Generate a random solid colour."""
    r, g, b = rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)
    dst = work_dir / f"layer_{tag}_solid_rgb.mp4"
    generate_solid(dst, dur, color=(r, g, b),
                    width=width, height=height, fps=fps, cfg=cfg)
    return dst


# ─── Layer pre-processing ───────────────────────────────────────────────────

def _preprocess_layer(
    layer: Path, work_dir: Path, tag: str,
    rng: random.Random,
    shader_dir: Optional[Path],
    brain_wipe_dir: Optional[Path],
    cfg: Config,
) -> Path:
    """Optionally warp a layer through 1-2 spatial shaders before compositing."""
    if rng.random() > 0.5:
        return layer

    # Build a pool of warp shader paths from available sources
    warp_paths: list[Path] = []

    if shader_dir and shader_dir.exists():
        all_shaders = load_shader_dir(shader_dir)
        for stem in WARP_SHADER_STEMS:
            if stem in all_shaders:
                warp_paths.append(all_shaders[stem].path)

    if brain_wipe_dir and brain_wipe_dir.exists():
        bw_shaders = load_shader_dir(brain_wipe_dir)
        warp_pool = filter_shaders(bw_shaders, has_image_input=True)
        warp_paths.extend(s.path for s in warp_pool.values())

    if not warp_paths:
        return layer

    n_shaders = rng.randint(1, 2)
    chosen = rng.sample(warp_paths, min(n_shaders, len(warp_paths)))

    dst = work_dir / f"{tag}_warped.mp4"
    apply_shader_stack(layer, dst, chosen, cfg=cfg)
    return dst


# ─── Compositing operations ─────────────────────────────────────────────────

def _op_blend(layers: list[Path], work_dir: Path, tag: str,
              rng: random.Random, cfg: Config) -> list[Path]:
    """Blend two random layers with a random mode and opacity."""
    if len(layers) < 2:
        return layers
    a_idx, b_idx = rng.sample(range(len(layers)), 2)
    mode = rng.choice(BLEND_MODES)
    opacity = 0.2 + rng.random() * 0.6
    dst = work_dir / f"comp_{tag}_blend.mp4"
    blend_layers(layers[a_idx], layers[b_idx], dst,
                  mode=mode, opacity=opacity, cfg=cfg)
    # Replace both consumed layers with the result
    result = [l for i, l in enumerate(layers) if i not in (a_idx, b_idx)]
    result.append(dst)
    return result


def _op_masked(layers: list[Path], work_dir: Path, tag: str,
               rng: random.Random, width: int, height: int,
               fps: float, dur: float, cfg: Config) -> list[Path]:
    """Masked composite of two layers using a randomly-generated mask."""
    if len(layers) < 2:
        return layers
    a_idx, b_idx = rng.sample(range(len(layers)), 2)
    mask_type = rng.choice(MASK_TYPES)
    mask_path = work_dir / f"comp_{tag}_mask.mp4"

    # Generate mask from one of the two layers (random choice)
    mask_src = layers[rng.choice([a_idx, b_idx])]
    if mask_type == "luma":
        thresh = 0.2 + rng.random() * 0.6
        luma_mask(mask_src, mask_path, threshold=thresh,
                   invert=rng.random() < 0.3, blur=rng.choice([0, 3, 5]),
                   cfg=cfg)
    elif mask_type == "edge":
        low = rng.randint(20, 60)
        high = low + rng.randint(40, 100)
        edge_mask(mask_src, mask_path, low=low, high=high,
                   dilate=rng.randint(0, 3), cfg=cfg)
    elif mask_type == "motion":
        motion_mask(mask_src, mask_path,
                     threshold=rng.randint(10, 40),
                     blur=rng.choice([3, 5, 7]), cfg=cfg)
    elif mask_type == "gradient":
        direction = rng.choice(["horizontal", "vertical", "radial"])
        gradient_mask(mask_path, width, height, dur, fps=fps,
                       direction=direction, cfg=cfg)

    dst = work_dir / f"comp_{tag}_masked.mp4"
    masked_composite(layers[a_idx], layers[b_idx], mask_path, dst, cfg=cfg)
    result = [l for i, l in enumerate(layers) if i not in (a_idx, b_idx)]
    result.append(dst)
    return result


def _op_pip(layers: list[Path], work_dir: Path, tag: str,
            rng: random.Random, width: int, height: int,
            cfg: Config) -> list[Path]:
    """Picture-in-picture: overlay one layer scaled down onto another."""
    if len(layers) < 2:
        return layers
    a_idx, b_idx = rng.sample(range(len(layers)), 2)
    scale = 0.15 + rng.random() * 0.35
    pip_w = int(width * scale)
    max_x = max(0, width - pip_w)
    max_y = max(0, height - int(height * scale))
    x = rng.randint(0, max_x) if max_x > 0 else 0
    y = rng.randint(0, max_y) if max_y > 0 else 0
    dst = work_dir / f"comp_{tag}_pip.mp4"
    picture_in_picture(layers[a_idx], layers[b_idx], dst,
                        x=x, y=y, scale=scale, cfg=cfg)
    result = [l for i, l in enumerate(layers) if i not in (a_idx, b_idx)]
    result.append(dst)
    return result


def _op_self_keyed(layers: list[Path], work_dir: Path, tag: str,
                   rng: random.Random, width: int, height: int,
                   fps: float, dur: float, cfg: Config) -> list[Path]:
    """Self-keyed overlay: mask derived FROM the overlay's own content.

    The overlay's brightness, edges, motion, or color determines where
    it becomes transparent, revealing the base through its own structure.
    """
    if len(layers) < 2:
        return layers
    base_idx, overlay_idx = rng.sample(range(len(layers)), 2)
    overlay = layers[overlay_idx]

    mask_type = rng.choice(MASK_TYPES_EXTENDED)
    mask_path = work_dir / f"comp_{tag}_selfkey_mask.mp4"

    if mask_type == "luma":
        thresh = 0.3 + rng.random() * 0.4
        luma_mask(overlay, mask_path, threshold=thresh,
                  invert=rng.random() < 0.5, blur=rng.choice([3, 5, 7]),
                  cfg=cfg)
    elif mask_type == "edge":
        low = rng.randint(30, 80)
        high = low + rng.randint(40, 120)
        edge_mask(overlay, mask_path, low=low, high=high,
                  dilate=rng.randint(1, 4), cfg=cfg)
    elif mask_type == "motion":
        motion_mask(overlay, mask_path,
                    threshold=rng.randint(10, 40),
                    blur=rng.choice([3, 5, 7]), cfg=cfg)
    elif mask_type == "chroma":
        hue = rng.choice(CHROMA_HUE_TARGETS)
        chroma_mask(overlay, mask_path,
                    hue_center=hue, hue_range=rng.randint(10, 30),
                    sat_min=rng.randint(30, 80),
                    invert=rng.random() < 0.5,
                    blur=rng.choice([3, 5]), cfg=cfg)
    elif mask_type == "gradient":
        direction = rng.choice(["horizontal", "vertical", "radial"])
        gradient_mask(mask_path, width, height, dur, fps=fps,
                      direction=direction, cfg=cfg)

    dst = work_dir / f"comp_{tag}_selfkey.mp4"
    masked_composite(layers[base_idx], overlay, mask_path, dst, cfg=cfg)
    result = [l for i, l in enumerate(layers) if i not in (base_idx, overlay_idx)]
    result.append(dst)
    return result


def _op_chromakey(layers: list[Path], work_dir: Path, tag: str,
                  rng: random.Random, cfg: Config) -> list[Path]:
    """Chromakey: remove a random colour from overlay, show base through it."""
    if len(layers) < 2:
        return layers
    base_idx, overlay_idx = rng.sample(range(len(layers)), 2)
    hue = rng.choice(CHROMA_HUE_TARGETS)
    dst = work_dir / f"comp_{tag}_ckey.mp4"
    chromakey_composite(layers[base_idx], layers[overlay_idx], dst,
                        hue_center=hue, hue_range=rng.randint(10, 35),
                        sat_min=rng.randint(30, 70),
                        blur=rng.choice([3, 5]), cfg=cfg)
    result = [l for i, l in enumerate(layers) if i not in (base_idx, overlay_idx)]
    result.append(dst)
    return result


def _op_multi_pip(layers: list[Path], work_dir: Path, tag: str,
                  rng: random.Random, width: int, height: int,
                  cfg: Config) -> list[Path]:
    """Place 2-3 scaled overlays onto a single base at random positions."""
    n_overlays = min(rng.randint(2, 3), len(layers) - 1)
    if len(layers) < n_overlays + 1:
        return _op_pip(layers, work_dir, tag, rng, width, height, cfg)

    indices = rng.sample(range(len(layers)), n_overlays + 1)
    base_idx = indices[0]
    overlay_indices = indices[1:]

    current = layers[base_idx]
    for j, ov_idx in enumerate(overlay_indices):
        scale = 0.15 + rng.random() * 0.45
        pip_w = int(width * scale)
        max_x = max(0, width - pip_w)
        max_y = max(0, height - int(height * scale))
        x = rng.randint(0, max_x) if max_x > 0 else 0
        y = rng.randint(0, max_y) if max_y > 0 else 0
        step_dst = work_dir / f"comp_{tag}_mpip_{j}.mp4"
        picture_in_picture(current, layers[ov_idx], step_dst,
                           x=x, y=y, scale=scale, cfg=cfg)
        current = step_dst

    consumed = set(indices)
    result = [l for i, l in enumerate(layers) if i not in consumed]
    result.append(current)
    return result


# ─── Single-sample builder ──────────────────────────────────────────────────

@flow(name="build-composite-sample", log_prints=True)
def _build_sample(
    sample_idx: int,
    seed: int,
    segment_dur: float,
    width: int, height: int, fps: float,
    work_dir: Path, out_dir: Path,
    shader_dir: Optional[Path],
    brain_wipe_dir: Optional[Path],
    generators: dict,
    warpers: dict,
    min_warps: int,
    max_warps: int,
    cfg: Config,
) -> Path:
    """Build one composite sample with a random recipe."""
    rng = random.Random(seed)
    sample_work = work_dir / f"sample_{sample_idx:03d}"
    sample_work.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Generate layers ─────────────────────────────────────────
    n_layers = rng.randint(3, 6)
    layers: list[Path] = []

    # Weighted layer types — brain_wipe is dominant
    layer_types = ["brain_wipe", "solid"]
    layer_weights = [5, 1]

    for j in range(n_layers):
        lt = rng.choices(layer_types, weights=layer_weights, k=1)[0]
        tag = f"s{sample_idx:03d}_l{j}"

        if lt == "brain_wipe":
            layers.append(_make_layer_brain_wipe(
                segment_dur, width, height, fps,
                sample_work, tag, rng, generators, warpers,
                min_warps, max_warps, cfg))
        elif lt == "solid":
            layers.append(_make_layer_solid(
                segment_dur, width, height, fps,
                sample_work, tag, rng, cfg))

    # ── Step 1.5: Pre-process layers (random warp/shader) ────────────────
    preprocessed: list[Path] = []
    for j, layer in enumerate(layers):
        tag = f"s{sample_idx:03d}_pre{j}"
        preprocessed.append(
            _preprocess_layer(layer, sample_work, tag, rng,
                              shader_dir, brain_wipe_dir, cfg)
        )
    layers = preprocessed

    # ── Step 2: Random compositing operations ────────────────────────────
    n_ops = rng.randint(2, 5)
    op_pool = []
    for op_name, weight in OP_WEIGHTS.items():
        op_pool.extend([op_name] * weight)
    step = 0

    for _ in range(n_ops):
        if len(layers) < 2:
            break
        op = rng.choice(op_pool)
        tag = f"s{sample_idx:03d}_op{step}"
        step += 1

        if op == "blend":
            layers = _op_blend(layers, sample_work, tag, rng, cfg)
        elif op == "masked":
            layers = _op_masked(layers, sample_work, tag, rng,
                                 width, height, fps, segment_dur, cfg)
        elif op == "pip":
            layers = _op_pip(layers, sample_work, tag, rng, width, height, cfg)
        elif op == "self_keyed":
            layers = _op_self_keyed(layers, sample_work, tag, rng,
                                     width, height, fps, segment_dur, cfg)
        elif op == "chromakey":
            layers = _op_chromakey(layers, sample_work, tag, rng, cfg)
        elif op == "multi_pip":
            layers = _op_multi_pip(layers, sample_work, tag, rng,
                                    width, height, cfg)

    # If we still have multiple layers, blend them all down
    while len(layers) > 1:
        tag = f"s{sample_idx:03d}_final{step}"
        step += 1
        layers = _op_blend(layers, sample_work, tag, rng, cfg)

    result = layers[0]

    # ── Step 3: Optional post-processing ─────────────────────────────────
    if shader_dir and rng.random() < 0.5:
        post_shaded = sample_work / f"s{sample_idx:03d}_post_shader.mp4"
        apply_random_shader_stack(result, post_shaded, shader_dir,
                                   min_shaders=1, max_shaders=1,
                                   seed=rng.randint(0, 2**31), cfg=cfg)
        result = post_shaded

    if rng.random() < 0.5:
        post_norm = sample_work / f"s{sample_idx:03d}_post_norm.mp4"
        normalize_levels(result, post_norm, cfg=cfg)
        result = post_norm

    # ── Step 4: Write final output ───────────────────────────────────────
    recipe_hash = hashlib.sha1(f"comp_{sample_idx}_{seed}".encode()).hexdigest()[:8]
    dst = out_dir / f"comp_{sample_idx:03d}_{recipe_hash}.mp4"

    import shutil
    shutil.copy2(result, dst)
    return dst


# ─── Flow ────────────────────────────────────────────────────────────────────

@flow(name="compositing-lab", log_prints=True)
def compositing_lab(
    n_samples: int = 8,
    seed: Optional[int] = None,
    segment_dur: float = 10.0,
    width: int = 1280,
    height: int = 720,
    fps: float = 30.0,
    shader_dir: Optional[Path] = None,
    brain_wipe_dir: Path = Path("brain-wipe-shaders"),
    min_warps: int = 0,
    max_warps: int = 2,
    output_dir: Optional[Path] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """
    Generate N random composite samples from brain wipe layers.

    Each sample gets a unique random recipe: brain wipe renders (generator
    shaders + optional warp chains), TV static, and solid colours composed
    through random blends, masked merges, keying, and picture-in-picture.
    No source footage needed.
    """
    c = cfg or Config()
    c.ensure_dirs()

    out_dir = Path(output_dir) if output_dir else c.output_dir / "comp_lab"
    out_dir.mkdir(parents=True, exist_ok=True)
    work = c.work_dir / "comp_lab"
    work.mkdir(parents=True, exist_ok=True)

    base_seed = seed if seed is not None else random.randint(0, 2**31)

    # Load brain wipe shader library
    all_bw_shaders = load_shader_dir(brain_wipe_dir)
    generators = filter_shaders(all_bw_shaders, has_image_input=False)
    warpers = filter_shaders(all_bw_shaders, has_image_input=True)

    if not generators:
        raise ValueError(
            f"No generator shaders found in {brain_wipe_dir}. "
            f"Generator shaders are required for compositing lab."
        )

    print(f"Brain wipe library: {len(generators)} generators, "
          f"{len(warpers)} warpers from {brain_wipe_dir}")

    # Validate shader_dir (used for preprocessing warps + post-processing)
    s_dir = shader_dir
    if s_dir and not s_dir.exists():
        print(f"Warning: shader_dir {s_dir} not found, disabling shader extras")
        s_dir = None

    print(f"Compositing lab: {n_samples} samples, {segment_dur}s each, "
          f"{width}x{height}@{fps}fps, seed={base_seed}")
    print(f"Warps per layer: {min_warps}–{max_warps}")
    print(f"Output: {out_dir}/\n")

    # Run samples as concurrent subflows via ThreadPoolExecutor.
    # Each subflow gets its own flow run in the Prefect UI, with all
    # inner task runs scoped to it for full observability.
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Resolve brain_wipe_dir for pre-processing warp shaders
    bw_dir = brain_wipe_dir if brain_wipe_dir.exists() else None

    def _run_sample(i):
        return i, _build_sample(
            sample_idx=i,
            seed=base_seed + i,
            segment_dur=segment_dur,
            width=width, height=height, fps=fps,
            work_dir=work, out_dir=out_dir,
            shader_dir=s_dir,
            brain_wipe_dir=bw_dir,
            generators=generators,
            warpers=warpers,
            min_warps=min_warps,
            max_warps=max_warps,
            cfg=c,
        )

    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_run_sample, i): i
                   for i in range(n_samples)}
        for future in as_completed(futures):
            i, dst = future.result()
            results.append(dst)
            print(f"  done: sample {i:03d} → {dst.name}")

    print(f"\n{len(results)} samples complete → {out_dir}/")
    return out_dir


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _cli():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compositing Lab — brain wipe composite explorer.",
    )
    parser.add_argument("-n", "--n-samples", type=int, default=8,
                        help="Number of composite samples (default: 8)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--segment-dur", type=float, default=10.0,
                        help="Duration per sample in seconds (default: 10)")
    parser.add_argument("--width", type=int, default=1280,
                        help="Output width (default: 1280)")
    parser.add_argument("--height", type=int, default=720,
                        help="Output height (default: 720)")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Output frame rate (default: 30)")
    parser.add_argument("--shader-dir", type=Path, default=None,
                        help="Glitch shader directory for pre/post-processing")
    parser.add_argument("--brain-wipe-dir", type=Path,
                        default=Path("brain-wipe-shaders"),
                        help="Brain wipe shader directory (default: brain-wipe-shaders)")
    parser.add_argument("--min-warps", type=int, default=0,
                        help="Min warp shaders per brain wipe layer (default: 0)")
    parser.add_argument("--max-warps", type=int, default=2,
                        help="Max warp shaders per brain wipe layer (default: 2)")
    parser.add_argument("-o", "--output-dir", type=Path, default=None,
                        help="Output directory (default: output/comp_lab)")

    args = parser.parse_args()
    cfg = Config()
    cfg.ensure_dirs()

    compositing_lab(
        n_samples=args.n_samples,
        seed=args.seed,
        segment_dur=args.segment_dur,
        width=args.width,
        height=args.height,
        fps=args.fps,
        shader_dir=args.shader_dir,
        brain_wipe_dir=args.brain_wipe_dir,
        min_warps=args.min_warps,
        max_warps=args.max_warps,
        output_dir=args.output_dir,
        cfg=cfg,
    )


if __name__ == "__main__":
    _cli()
