"""
Example Prefect flows demonstrating pipeline composition.

These are starting points — copy and customise for your own workflows.
Run any flow with:
    python -m pipeline.flows.examples          (runs all examples)
    prefect deployment run 'flow-name/...'     (via Prefect server)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from prefect import flow

from ..config import Config
from ..tasks import (
    detect_cuts, segment_at_cuts, random_segments,
    concat_clips, shuffle_clips, interleave_clips,
    generate_static,
    apply_shader, apply_shader_stack, apply_random_shader_stack,
    blend_layers, masked_composite, multi_layer_composite,
    luma_mask, edge_mask, motion_mask, gradient_mask,
)
from ..tasks.cut import sweep_thresholds
from ..isf import load_shader_dir
from .stooges import stooges_channels
from .brain_wipe import warp_chain, brain_wipe_render


# ─── Flow 1: Cut → Shuffle → Shader ─────────────────────────────────────────

@flow(name="cut-shuffle-shader")
def cut_shuffle_shader(
    src: Path,
    shader_path: Path,
    output: Optional[Path] = None,
    scene_threshold: float = 30.0,
    cfg: Optional[Config] = None,
) -> Path:
    """
    1. Detect scene cuts in source footage
    2. Segment at cut points
    3. Shuffle the segments randomly
    4. Apply a single ISF shader to the shuffled result
    """
    from ..ffmpeg import probe as _probe

    c = cfg or Config()
    c.ensure_dirs()
    out = output or c.output_dir / "cut_shuffle_shader.mp4"

    if c.default_video_bitrate is None:
        _info = _probe(src, c)
        if _info.bitrate > 0:
            c.default_video_bitrate = _info.bitrate

    cuts = detect_cuts(src, threshold=scene_threshold, cfg=c)
    segments = segment_at_cuts(src, cuts, cfg=c)
    shuffled = c.work_dir / "shuffled.mp4"
    shuffle_clips(segments, shuffled, cfg=c)
    apply_shader(shuffled, out, shader_path, cfg=c)

    return out


# ─── Flow 2: Random Segments → Shader Stack ─────────────────────────────────

@flow(name="random-shader-collage")
def random_shader_collage(
    src: Path,
    count: int = 8,
    min_dur: float = 5.0,
    max_dur: float = 15.0,
    output: Optional[Path] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """
    1. Pull N random segments from source footage
    2. Apply a random shader stack to each segment
    3. Concatenate the results
    """
    from ..ffmpeg import probe as _probe

    c = cfg or Config()
    c.ensure_dirs()
    out = output or c.output_dir / "random_shader_collage.mp4"

    if c.default_video_bitrate is None:
        _info = _probe(src, c)
        if _info.bitrate > 0:
            c.default_video_bitrate = _info.bitrate

    segments = random_segments(src, count,
                               min_dur=min_dur, max_dur=max_dur, cfg=c)

    processed = []
    for i, seg in enumerate(segments):
        dst = c.work_dir / f"shaded_{i:04d}.mp4"
        apply_random_shader_stack(seg, dst,
                                   min_shaders=1, max_shaders=3,
                                   seed=i, cfg=c)
        processed.append(dst)

    concat_clips(processed, out, cfg=c)
    return out


# ─── Flow 3: Density Composite ──────────────────────────────────────────────

@flow(name="density-composite")
def density_composite(
    sources: list[Path],
    output: Optional[Path] = None,
    mode: str = "screen",
    opacity: float = 0.4,
    cfg: Optional[Config] = None,
) -> Path:
    """
    Layer multiple source videos on top of each other using a blend mode.
    Builds visual density by stacking footage.

    sources: at least 2 video paths (same resolution).
    """
    c = cfg or Config()
    c.ensure_dirs()
    out = output or c.output_dir / "density_composite.mp4"

    if len(sources) < 2:
        raise ValueError("Need at least 2 source videos")

    # Build layers: base at full opacity, overlays at specified opacity
    layers = [(sources[0], 1.0, "normal")]
    for s in sources[1:]:
        layers.append((s, opacity, mode))

    multi_layer_composite(layers, out, cfg=c)
    return out


# ─── Flow 4: Masked Shader Overlay ──────────────────────────────────────────

@flow(name="masked-shader-overlay")
def masked_shader_overlay(
    src: Path,
    shader_path: Path,
    mask_type: str = "edge",
    output: Optional[Path] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """
    1. Apply a shader to the source footage
    2. Generate a mask from the original (edge, motion, or luma)
    3. Composite: original shows through where mask is black,
       shader-processed shows through where mask is white.

    This creates a selective shader effect driven by the footage itself.
    """
    from ..ffmpeg import probe as _probe

    c = cfg or Config()
    c.ensure_dirs()
    out = output or c.output_dir / "masked_shader_overlay.mp4"

    if c.default_video_bitrate is None:
        _info = _probe(src, c)
        if _info.bitrate > 0:
            c.default_video_bitrate = _info.bitrate

    # Shader pass
    shaded = c.work_dir / "shaded_full.mp4"
    apply_shader(src, shaded, shader_path, cfg=c)

    # Mask pass
    mask_path = c.work_dir / "auto_mask.mp4"
    if mask_type == "edge":
        edge_mask(src, mask_path, low=40, high=120, dilate=2, cfg=c)
    elif mask_type == "motion":
        motion_mask(src, mask_path, threshold=20, blur=7, cfg=c)
    elif mask_type == "luma":
        luma_mask(src, mask_path, threshold=0.5, cfg=c)
    else:
        raise ValueError(f"Unknown mask type: {mask_type}")

    # Composite
    masked_composite(src, shaded, mask_path, out, cfg=c)
    return out


# ─── Flow 5: Texture Builder ────────────────────────────────────────────────

@flow(name="texture-builder")
def texture_builder(
    src: Path,
    shader_paths: list[Path],
    output: Optional[Path] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """
    Build a dense textural video by:
    1. Creating a motion mask from the source
    2. Creating an edge mask from the source
    3. Applying a shader stack to the source
    4. Blending the shader output with static noise via the motion mask
    5. Overlaying edges in 'add' mode for definition

    Produces thick, abstract, textural results.
    """
    from ..ffmpeg import probe

    c = cfg or Config()
    c.ensure_dirs()
    out = output or c.output_dir / "texture_builder.mp4"

    info = probe(src, c)
    if c.default_video_bitrate is None and info.bitrate > 0:
        c.default_video_bitrate = info.bitrate

    # Parallel mask generation
    motion_path = c.work_dir / "motion_mask.mp4"
    edge_path = c.work_dir / "edge_mask.mp4"
    motion_mask(src, motion_path, threshold=15, blur=5, cfg=c)
    edge_mask(src, edge_path, low=30, high=100, dilate=1, cfg=c)

    # Shader stack on source
    shaded = c.work_dir / "shaded_texture.mp4"
    apply_shader_stack(src, shaded, shader_paths, cfg=c)

    # Generate noise layer
    noise = c.work_dir / "noise_texture.mp4"
    generate_static(noise, duration=info.duration,
                    width=info.width, height=info.height,
                    fps=info.fps, cfg=c)

    # Blend shader + noise through motion mask
    motion_blend = c.work_dir / "motion_blend.mp4"
    masked_composite(shaded, noise, motion_path, motion_blend, cfg=c)

    # Add edges on top
    blend_layers(motion_blend, edge_path, out,
                 mode="add", opacity=0.3, cfg=c)

    return out


# ─── Flow 6: Shuffled Scene Shaders ────────────────────────────────────────

@flow(name="shuffled-scene-shaders")
def shuffled_scene_shaders(
    src: Path,
    output: Optional[Path] = None,
    scene_threshold: float = 30.0,
    min_shaders: int = 1,
    max_shaders: int = 4,
    seed: Optional[int] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """
    1. Detect scene cuts in source footage
    2. Segment at cut points
    3. Apply a different random shader stack to each segment
    4. Shuffle the processed segments
    5. Concatenate into final output

    Each segment gets a unique shader selection and randomised parameters.
    Pass seed for reproducible runs (segment i uses seed + i).
    """
    from ..ffmpeg import probe as _probe

    c = cfg or Config()
    c.ensure_dirs()
    out = output or c.output_dir / "shuffled_scene_shaders.mp4"

    if c.default_video_bitrate is None:
        _info = _probe(src, c)
        if _info.bitrate > 0:
            c.default_video_bitrate = _info.bitrate

    cuts = detect_cuts(src, threshold=scene_threshold, cfg=c)
    segments = segment_at_cuts(src, cuts, cfg=c)

    # Submit shader tasks concurrently — each returns a future
    futures = []
    processed = []
    for i, seg in enumerate(segments):
        dst = c.work_dir / f"scene_shaded_{i:04d}.mp4"
        seg_seed = (seed + i) if seed is not None else None
        future = apply_random_shader_stack.submit(
            seg, dst,
            min_shaders=min_shaders, max_shaders=max_shaders,
            seed=seg_seed, cfg=c,
        )
        futures.append(future)
        processed.append(dst)

    # Wait for all shader tasks to finish
    for future in futures:
        future.result()

    shuffle_clips(processed, out, seed=seed, cfg=c)
    return out


# ─── Flow 7: Deep Color ────────────────────────────────────────────────────

@flow(name="deep-color", log_prints=True)
def deep_color(
    src: Path,
    output: Optional[Path] = None,
    scene_threshold: float = 30.0,
    shaders_per_pass: int = 3,
    crush: float = 0.7,
    crush_codec: str = "libx264",
    density_mode: str = "screen",
    density_opacity: float = 0.35,
    edge_strength: float = 0.25,
    seed: Optional[int] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """
    Transform narrative B&W footage into colorful, textured, layered,
    partly abstract video.

    Per-scene processing is a crush-shader sandwich:
      crush → N shaders → crush → N shaders → normalize

    The first crush introduces compression texture, the first shader
    pass colorizes/transforms it, the second crush bakes those artifacts
    and adds a new layer of degradation, and the second shader pass
    processes everything again. Normalization rescues the levels.

    Then scenes are reassembled, composited with a shuffled ghost layer
    gated by motion, and edge structure is added back.

    shaders_per_pass: shaders per crush-shader pass (default 3, total 6)
    crush:            0.0–1.0 crush intensity (0=mild, 1=max destruction)
    """
    import random as rng
    from ..ffmpeg import probe
    from ..tasks.glitch import bitrate_crush
    from ..tasks.color import normalize_levels

    c = cfg or Config()
    c.ensure_dirs()
    out = output or c.output_dir / "deep_color.mp4"

    # Match source bitrate for all re-encoding in this flow
    if c.default_video_bitrate is None:
        info = probe(src, c)
        if info.bitrate > 0:
            c.default_video_bitrate = info.bitrate

    # Load shader library from all active packs
    all_shaders = {}
    for s_dir in c.pack_shader_dirs():
        all_shaders.update(load_shader_dir(s_dir))
    if not all_shaders:
        raise ValueError("No .fs shaders found in any pack (packs/*/shaders/)")
    shader_names = list(all_shaders.keys())
    print(f"Shader library: {len(all_shaders)} shaders from {len(c.pack_shader_dirs())} pack(s)")
    print(f"Pipeline: crush({crush:.0%}) → {shaders_per_pass} shaders "
          f"→ crush({crush:.0%}) → {shaders_per_pass} shaders → normalize")

    # ── Phase 1: Scene segmentation ──────────────────────────────────────
    cuts = detect_cuts(src, threshold=scene_threshold, cfg=c)
    segments = segment_at_cuts(src, cuts, cfg=c)
    print(f"\n{len(segments)} scenes detected.")

    # ── Phase 2: Masks (run during scene processing) ─────────────────────
    motion_path = c.work_dir / "dc_motion.mp4"
    edge_path = c.work_dir / "dc_edges.mp4"

    f_motion = motion_mask.submit(src, motion_path,
                                  threshold=20, blur=7, cfg=c)
    f_edge = edge_mask.submit(src, edge_path,
                              low=30, high=120, dilate=1, cfg=c)

    # ── Phase 3: Crush-shader sandwich per scene ─────────────────────────
    # Each scene runs its full pipeline independently:
    #   crush → shaders → crush → shaders → normalize → copy to scenes/
    # Intermediate results surface to scenes_dir as they finish.
    import shutil

    if seed is not None:
        rng.seed(seed)

    scenes_dir = out.parent / "deep_color_scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)
    print(f"Scene previews → {scenes_dir}/")

    def _pick_shaders(n, rng_inst):
        """Pick n shaders and randomize their float params."""
        chosen = rng_inst.sample(shader_names, min(n, len(shader_names)))
        shaders = [all_shaders[name] for name in chosen]
        overrides = {}
        for shader in shaders:
            sp = {}
            for inp in shader.param_inputs:
                if inp.type == "float" and inp.min is not None and inp.max is not None:
                    sp[inp.name] = round(
                        inp.min + rng_inst.random() * (inp.max - inp.min), 3
                    )
            if sp:
                overrides[shader.path.stem] = sp
        return shaders, overrides

    def _print_recipe(shaders, overrides, indent="    "):
        for j, s in enumerate(shaders):
            params = overrides.get(s.path.stem, {})
            param_str = "  ".join(f"{k}={v}" for k, v in params.items())
            print(f"{indent}{j+1}. {s.path.stem:<20s} {param_str}")

    # Pre-generate all recipes so we can print them before processing
    scene_plans = []
    for i in range(len(segments)):
        seg_seed_1 = (seed + i) if seed is not None else None
        seg_seed_2 = (seed + i + 1000) if seed is not None else None

        if seg_seed_1 is not None:
            rng.seed(seg_seed_1)
        s1, o1 = _pick_shaders(shaders_per_pass, rng)

        if seg_seed_2 is not None:
            rng.seed(seg_seed_2)
        s2, o2 = _pick_shaders(shaders_per_pass, rng)

        scene_plans.append((s1, o1, s2, o2))

        print(f"\nscene {i:03d}:")
        print(f"  crush {crush:.0%} →")
        _print_recipe(s1, o1)
        print(f"  crush {crush:.0%} →")
        _print_recipe(s2, o2)

    # Process each scene: crush → shader → crush → shader → normalize
    # Scenes run in parallel via threads; steps within a scene are sequential.
    # Each scene copies its result to scenes_dir as soon as it finishes.
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _process_scene(seg, idx, shaders1, overrides1, shaders2, overrides2):
        tag = f"dc_{idx:04d}"

        # Step 1: crush
        crush1 = c.work_dir / f"{tag}_crush1.mp4"
        bitrate_crush.fn(seg, crush1, crush=crush,
                         codec=crush_codec, cfg=c)

        # Step 2: first shader stack
        pass1 = c.work_dir / f"{tag}_pass1.mp4"
        apply_shader_stack.fn(
            crush1, pass1, [s.path for s in shaders1],
            param_overrides=overrides1, cfg=c,
        )

        # Step 3: crush again
        crush2 = c.work_dir / f"{tag}_crush2.mp4"
        bitrate_crush.fn(pass1, crush2, crush=crush,
                         codec=crush_codec, cfg=c)

        # Step 4: second shader stack
        pass2 = c.work_dir / f"{tag}_pass2.mp4"
        apply_shader_stack.fn(
            crush2, pass2, [s.path for s in shaders2],
            param_overrides=overrides2, cfg=c,
        )

        # Step 5: normalize
        final = c.work_dir / f"{tag}_final.mp4"
        normalize_levels.fn(pass2, final, cfg=c)

        # Copy to scenes dir for immediate preview
        preview = scenes_dir / f"scene_{idx:03d}.mp4"
        shutil.copy2(final, preview)
        print(f"  scene {idx:03d} done → {preview.name}")

        return final

    print(f"\nProcessing {len(segments)} scenes...")
    final_segs = [None] * len(segments)
    # Cap threads — each scene runs ffmpeg subprocesses, don't overwhelm
    max_workers = min(4, len(segments))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_idx = {}
        for i, seg in enumerate(segments):
            s1, o1, s2, o2 = scene_plans[i]
            fut = pool.submit(_process_scene, seg, i, s1, o1, s2, o2)
            future_to_idx[fut] = i

        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            final_segs[idx] = fut.result()

    # ── Phase 4: Reassemble + shuffle (parallel) ─────────────────────────
    colorized_full = c.work_dir / "dc_colorized.mp4"
    shuffled_full = c.work_dir / "dc_shuffled.mp4"

    f_concat = concat_clips.submit(final_segs, colorized_full, cfg=c)
    f_shuffle = shuffle_clips.submit(final_segs, shuffled_full,
                                     seed=seed, cfg=c)

    # Wait for everything before compositing
    f_concat.result()
    f_shuffle.result()
    f_motion.result()
    f_edge.result()

    # ── Phase 5: Compositing stack (sequential) ──────────────────────────
    print("\nCompositing...")

    # 1. Blend ghost layer over colorized → dense, layered image
    dense = c.work_dir / "dc_dense.mp4"
    blend_layers(colorized_full, shuffled_full, dense,
                 mode=density_mode, opacity=density_opacity, cfg=c)

    # 2. Motion mask gates the density: motion → abstract, still → clean
    selective = c.work_dir / "dc_selective.mp4"
    masked_composite(colorized_full, dense, motion_path, selective, cfg=c)

    # 3. Add edge structure back for definition
    blend_layers(selective, edge_path, out,
                 mode="softlight", opacity=edge_strength, cfg=c)

    print(f"\nOutput: {out}")
    return out


# ─── Flow 8: Scene Split (diagnostic) ─────────────────────────────────────

@flow(name="scene-split")
def scene_split(
    src: Path,
    output_dir: Optional[Path] = None,
    scene_threshold: float = 27.0,
    adaptive_threshold: float = 5.0,
    method: str = "adaptive",
    luma_only: bool = True,
    min_segment: float = 2.0,
    cfg: Optional[Config] = None,
) -> list[Path]:
    """
    Diagnostic flow: detect scene cuts and save each scene as a separate file.

    Prints cut timestamps, segment count, and per-segment durations
    so you can verify scene detection before feeding into other flows.
    """
    from ..ffmpeg import probe

    c = cfg or Config()
    c.ensure_dirs()
    out_dir = output_dir or c.output_dir / "scenes"
    out_dir.mkdir(parents=True, exist_ok=True)

    info = probe(src, c)
    print(f"Source: {src.name}  duration={info.duration:.1f}s  "
          f"{info.width}x{info.height} @ {info.fps:.2f}fps")

    cuts = detect_cuts(src, threshold=scene_threshold,
                       adaptive_threshold=adaptive_threshold,
                       method=method, luma_only=luma_only, cfg=c)
    print(f"\nDetected {len(cuts)} cuts (method={method}, luma_only={luma_only}):")
    for i, t in enumerate(cuts):
        print(f"  cut {i:3d}  @ {t:.2f}s")

    segments = segment_at_cuts(src, cuts, min_segment=min_segment,
                                output_dir=out_dir, cfg=c)

    print(f"\n{len(segments)} segments (min_segment={min_segment}s):")
    total = 0.0
    for seg in segments:
        seg_info = probe(seg, c)
        total += seg_info.duration
        print(f"  {seg.name}  {seg_info.duration:.2f}s")

    print(f"\nTotal segment duration: {total:.1f}s "
          f"(source: {info.duration:.1f}s, "
          f"diff: {info.duration - total:+.1f}s)")

    return segments


# ─── Flow 9: Shader Lab ───────────────────────────────────────────────────

@flow(name="shader-lab", log_prints=True)
def shader_lab(
    src: Path,
    count: int = 4,
    duration: float = 30.0,
    min_shaders: int = 1,
    max_shaders: int = 3,
    seed: Optional[int] = None,
    output_dir: Optional[Path] = None,
    cfg: Optional[Config] = None,
) -> list[Path]:
    """
    Exploratory shader stack tester.

    Grabs N random clips from source footage and applies a different random
    shader stack to each one. Prints the full recipe (shaders + params) for
    every sample so you can identify combinations worth keeping.

    No scene detection, no compositing — just raw shader stacks on footage.
    """
    import hashlib
    import random as rng
    from ..ffmpeg import probe

    c = cfg or Config()
    c.ensure_dirs()
    out_dir = output_dir or c.output_dir / "shader_lab"
    out_dir.mkdir(parents=True, exist_ok=True)

    info = probe(src, c)
    print(f"Source: {src.name}  duration={info.duration:.1f}s  "
          f"{info.width}x{info.height} @ {info.fps:.2f}fps  "
          f"{info.bitrate} kbps")

    # Match source bitrate for all re-encoding in this flow
    if c.default_video_bitrate is None and info.bitrate > 0:
        c.default_video_bitrate = info.bitrate

    # Load all available shaders from active packs
    all_shaders = {}
    for s_dir in c.pack_shader_dirs():
        all_shaders.update(load_shader_dir(s_dir))
    if not all_shaders:
        raise ValueError("No .fs shaders found in any pack (packs/*/shaders/)")
    print(f"Shader library: {len(all_shaders)} shaders from {len(c.pack_shader_dirs())} pack(s)")

    # Extract random clips
    segments = random_segments(src, count,
                                min_dur=duration, max_dur=duration,
                                output_dir=c.work_dir / "shader_lab_clips",
                                cfg=c)

    # Build shader stacks with full logging, then submit concurrently
    if seed is not None:
        rng.seed(seed)

    futures = []
    outputs = []
    for i, seg in enumerate(segments):
        sample_seed = (seed + i) if seed is not None else None
        if sample_seed is not None:
            rng.seed(sample_seed)

        n = rng.randint(min_shaders, min(max_shaders, len(all_shaders)))
        chosen_names = rng.sample(list(all_shaders.keys()), n)
        shaders = [all_shaders[name] for name in chosen_names]
        shader_paths = [s.path for s in shaders]

        # Randomise float params
        overrides: dict[str, dict[str, float]] = {}
        for shader in shaders:
            sp = {}
            for inp in shader.param_inputs:
                if inp.type == "float" and inp.min is not None and inp.max is not None:
                    sp[inp.name] = round(inp.min + rng.random() * (inp.max - inp.min), 3)
            if sp:
                overrides[shader.path.stem] = sp

        # Build a short hash from the recipe for the filename
        recipe_str = ";".join(
            f"{s.path.stem}:" + ",".join(
                f"{k}={v}" for k, v in sorted(overrides.get(s.path.stem, {}).items())
            )
            for s in shaders
        )
        tag = hashlib.sha1(recipe_str.encode()).hexdigest()[:8]

        # Print the recipe
        seed_str = f" (seed={sample_seed})" if sample_seed is not None else ""
        print(f"\nsample {tag}{seed_str}:")
        for j, shader in enumerate(shaders):
            params = overrides.get(shader.path.stem, {})
            param_str = "  ".join(f"{k}={v}" for k, v in params.items())
            print(f"  {j+1}. {shader.path.stem:<20s} {param_str}")

        dst = out_dir / f"sample_{tag}.mp4"
        future = apply_shader_stack.submit(
            seg, dst, shader_paths,
            param_overrides=overrides, cfg=c,
        )
        futures.append(future)
        outputs.append(dst)

    # Wait for shader stacks to finish
    for future in futures:
        future.result()

    # Normalize levels on each sample — fixes cumulative darkening from stacking
    from ..tasks.color import normalize_levels
    print("\nNormalizing levels...")
    norm_futures = []
    for dst in outputs:
        norm_tmp = dst.with_suffix(".norm.mp4")
        norm_futures.append(normalize_levels.submit(dst, norm_tmp, cfg=c))

    for nf, dst in zip(norm_futures, outputs):
        norm_tmp = nf.result()
        # Replace original with normalized version
        norm_tmp.rename(dst)

    print(f"\n{len(outputs)} samples written to {out_dir}/")
    return outputs


# ─── Flow 10: Crush Lab ──────────────────────────────────────────────────────

@flow(name="crush-lab", log_prints=True)
def crush_lab(
    src: Path,
    count: int = 4,
    duration: float = 30.0,
    min_shaders: int = 1,
    max_shaders: int = 6,
    crush: float = 0.7,
    crush_codec: str = "libx264",
    seed: Optional[int] = None,
    output_dir: Optional[Path] = None,
    cfg: Optional[Config] = None,
) -> list[Path]:
    """
    Crush random clips then apply random shader stacks.

    The crush pass introduces compression artifacts (macroblocking, DCT
    ringing, temporal smearing) which the shader stack then transforms
    into something interesting. Artifacts become texture.

    crush:       0.0–1.0 crush intensity (0=mild, 1=max destruction)
    crush_codec: libx264, mpeg2video, or mpeg4
    """
    import hashlib
    import random as rng
    from ..ffmpeg import probe
    from ..tasks.glitch import bitrate_crush
    from ..tasks.color import normalize_levels

    c = cfg or Config()
    c.ensure_dirs()
    out_dir = output_dir or c.output_dir / "crush_lab"
    out_dir.mkdir(parents=True, exist_ok=True)

    info = probe(src, c)
    print(f"Source: {src.name}  duration={info.duration:.1f}s  "
          f"{info.width}x{info.height} @ {info.fps:.2f}fps  "
          f"{info.bitrate} kbps")

    if c.default_video_bitrate is None and info.bitrate > 0:
        c.default_video_bitrate = info.bitrate

    all_shaders = {}
    for s_dir in c.pack_shader_dirs():
        all_shaders.update(load_shader_dir(s_dir))
    if not all_shaders:
        raise ValueError("No .fs shaders found in any pack (packs/*/shaders/)")
    print(f"Shader library: {len(all_shaders)} shaders from {len(c.pack_shader_dirs())} pack(s)")
    print(f"Crush: {crush:.0%} intensity, codec={crush_codec}")

    # Extract random clips
    segments = random_segments(src, count,
                                min_dur=duration, max_dur=duration,
                                output_dir=c.work_dir / "crush_lab_clips",
                                cfg=c)

    # Crush all clips concurrently
    print("\nCrushing clips...")
    crush_futures = []
    crushed_clips = []
    for i, seg in enumerate(segments):
        crushed = c.work_dir / f"crush_lab_crushed_{i:04d}.mp4"
        future = bitrate_crush.submit(
            seg, crushed,
            crush=crush, codec=crush_codec, cfg=c,
        )
        crush_futures.append(future)
        crushed_clips.append(crushed)

    for future in crush_futures:
        future.result()
    print("Crush pass done.")

    # Build shader stacks with full logging, then submit concurrently
    if seed is not None:
        rng.seed(seed)

    futures = []
    outputs = []
    for i, seg in enumerate(crushed_clips):
        sample_seed = (seed + i) if seed is not None else None
        if sample_seed is not None:
            rng.seed(sample_seed)

        n = rng.randint(min_shaders, min(max_shaders, len(all_shaders)))
        chosen_names = rng.sample(list(all_shaders.keys()), n)
        shaders = [all_shaders[name] for name in chosen_names]
        shader_paths = [s.path for s in shaders]

        overrides: dict[str, dict[str, float]] = {}
        for shader in shaders:
            sp = {}
            for inp in shader.param_inputs:
                if inp.type == "float" and inp.min is not None and inp.max is not None:
                    sp[inp.name] = round(inp.min + rng.random() * (inp.max - inp.min), 3)
            if sp:
                overrides[shader.path.stem] = sp

        recipe_str = f"crush:{crush:.2f}/{crush_codec};" + ";".join(
            f"{s.path.stem}:" + ",".join(
                f"{k}={v}" for k, v in sorted(overrides.get(s.path.stem, {}).items())
            )
            for s in shaders
        )
        tag = hashlib.sha1(recipe_str.encode()).hexdigest()[:8]

        seed_str = f" (seed={sample_seed})" if sample_seed is not None else ""
        print(f"\nsample {tag}{seed_str}:")
        print(f"  crush: {crush:.0%} {crush_codec}")
        for j, shader in enumerate(shaders):
            params = overrides.get(shader.path.stem, {})
            param_str = "  ".join(f"{k}={v}" for k, v in params.items())
            print(f"  {j+1}. {shader.path.stem:<20s} {param_str}")

        dst = out_dir / f"sample_{tag}.mp4"
        future = apply_shader_stack.submit(
            seg, dst, shader_paths,
            param_overrides=overrides, cfg=c,
        )
        futures.append(future)
        outputs.append(dst)

    for future in futures:
        future.result()

    # Normalize
    print("\nNormalizing levels...")
    norm_futures = []
    for dst in outputs:
        norm_tmp = dst.with_suffix(".norm.mp4")
        norm_futures.append(normalize_levels.submit(dst, norm_tmp, cfg=c))

    for nf, dst in zip(norm_futures, outputs):
        norm_tmp = nf.result()
        norm_tmp.rename(dst)

    print(f"\n{len(outputs)} samples written to {out_dir}/")
    return outputs


# ─── CLI entry point ────────────────────────────────────────────────────────

FLOWS = {
    "cut-shuffle-shader": cut_shuffle_shader,
    "random-shader-collage": random_shader_collage,
    "density-composite": density_composite,
    "masked-shader-overlay": masked_shader_overlay,
    "texture-builder": texture_builder,
    "shuffled-scene-shaders": shuffled_scene_shaders,
    "deep-color": deep_color,
    "scene-split": scene_split,
    "shader-lab": shader_lab,
    "crush-lab": crush_lab,
    "stooges-channels": stooges_channels,
    "warp-chain": warp_chain,
    "brain-wipe-render": brain_wipe_render,
}


def _cli():
    import argparse

    parser = argparse.ArgumentParser(
        description="ULP Video Pipeline — run Prefect flows from the command line.",
    )
    sub = parser.add_subparsers(dest="flow", required=True)

    # ── cut-shuffle-shader ──
    p = sub.add_parser("cut-shuffle-shader",
                       help="Detect scenes, shuffle, apply one shader")
    p.add_argument("src", type=Path)
    p.add_argument("shader", type=Path, help="ISF shader file")
    p.add_argument("-o", "--output", type=Path)
    p.add_argument("--threshold", type=float, default=30.0)

    # ── random-shader-collage ──
    p = sub.add_parser("random-shader-collage",
                       help="Random segments with random shader stacks")
    p.add_argument("src", type=Path)
    p.add_argument("--pack", action="append", dest="packs",
                   help="Restrict to specific shader packs (repeatable)")
    p.add_argument("-n", "--count", type=int, default=8)
    p.add_argument("--min-dur", type=float, default=5.0)
    p.add_argument("--max-dur", type=float, default=15.0)
    p.add_argument("-o", "--output", type=Path)

    # ── density-composite ──
    p = sub.add_parser("density-composite",
                       help="Layer multiple videos with a blend mode")
    p.add_argument("sources", type=Path, nargs="+")
    p.add_argument("-o", "--output", type=Path)
    p.add_argument("--mode", default="screen")
    p.add_argument("--opacity", type=float, default=0.4)

    # ── masked-shader-overlay ──
    p = sub.add_parser("masked-shader-overlay",
                       help="Shader + auto-mask composite")
    p.add_argument("src", type=Path)
    p.add_argument("shader", type=Path, help="ISF shader file")
    p.add_argument("--mask-type", default="edge",
                   choices=["edge", "motion", "luma"])
    p.add_argument("-o", "--output", type=Path)

    # ── texture-builder ──
    p = sub.add_parser("texture-builder",
                       help="Dense textural video from shader stack + masks")
    p.add_argument("src", type=Path)
    p.add_argument("shaders", type=Path, nargs="+", help="ISF shader files")
    p.add_argument("-o", "--output", type=Path)

    # ── shuffled-scene-shaders ──
    p = sub.add_parser("shuffled-scene-shaders",
                       help="Detect scenes, per-scene random shaders, shuffle")
    p.add_argument("src", type=Path)
    p.add_argument("--pack", action="append", dest="packs",
                   help="Restrict to specific shader packs (repeatable)")
    p.add_argument("-o", "--output", type=Path)
    p.add_argument("--threshold", type=float, default=30.0)
    p.add_argument("--min-shaders", type=int, default=1)
    p.add_argument("--max-shaders", type=int, default=4)
    p.add_argument("--seed", type=int, default=None)

    # ── deep-color ──
    p = sub.add_parser("deep-color",
                       help="B&W narrative → crush-shader sandwich → composite")
    p.add_argument("src", type=Path)
    p.add_argument("--pack", action="append", dest="packs",
                   help="Restrict to specific shader packs (repeatable)")
    p.add_argument("-o", "--output", type=Path)
    p.add_argument("--threshold", type=float, default=30.0)
    p.add_argument("--shaders-per-pass", type=int, default=3,
                   help="Shaders per crush-shader pass (default: 3, total 2x)")
    p.add_argument("--crush", type=float, default=0.7,
                   help="Crush intensity 0.0–1.0 (default: 0.7)")
    p.add_argument("--crush-codec", default="libx264",
                   choices=["libx264", "mpeg2video", "mpeg4"])
    p.add_argument("--density-mode", default="screen",
                   choices=["screen", "add", "overlay", "softlight"])
    p.add_argument("--density-opacity", type=float, default=0.35)
    p.add_argument("--edge-strength", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=None)

    # ── scene-split ──
    p = sub.add_parser("scene-split",
                       help="Diagnostic: detect scenes and save each as a file")
    p.add_argument("src", type=Path)
    p.add_argument("-o", "--output-dir", type=Path)
    p.add_argument("--method", default="adaptive",
                   choices=["adaptive", "content"],
                   help="Detection method (default: adaptive)")
    p.add_argument("--threshold", type=float, default=27.0,
                   help="ContentDetector threshold (default: 27.0)")
    p.add_argument("--adaptive-threshold", type=float, default=5.0,
                   help="AdaptiveDetector threshold (default: 5.0)")
    p.add_argument("--luma-only", action="store_true", default=True,
                   help="Use luminance only (default: on)")
    p.add_argument("--no-luma-only", dest="luma_only", action="store_false",
                   help="Use full hue/sat/lum scoring")
    p.add_argument("--min-segment", type=float, default=2.0)

    # ── sweep (threshold tuning) ──
    p = sub.add_parser("sweep",
                       help="Test multiple detector configs, print comparison table")
    p.add_argument("src", type=Path)

    # ── shader-lab ──
    p = sub.add_parser("shader-lab",
                       help="Random shader stacks on random clips — exploratory")
    p.add_argument("src", type=Path)
    p.add_argument("-n", "--count", type=int, default=4)
    p.add_argument("--duration", type=float, default=30.0,
                   help="Clip duration in seconds (default: 30)")
    p.add_argument("--pack", action="append", dest="packs",
                   help="Restrict to specific shader packs (repeatable)")
    p.add_argument("--min-shaders", type=int, default=1)
    p.add_argument("--max-shaders", type=int, default=3)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("-o", "--output-dir", type=Path)

    # ── crush-lab ──
    p = sub.add_parser("crush-lab",
                       help="Bitrate-crush clips then apply random shader stacks")
    p.add_argument("src", type=Path)
    p.add_argument("-n", "--count", type=int, default=4)
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--pack", action="append", dest="packs",
                   help="Restrict to specific shader packs (repeatable)")
    p.add_argument("--min-shaders", type=int, default=1)
    p.add_argument("--max-shaders", type=int, default=6)
    p.add_argument("--crush", type=float, default=0.7,
                   help="Crush intensity 0.0–1.0 (default: 0.7)")
    p.add_argument("--crush-codec", default="libx264",
                   choices=["libx264", "mpeg2video", "mpeg4"])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("-o", "--output-dir", type=Path)

    # ── warp-chain ──
    p = sub.add_parser("warp-chain",
                       help="Apply a chain of warp shaders to source footage")
    p.add_argument("src", type=Path)
    p.add_argument("--pack", action="append", dest="packs",
                   help="Restrict to specific shader packs (repeatable)")
    p.add_argument("--shaders", type=Path, nargs="+", default=None,
                   dest="shader_paths",
                   help="Explicit shader list (skips random selection)")
    p.add_argument("--categories", nargs="+", default=["Warp", "Brain Wipe"])
    p.add_argument("-n", "--n-shaders", type=int, default=2)
    p.add_argument("--all-shaders", dest="warpers_only",
                   action="store_false", default=True)
    p.add_argument("--no-normalize", dest="normalize",
                   action="store_false", default=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("-o", "--output", type=Path, default=None)

    # ── brain-wipe-render ──
    p = sub.add_parser("brain-wipe-render",
                       help="Pre-render a long-form brain wipe sequence")
    p.add_argument("src", type=Path)
    p.add_argument("--pack", action="append", dest="packs",
                   help="Restrict to specific shader packs (repeatable)")
    p.add_argument("--categories", nargs="+", default=["Warp", "Brain Wipe"])
    p.add_argument("-n", "--n-segments", type=int, default=8)
    p.add_argument("--segment-dur", type=float, default=20.0)
    p.add_argument("--n-warp-shaders", type=int, default=2)
    p.add_argument("--use-generators", action="store_true", default=False)
    p.add_argument("--no-shuffle", dest="shuffle",
                   action="store_false", default=True)
    p.add_argument("--no-normalize", dest="normalize",
                   action="store_false", default=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("-o", "--output", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)

    # ── stooges-channels ──
    p = sub.add_parser("stooges-channels",
                       help="Build multi-channel CRT TV content with static bursts")
    p.add_argument("src", type=Path,
                   help="Source footage (e.g. tooth_will_out.mp4)")
    p.add_argument("--pack", action="append", dest="packs",
                   help="Restrict to specific shader packs (repeatable)")
    p.add_argument("-n", "--n-channels", type=int, default=4,
                   help="Number of channel files to produce (default: 4)")
    p.add_argument("--static-duration", type=float, default=0.3,
                   help="Static burst length in seconds (default: 0.3)")
    p.add_argument("--no-shuffle", dest="shuffle", action="store_false",
                   default=True,
                   help="Disable per-channel segment shuffling")
    p.add_argument("--min-shaders", type=int, default=1)
    p.add_argument("--max-shaders", type=int, default=3)
    p.add_argument("--threshold", type=float, default=27.0,
                   help="Scene cut detection threshold (default: 27.0)")
    p.add_argument("--min-segment", type=float, default=1.5,
                   help="Minimum segment duration in seconds (default: 1.5)")
    p.add_argument("--seed", type=int, default=None,
                   help="Master random seed for reproducibility")
    p.add_argument("-o", "--output-dir", type=Path, default=None)

    args = parser.parse_args()
    cfg = Config(packs=getattr(args, "packs", None))
    cfg.ensure_dirs()

    if args.flow == "cut-shuffle-shader":
        out = cut_shuffle_shader(
            args.src, args.shader, output=args.output,
            scene_threshold=args.threshold, cfg=cfg,
        )
    elif args.flow == "random-shader-collage":
        out = random_shader_collage(
            args.src,
            count=args.count, min_dur=args.min_dur, max_dur=args.max_dur,
            output=args.output, cfg=cfg,
        )
    elif args.flow == "density-composite":
        out = density_composite(
            args.sources, output=args.output,
            mode=args.mode, opacity=args.opacity, cfg=cfg,
        )
    elif args.flow == "masked-shader-overlay":
        out = masked_shader_overlay(
            args.src, args.shader, mask_type=args.mask_type,
            output=args.output, cfg=cfg,
        )
    elif args.flow == "texture-builder":
        out = texture_builder(
            args.src, args.shaders, output=args.output, cfg=cfg,
        )
    elif args.flow == "shuffled-scene-shaders":
        out = shuffled_scene_shaders(
            args.src,
            output=args.output, scene_threshold=args.threshold,
            min_shaders=args.min_shaders, max_shaders=args.max_shaders,
            seed=args.seed, cfg=cfg,
        )
    elif args.flow == "scene-split":
        segs = scene_split(
            args.src, output_dir=args.output_dir,
            scene_threshold=args.threshold,
            adaptive_threshold=args.adaptive_threshold,
            method=args.method,
            luma_only=args.luma_only,
            min_segment=args.min_segment, cfg=cfg,
        )
        print(f"\n{len(segs)} scene files written.")
        return
    elif args.flow == "sweep":
        sweep_thresholds(args.src, cfg=cfg)
        return
    elif args.flow == "shader-lab":
        samples = shader_lab(
            args.src, count=args.count, duration=args.duration,
            min_shaders=args.min_shaders, max_shaders=args.max_shaders,
            seed=args.seed, output_dir=args.output_dir, cfg=cfg,
        )
        print(f"\n{len(samples)} samples written.")
        return
    elif args.flow == "crush-lab":
        samples = crush_lab(
            args.src, count=args.count, duration=args.duration,
            min_shaders=args.min_shaders, max_shaders=args.max_shaders,
            crush=args.crush, crush_codec=args.crush_codec,
            seed=args.seed, output_dir=args.output_dir, cfg=cfg,
        )
        print(f"\n{len(samples)} samples written.")
        return
    elif args.flow == "deep-color":
        out = deep_color(
            args.src,
            output=args.output, scene_threshold=args.threshold,
            shaders_per_pass=args.shaders_per_pass,
            crush=args.crush,
            crush_codec=args.crush_codec,
            density_mode=args.density_mode,
            density_opacity=args.density_opacity,
            edge_strength=args.edge_strength,
            seed=args.seed, cfg=cfg,
        )
    elif args.flow == "warp-chain":
        out = warp_chain(
            src=args.src,
            shader_paths=args.shader_paths,
            shader_categories=args.categories,
            n_shaders=args.n_shaders,
            warpers_only=args.warpers_only,
            output=args.output,
            normalize=args.normalize,
            seed=args.seed,
            cfg=cfg,
        )
    elif args.flow == "brain-wipe-render":
        out = brain_wipe_render(
            src=args.src,
            shader_categories=args.categories,
            n_segments=args.n_segments,
            segment_dur=args.segment_dur,
            n_warp_shaders=args.n_warp_shaders,
            use_generators=args.use_generators,
            shuffle=args.shuffle,
            normalize=args.normalize,
            seed=args.seed,
            output=args.output,
            output_dir=args.output_dir,
            cfg=cfg,
        )
    elif args.flow == "stooges-channels":
        channels = stooges_channels(
            src=args.src,
            n_channels=args.n_channels,
            static_duration=args.static_duration,
            shuffle=args.shuffle,
            min_shaders=args.min_shaders,
            max_shaders=args.max_shaders,
            scene_threshold=args.threshold,
            min_segment=args.min_segment,
            seed=args.seed,
            output_dir=args.output_dir,
            cfg=cfg,
        )
        for ch in channels:
            print(ch)
        return

    print(f"Output: {out}")


if __name__ == "__main__":
    _cli()
