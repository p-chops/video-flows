# ULP Video Pipeline

Automated video processing pipeline for the Undersea Lair Project. Cuts, sequences, applies ISF shader stacks to, and composites video — all orchestrated with Prefect.

## Project Context

This is a creative/art tool, not a production web service. The owner (P-Chops) makes abstract, drone, glitchy live visuals for Twitch livestreams. The pipeline processes source footage into dense, textural, otherworldly video by combining cuts, shader effects, masks, and compositing. Think structural film, not YouTube edits.

Source footage is intentionally left ungraded — no archival color correction — so it can be arbitrarily processed throughout the project.

## Architecture

```
pipeline/
├── config.py              # Config dataclass — paths, ffmpeg settings, video defaults
├── recipe.py              # Recipe data model — BrainWipeRecipe, sources, steps, lanes
├── ffmpeg.py              # All video I/O: probe, read_frames, FrameWriter, extract, concat
├── gl.py                  # moderngl standalone context, FBO ping-pong, quad geometry
├── isf.py                 # ISF v2 parser + GLSL translator
├── tasks/                 # Prefect @task functions (the building blocks)
│   ├── cut.py             # Scene detection, segment extraction
│   ├── sequence.py        # Concat, shuffle, interleave, generate static/solid
│   ├── shader.py          # Apply ISF shaders to video via moderngl
│   ├── composite.py       # Blend modes, masked composite, multi-layer, chromakey
│   ├── mask.py            # Luma, edge (Canny), motion, chroma, gradient masks
│   ├── color.py           # normalize_levels — percentile-based level stretch
│   └── glitch.py          # bitrate_crush — codec-based compression artifacts
└── flows/                 # Prefect @flow compositions (example pipelines)
    ├── examples.py        # 10 flows: cut-shuffle-shader, deep-color, shader-lab, crush-lab, etc.
    ├── stooges.py         # Multi-channel CRT TV content (crush sandwich + parallel)
    ├── brain_wipe.py      # Meta-flow + warp chains + long-form brain wipe renders
    └── compositing_lab.py # Random layer compositing experiments
```

Tasks are atomic Prefect `@task` functions. Flows are `@flow` functions that compose tasks into pipelines. New workflows should follow the same pattern: write tasks for new primitives, compose them in flows.

## Key Patterns

**Frame pipeline**: ffmpeg decodes → raw RGB24 numpy arrays → processing → ffmpeg encodes. All frame I/O goes through `ffmpeg.py`. Frames are `(H, W, 3)` uint8 numpy arrays.

**Shader rendering**: moderngl standalone context (headless, no window). Full-screen quad + ping-pong FBOs for shader chaining. Frames are uploaded as textures, rendered through shaders, read back. The Y axis is flipped (`np.flipud`) on upload and readback because OpenGL's origin is bottom-left.

**ISF translation**: ISF v2 `.fs` files (JSON header + GLSL body) are parsed and translated to standard `#version 330` GLSL. The translator handles:
- `isf_FragNormCoord` → `v_texcoord`
- `IMG_NORM_PIXEL(tex, uv)` → `texture(tex, uv)`
- `IMG_PIXEL(tex, px)` → `texelFetch(tex, ivec2(px), 0)`
- `RENDERSIZE` → `u_rendersize`
- `TIME` → `u_time`
- `PASSINDEX` → `u_passindex`
- `FRAMEINDEX` → `u_frameindex`
- `gl_FragColor` → `fragColor`

Uniforms for ISF `INPUTS` are declared based on type (float, bool/event as float, long as int, point2D as vec2, color as vec4, image as sampler2D). Persistent buffer targets also get sampler2D uniforms.

**Config propagation**: Every task/flow takes an optional `Config` object. `Config()` with no args uses sensible defaults rooted at `.` (overridable via `VP_PROJECT_ROOT` env var). Always pass `cfg` through the chain.

**Masks are 3-channel**: Even though masks are conceptually grayscale, they're stored as 3-channel RGB (gray duplicated across channels) for pipeline compatibility with FrameWriter. The red channel is used as alpha during compositing.

## Dependencies

- `prefect>=3.0` — orchestration, `@task`/`@flow` decorators
- `moderngl` — headless OpenGL for GLSL shader rendering
- `numpy` — frame buffers, array operations
- `opencv-python-headless` — masks (Canny, thresholding, morphology, color space conversion)
- `ffmpeg` / `ffprobe` — external binaries, must be on PATH

## ISF Shader Conventions

Shaders live in `.fs` files in the shader directory. They use ISF v2 format: a `/*{ JSON }*/` header followed by GLSL body. The pipeline loads them via `parse_isf()` or `load_shader_dir()`.

Known ISF host quirks (from Magic Music Visuals testing):
- `RENDERSIZE` can be unreliable — some shaders use a `sim_scale` parameter instead
- Persistent buffer temporal feedback doesn't work in all hosts — stateless mathematical approaches (recursive zoom loops, etc.) are more portable
- When seeding reaction-diffusion or other sim shaders, full-field random initialization works better than sparse single-pixel seeds (isolated pixels lack critical mass and diffuse away)

## Blend Modes

Available in `composite.py`: `normal`, `add`, `multiply`, `screen`, `overlay`, `difference`, `softlight`. All operate on float32 [0,1] arrays. Adding new modes: write a function `_blend_foo(a, b, opacity) -> np.ndarray` and add it to the `BLEND_MODES` dict.

## Adding New Capabilities

**New task**: Add a function in `pipeline/tasks/`, decorate with `@task(name="kebab-case-name")`. Follow the existing signature pattern: `(src: Path, dst: Path, ..., cfg: Optional[Config] = None) -> Path`. Export from `tasks/__init__.py`.

**New flow**: Add a function in `pipeline/flows/`, decorate with `@flow(name="kebab-case-name")`. Compose existing tasks. Call `cfg.ensure_dirs()` early. Export from `flows/__init__.py`.

**New blend mode**: Add a `_blend_*` function to `composite.py` and register it in `BLEND_MODES`.

**New mask type**: Add a `@task` to `mask.py`. Output 3-channel grayscale video (white=include, black=exclude).

**New ISF built-in**: Add the replacement to `_translate_isf_to_glsl()` in `isf.py`.

**New recipe step type** (from labs): Add a dataclass to `recipe.py`, add it to the `Step` union type alias, add a `case` branch in `_submit_step()` in `brain_wipe.py`. The step dataclass should have fields matching the corresponding task's parameters.

## Glitch Tasks

`pipeline/tasks/glitch.py` contains codec-abuse effects that exploit encoder behavior rather than processing frames directly.

**bitrate_crush**: Encodes at aggressively low quality (fixed QP) then re-encodes clean to bake artifacts in. Key design decisions:
- Uses **fixed QP mode** (not CRF, not bitrate). CRF varies QP per frame; bitrate mode causes VBV buffer oscillation. Fixed QP = truly uniform quality per frame.
- **`-sc_threshold 0` + `-keyint_min` = total frames**: Disables x264 auto scene detection which inserts keyframes that look different (brighter) than P-frames, causing "flash and decay."
- GOP = total frame count so only one keyframe exists (the first frame). All subsequent frames are P-frames.
- `crush` parameter: 0.0–1.0 maps to QP 30–51 (libx264) or q:v 10–31 (mpeg2/mpeg4).
- `downscale` parameter: shrinks before crushing, nearest-neighbor upscale baked into the dirty pass. Produces bigger blocks.

## Recipe System

`pipeline/recipe.py` — declarative data model for describing full render pipelines.

A `BrainWipeRecipe` describes: **lanes** (parallel processing streams), **compositing** (how lanes combine), and **post-processing** (final steps). Each lane has a **source** (footage, generator, static, solid), a **recipe** (ordered list of processing steps), and **sequencing** (shuffle, concat, optional static interleaving).

**Step types**: `CrushStep`, `ShaderStep`, `NormalizeStep`, `ScrubStep`, `DriftStep`, `PingPongStep`, `EchoStep`, `PatchStep`. Adding new step types from labs: add a dataclass to `recipe.py`, add to the `Step` union, add a `case` branch in `_submit_step()` in `brain_wipe.py`.

**Source types**: `FootageSource` (random or scene-based segmentation), `GeneratorSource` (generator shaders + optional warps), `StaticSource`, `SolidSource`.

**Composite types**: `BlendComposite`, `MaskedComposite`, `RandomComposite` (not yet implemented — use `compositing_lab` directly).

**Recipe builders** construct common patterns:
- `crush_sandwich_recipe(src)` — crush → shaders → crush → shaders → normalize
- `stooges_recipe(src, segment_counts=[8,10,12])` — multi-channel CRT content
- `generator_render_recipe()` — generator shaders + warp chains, no source needed
- `composite_recipe(src)` — two lanes composited via mask

**Recipe utilities**: `print_recipe()` pretty-prints, `hash_recipe()` returns 8-char hex hash for output naming.

## Brain Wipe Flows

`pipeline/flows/brain_wipe.py` — three flows:

**`brain_wipe`** (meta-flow): Takes a `BrainWipeRecipe` and executes it. Materialises sources, processes segments through recipe steps concurrently (Prefect future chaining), sequences per lane, composites lanes, applies post-processing. Can express all other complex flows as recipes.

```python
from pipeline.recipe import crush_sandwich_recipe
from pipeline.flows.brain_wipe import brain_wipe
brain_wipe(crush_sandwich_recipe(Path("source/footage.mp4"), seed=42))
```

```
python -m pipeline.flows.brain_wipe brain-wipe --preset crush-sandwich source.mp4 --seed 42
python -m pipeline.flows.brain_wipe brain-wipe --preset stooges source.mp4 --segment-counts 8,10,12
python -m pipeline.flows.brain_wipe brain-wipe --preset generator-render -n 12 --seed 42
```

**`warp_chain`**: Apply a chain of warp shaders to source footage. Single input → single output. Filters to `["Warp", "Brain Wipe"]` categories by default.

```
python -m pipeline.flows.brain_wipe warp-chain source/footage.mp4 --n-shaders 2 --seed 7
```

**`brain_wipe_render`**: Pre-render a long-form brain wipe sequence via generator shaders. No source footage needed.

```
python -m pipeline.flows.brain_wipe brain-wipe-render -n 12 --segment-dur 20 --seed 42
```

All flows use `ConcurrentTaskRunner(max_workers=4)` for parallelism.

## Stooges Flow

`pipeline/flows/stooges.py` — multi-channel CRT TV content for OBS composite. Each channel draws random segments from source, runs a crush sandwich (crush → 3 shaders → crush → 3 shaders → normalize) per segment, shuffles, interleaves with TV static, and concatenates. Uses `ConcurrentTaskRunner(max_workers=4)` — all segment pipelines across all channels run concurrently with future-based dependency chaining. Confirmed thread-safe for moderngl shader tasks.

```
python -m pipeline.flows.stooges input/footage.mp4 \
    --n-channels 5 --segment-counts 8,10,12,14,16 --seed 42
```

## Shader Library

Shaders in `shaders/` directory, organized by function. Shaders use ISF v2 format with `CATEGORIES` tags in the JSON header for filtering.

### Glitch/color shaders (21 — in `shaders/`)

**Spatial/distortion**: warp, echo, pixel_sort, databend, scan_tear, bit_crush, feedback_zoom, block_shift, stereo_project, posterize, lens_warp, shear
**Color/tone**: duotone, chromatic_shift, gradient_map, edge_glow, cyanotype, heat_sig, oxide, neon_bleed, bruise

Deleted shaders (for reference): chromawave, false_color, plasma_tint, palette_cycle, interference, solarize_color, mirror_fracture, droste, swirl, tunnel — removed for aesthetic or functional reasons.

### Brain wipe / warp shaders (in `brain-wipe-shaders/`)

Separate shader directory from the glitch/color library. Use `--shader-dir brain-wipe-shaders` with the brain wipe flows, or pass the path as `shader_dir` in Python.

**Currently present (4 video warpers — have `inputImage`):**
- `ulp-warp-fbm` — fractal domain warp (iterative FBM noise, 1–3 passes)
- `ulp-warp-curl` — curl noise flow (divergence-free, no tearing)
- `ulp-warp-gravitational` — multi-point gravitational lensing (up to 5 orbiting masses)
- `ulp-warp-voronoi` — Voronoi cell refraction (convex/concave/edge-push modes)

**Planned but not yet written:**
- `ulp-brain-wipe-plasma` — sinusoidal plasma field (generator)
- `ulp-brain-wipe-tunnel` — infinite geometric tunnel (generator)
- `ulp-brain-wipe-chladni` — vibrating plate resonance figures (generator)
- `ulp-brain-wipe-tunnel-video` — maps video onto tunnel surface (warper)
- `ulp-brain-wipe-chladni-video` — Chladni gradient displacement (warper)

All are tagged `"Warp"` or `"Brain Wipe"` in ISF `CATEGORIES` and compatible with Magic Music Visuals for live use.

### Shader categories and filtering

Shaders declare categories in their ISF `CATEGORIES` header array. The `filter_shaders()` utility in `brain_wipe.py` filters by category and/or by whether the shader has an `inputImage` input (warper vs generator). The `ISFShader` dataclass exposes `.categories` (list[str]) and `.image_inputs` (list of inputs with type `"image"`).

## Known Issues / TODO

- `FrameWriter` constructor signature inconsistency: `sequence.py` passes `(dst, width, height, fps=fps, cfg=c)` while `ffmpeg.py` defines `__init__(self, path, info, cfg)` expecting a `VideoInfo`. One of these needs to be reconciled — the tasks assume width/height/fps kwargs but the class takes a VideoInfo object.
- `GLContext.texture()` requires `data` arg in its signature but `shader.py` calls `gl.texture(w, h)` without data. Needs a default `data=None` or the call needs updating.
- `GLContext` constructor doesn't accept `(w, h)` but `shader.py` calls `GLContext(w, h)`. Needs reconciling.
- Multi-pass ISF shaders (persistent buffers) are parsed but not fully rendered — `shader.py` currently does single-pass-per-shader chaining only. Adding multi-pass support would require per-shader FBO management keyed to pass targets.
- No audio passthrough — all tasks strip audio (`-an`). This is intentional for now (visual processing only) but could be added via ffmpeg's `-c:a copy` flag.
- Work directory cleanup is not automatic. Intermediate files accumulate in `work/`.

## Style

Python 3.10+. Type hints throughout. `from __future__ import annotations` in every module. `Path` objects for all file references (never raw strings). Prefect 3.x API only (no Prefect 2 patterns).
