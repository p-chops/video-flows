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
├── cache.py               # Prefect cache policy — FileValidatedInputs (re-run if dst missing)
├── tasks/                 # Prefect @task functions (the building blocks)
│   ├── cut.py             # Scene detection, segment extraction
│   ├── sequence.py        # Concat, shuffle, interleave, generate static/solid
│   ├── shader.py          # Apply ISF shaders to video via moderngl
│   ├── composite.py       # Blend, masked composite, multi-layer, PIP, chromakey
│   ├── mask.py            # Luma, edge (Canny), motion, chroma, gradient masks
│   ├── color.py           # normalize_levels, auto_levels — level stretch + gamma correction
│   ├── glitch.py          # bitrate_crush — codec-based compression artifacts
│   ├── time.py            # Temporal effects: scrub, drift, ping-pong, echo, patch
│   └── transition.py      # Transitions: crossfade, luma_wipe, whip_pan, static_burst, flash
└── flows/                 # Prefect @flow compositions (example pipelines)
    ├── examples.py        # 10 flows: cut-shuffle-shader, deep-color, shader-lab, crush-lab, etc.
    ├── stooges.py         # Multi-channel CRT TV content (crush sandwich + parallel)
    ├── brain_wipe.py      # Meta-flow + warp chains + long-form brain wipe renders
    ├── compositing_lab.py # Random layer compositing experiments
    ├── time_lab.py        # Time effect lab flows + CLI
    ├── transition_lab.py  # Transition demos + CLI
    └── show_reel.py       # Channel-surfing generator show reel with random transitions
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

## Compositing

`composite.py` provides five tasks:

- **`blend_layers`** — blend two videos via ffmpeg's blend filter (modes: normal, add, multiply, screen, overlay, difference, softlight)
- **`masked_composite`** — composite overlay onto base using a grayscale mask video (ffmpeg maskedmerge)
- **`multi_layer_composite`** — chain multiple layers bottom-to-top via ffmpeg blend filters
- **`picture_in_picture`** — scale and position overlay at (x, y) on base (ffmpeg overlay filter)
- **`chromakey_composite`** — HSV-based colour removal (frame-by-frame Python/OpenCV, the only non-ffmpeg composite task)

Blend modes use ffmpeg filter names mapped via `FFMPEG_BLEND_MODES` dict. Adding a new mode: add the mapping to the dict (ffmpeg blend mode names may differ from common names, e.g. `"add"` → `"addition"`).

## Adding New Capabilities

**New task**: Add a function in `pipeline/tasks/`, decorate with `@task(name="kebab-case-name")`. Follow the existing signature pattern: `(src: Path, dst: Path, ..., cfg: Optional[Config] = None) -> Path`. Export from `tasks/__init__.py`.

**New flow**: Add a function in `pipeline/flows/`, decorate with `@flow(name="kebab-case-name")`. Compose existing tasks. Call `cfg.ensure_dirs()` early. Export from `flows/__init__.py`.

**New blend mode**: Add the ffmpeg blend mode name to `FFMPEG_BLEND_MODES` dict in `composite.py`.

**New mask type**: Add a `@task` to `mask.py`. Output 3-channel grayscale video (white=include, black=exclude).

**New ISF built-in**: Add the replacement to `_translate_isf_to_glsl()` in `isf.py`.

**New recipe step type** (from labs): Add a dataclass to `recipe.py`, add it to the `Step` union type alias, add a `case` branch in `_submit_step()` in `brain_wipe.py`. The step dataclass should have fields matching the corresponding task's parameters.

## Color Tasks

`pipeline/tasks/color.py` provides two tasks:

- **`normalize_levels`** — percentile-based level stretch via ffmpeg's `colorlevels` filter. Clips the darkest/brightest percentiles and stretches remaining range to 0–255. Static per-pixel LUT with negligible overhead.
- **`auto_levels`** — probe average brightness via frame sampling, compute gamma correction toward a target (default 0.45), apply via ffmpeg `eq=gamma=X`. Skips encode if no correction needed. Useful for individual clips but not ideal for show reels (tends to flatten character).

## Transition Tasks

`pipeline/tasks/transition.py` provides transitions between video segments.

**Transition types**: `crossfade`, `luma_wipe`, `whip_pan`, `static_burst`, `flash`, `random`

**Key functions**:
- **`transition_sequence`** — chain N clips with transitions. Public API used by recipes and flows.
- **`_streaming_chain`** — O(n) streaming implementation for frame-level transitions (luma_wipe, whip_pan, static_burst, flash, random). Uses deque-based tail buffering — each clip read exactly once.
- **`_xfade_chain`** — ffmpeg filter graph for crossfade (single ffmpeg command).

**Random transitions** (`transition_type="random"`): picks a different transition type with random parameters for each pair boundary. Types drawn from: crossfade, luma_wipe, whip_pan, static_burst, flash. Deterministic per-pair seeds derived from base seed.

**Wipe patterns** (for luma_wipe): horizontal, vertical, radial, diagonal, noise, star, directional.

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
- `temporal_sandwich_recipe(src)` — scrub → shaders → echo → shaders → patch (time as the crusher)
- `deep_time_recipe(src)` — drift → pingpong → echo → scrub → shader (recursive temporal folding)
- `hybrid_composite_recipe(src)` — footage + generator lanes, motion mask composite
- `codec_spectrum_recipe(src)` — mpeg2 → mpeg4 → x264 multi-codec crush cascade
- `breathing_wall_recipe(src)` — 3 lanes at different ping-pong rates, screen-blended polyrhythm
- `erosion_recipe(src)` — progressive downscale crush (2x → 4x → 8x)
- `palimpsest_recipe(src)` — same footage, two recipes, edge mask composite (overwritten memory)
- `generator_stooges_recipe()` — stooges but all-generator, no source (alien TV station)
- `gradient_dissolve_recipe(src)` — footage + generator via gradient mask (portal effect)
- `accretion_recipe(src)` — 4 lanes at escalating destruction, screen-blended (geological layering)

**Procedural recipe generator** (`random_recipe()`): picks a structural **archetype**, then fills in with complexity-scaled parameters.

8 archetypes:
| Archetype | Structure |
|-----------|-----------|
| `crush_sandwich` | Alternating crush/shader pairs (C-S-C-S), optional codec cascade |
| `deep_time` | 3–5 stacked time effects + 1 shader (temporal destruction) |
| `temporal_sandwich` | Alternating time/shader pairs (T-S-T-S) |
| `escalation` | Progressive parameter increase (within-lane crush/downscale, or cross-lane intensity) |
| `polyrhythm` | 2–4 lanes at harmonically-related temporal rates, brightness-neutral blend |
| `palimpsest` | 2 lanes, same source, contrasting treatments (crush vs time), masked composite |
| `hybrid` | Footage + generator lane, masked composite |
| `grab_bag` | Original behavior — independent step draws from pool |

Auto-selected from eligible set based on context (src, n_lanes, use_generators). Force via `archetype="deep_time"` etc. Complexity still scales all parameters within the chosen archetype.

**Blend modes**: brightness-neutral only (overlay, softlight, difference, multiply, normal). Screen and add removed — they compound generator brightness problems.

- **Performance impact**: complexity 0.2 renders at ~2–3x realtime; complexity 0.9 at ~11x realtime (for 3 min video)
- **Overrides**: `n_lanes`, `n_steps`, `n_segments`, `use_transitions`, `use_generators`, `seed`, `target_dur`, `archetype`

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
python -m pipeline.flows.brain_wipe brain-wipe --preset temporal-sandwich source.mp4 --seed 42
python -m pipeline.flows.brain_wipe brain-wipe --preset breathing-wall source.mp4 --seed 42
python -m pipeline.flows.brain_wipe brain-wipe --preset generator-stooges --segment-counts 6,8,10,8,6 --seed 42
python -m pipeline.flows.brain_wipe brain-wipe --preset accretion source.mp4 --seed 42
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

## Show Reel Flow

`pipeline/flows/show_reel.py` — channel-surfing through heterogeneous generator "shows". Each show is a short (5–15s) generator clip rendered at a random complexity level via `random_recipe` + `brain_wipe` subflow, so some are raw warped generators and others have crush, shaders, time effects, etc. Shows are joined with random per-pair transitions.

```
PREFECT_API_URL=http://127.0.0.1:4200/api \
python -m pipeline.flows.show_reel -n 15 --min-dur 10 --max-dur 15 --seed 777
```

Key parameters: `n_shows` (number of segments), `min_dur`/`max_dur` (duration range), `min_complexity`/`max_complexity` (complexity range per show), `transition_dur`, `width`/`height`, `seed`.

Note: connect to persistent Prefect server via `PREFECT_API_URL=http://127.0.0.1:4200/api` for UI visibility. Without it, flows start ephemeral servers.

## Shader Library

Shaders in `shaders/` directory, organized by function. Shaders use ISF v2 format with `CATEGORIES` tags in the JSON header for filtering.

### Glitch/color shaders (21 — in `shaders/`)

**Spatial/distortion**: warp, echo, pixel_sort, databend, scan_tear, bit_crush, feedback_zoom, block_shift, stereo_project, posterize, lens_warp, shear
**Color/tone**: duotone, chromatic_shift, gradient_map, edge_glow, cyanotype, heat_sig, oxide, neon_bleed, bruise

Deleted shaders (for reference): chromawave, false_color, plasma_tint, palette_cycle, interference, solarize_color, mirror_fracture, droste, swirl, tunnel — removed for aesthetic or functional reasons.

### Brain wipe / warp shaders (in `brain-wipe-shaders/`)

Separate shader directory from the glitch/color library. Use `--shader-dir brain-wipe-shaders` with the brain wipe flows, or pass the path as `shader_dir` in Python.

**4 video warpers (have `inputImage`):**
- `ulp-warp-fbm` — fractal domain warp (iterative FBM noise, 1–3 passes)
- `ulp-warp-curl` — curl noise flow (divergence-free, no tearing)
- `ulp-warp-gravitational` — multi-point gravitational lensing (up to 5 orbiting masses)
- `ulp-warp-voronoi` — Voronoi cell refraction (convex/concave/edge-push modes)

**7 generators (no `inputImage`):**
- `ulp-brain-wipe-plasma` — sinusoidal plasma field
- `ulp-brain-wipe-tunnel` — infinite geometric tunnel
- `ulp-brain-wipe-chladni` — vibrating plate resonance figures
- `ulp-brain-wipe-rd` — reaction-diffusion simulator (brightest generator)
- `ulp-abyssal-jellies-v4` — bioluminescent jellyfish swarm
- `ulp-bioluminescent-field` — deep-sea bioluminescence
- `abyssal_drift` — deep-sea ambient drift

**Planned but not yet written:**
- `ulp-brain-wipe-tunnel-video` — maps video onto tunnel surface (warper)
- `ulp-brain-wipe-chladni-video` — Chladni gradient displacement (warper)

All are tagged `"Warp"`, `"Brain Wipe"`, or `"Generator"` in ISF `CATEGORIES` and compatible with Magic Music Visuals for live use.

### Shader categories and filtering

Shaders declare categories in their ISF `CATEGORIES` header array. The `filter_shaders()` utility in `brain_wipe.py` filters by category and/or by whether the shader has an `inputImage` input (warper vs generator). The `ISFShader` dataclass exposes `.categories` (list[str]) and `.image_inputs` (list of inputs with type `"image"`).

## Known Issues / TODO

- **Generator dynamic range** — dark generators (jellies, bioluminescent-field, abyssal_drift: avg Y 17–38) and bright generators (RD: avg Y 193–232) both produce flat output with poor dynamic range. Post-processing (normalize, auto_levels) doesn't work well — it flattens character. Needs shader-level fixes: lift dark floors, clamp bright peaks, reduce solid white blocks.
- No audio passthrough — all tasks strip audio (`-an`). This is intentional for now (visual processing only) but could be added via ffmpeg's `-c:a copy` flag.
- Work directory cleanup is not automatic. Intermediate files accumulate in `work/`.

## Style

Python 3.10+. Type hints throughout. `from __future__ import annotations` in every module. `Path` objects for all file references (never raw strings). Prefect 3.x API only (no Prefect 2 patterns).
