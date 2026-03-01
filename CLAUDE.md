# ULP Video Pipeline

Automated video processing pipeline for the Undersea Lair Project. Cuts, sequences, applies ISF shader stacks to, and composites video — all orchestrated with Prefect.

## Project Context

This is a creative/art tool, not a production web service. The owner (P-Chops) makes abstract, drone, glitchy live visuals for Twitch livestreams. The pipeline processes source footage into dense, textural, otherworldly video by combining cuts, shader effects, masks, and compositing. Think structural film, not YouTube edits.

Source footage is intentionally left ungraded — no archival color correction — so it can be arbitrarily processed throughout the project.

## Architecture

```
boutique_stacks.yaml       # Hand-curated shader stacks with per-shader param randomization
scripts/                   # Standalone render scripts (heightfield_explorer, render_terrain_scan, render_boutique_reel)
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
│   ├── time.py            # Temporal effects: scrub, drift, ping-pong, echo, patch, slit_scan, temporal_tile, smear, bloom, frame_stack, slip, flow_warp, temporal_sort, extrema_hold, feedback_transform
│   ├── transform.py       # Spatial/color transforms: mirror, zoom, invert, hue_shift, saturate
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

All multi-input compositing tasks use both `shortest=1` in the filter graph AND `-shortest` as an output flag. Both are required — the filter-level flag handles mismatched inputs, the output flag tells the muxer to stop writing when the shortest stream ends. Without both, ffmpeg can hold the last frame of the shorter input.

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

## Time Effects

`pipeline/tasks/time.py` — temporal manipulation effects. Flows + CLI in `pipeline/flows/time_lab.py`. All effects: same duration in, same duration out. Each effect has a `_process_*` helper (pure function on `list[np.ndarray]`) and a thin `@task` wrapper for standalone use.

| Effect | Task | Key params | Memory model |
|--------|------|-----------|--------------|
| scrub | `time_scrub` | `smoothness`, `intensity`, `seed` | All frames in RAM |
| drift | `drift_loop` | `loop_dur`, `drift`, `seed` | All frames in RAM |
| pingpong | `ping_pong` | `window`, `seed` | All frames in RAM |
| echo | `echo_trail` | `delay`, `trail` | Ring buffer (delay_frames) |
| patch | `time_patch` | `patch_min`, `patch_max`, `seed` | Streaming (1 canvas) |
| slit_scan | `slit_scan` | `axis`, `scan_speed`, `seed` | All frames in RAM |
| temporal_tile | `temporal_tile` | `grid`, `offset_scale`, `seed` | All frames in RAM |
| smear | `smear` | `threshold` | Streaming (1 canvas) |
| bloom | `bloom` | `sensitivity` | Streaming |
| frame_stack | `frame_stack` | `window`, `mode` | Ring buffer |
| slip | `slip` | `n_bands`, `max_slip`, `axis`, `seed` | All frames in RAM |
| flow_warp | `flow_warp` | `amplify`, `smooth`, `seed` | Streaming (prev frame) |
| temporal_sort | `temporal_sort` | `mode`, `direction`, `seed` | All frames in RAM (3D volume) |
| extrema_hold | `extrema_hold` | `mode`, `decay`, `seed` | Streaming (float32 canvas) |
| feedback_transform | `feedback_transform` | `transform`, `amount`, `mix`, `seed` | Streaming (prev output) |

**Removed from random recipe pool** (still available for manual use): `extrema_hold` (drives to black/white), `smear` (stasis amplifier), `bloom` (stasis amplifier), `patch` (stasis amplifier).

**Fused time chain**: `fused_time_chain` task applies multiple time effects in a single decode→process→encode pass. See "Filter Chain Merging" section.

### Transform Tasks

Tasks in `pipeline/tasks/transform.py`. Pure ffmpeg filter-graph operations.

| Effect | Task | Key params |
|--------|------|-----------|
| mirror | `mirror` | `axis` (horizontal/vertical) |
| zoom | `zoom` | `factor`, `center_x`, `center_y` |
| invert | `invert` | (none) |
| hue_shift | `hue_shift` | `degrees` |
| saturate | `saturate` | `amount` |

## Recipe System

`pipeline/recipe.py` — declarative data model for describing full render pipelines.

A `BrainWipeRecipe` describes: **lanes** (parallel processing streams), **compositing** (how lanes combine), and **post-processing** (final steps). Each lane has a **source** (footage, generator, static, solid), a **recipe** (ordered list of processing steps), and **sequencing** (shuffle, concat, optional static interleaving).

**Step types**: `CrushStep`, `ShaderStep`, `NormalizeStep`, `ScrubStep`, `DriftStep`, `PingPongStep`, `EchoStep`, `PatchStep`, `SlitScanStep`, `TemporalTileStep`, `SmearStep`, `BloomStep`, `StackStep`, `SlipStep`, `FlowWarpStep`, `TemporalSortStep`, `ExtremaHoldStep`, `FeedbackTransformStep`, `MirrorStep`, `ZoomStep`, `InvertStep`, `HueShiftStep`, `SaturateStep`. Adding new step types from labs: add a dataclass to `recipe.py`, add to the `Step` union, add a `case` branch in `_submit_step()` in `brain_wipe.py`. All step types must also have `_step_to_dict`/`_step_from_dict` serialization for show reel manifest roundtripping.

**ShaderStep.param_overrides**: Optional `dict[str, dict[str, float]]` mapping shader stems to parameter dicts. Passed through to `apply_shader_stack` as custom uniform values. Used by boutique stacks to set per-shader parameters (e.g., `{"video_heightfield": {"MAXHEIGHT": 0.5, "SLICES": 128}}`).

**Source types**: `FootageSource` (random or scene-based segmentation), `GeneratorSource` (generator shaders + optional warps), `StaticSource`, `SolidSource`.

**Composite types**: `BlendComposite`, `MaskedComposite`, `RandomComposite` (not yet implemented — use `compositing_lab` directly).

**Recipe builders** construct common patterns:
- `crush_sandwich_recipe(src)` — crush → shaders → crush → shaders → normalize
- `stooges_recipe(src, segment_counts=[8,10,12])` — multi-channel CRT content
- `generator_render_recipe()` — generator shaders + warp chains, no source needed
- `composite_recipe(src)` — two lanes composited via mask
- `temporal_sandwich_recipe(src)` — scrub → shaders → echo → shaders → patch (time as the crusher)
- `deep_time_recipe(src)` — drift → pingpong → echo → scrub (recursive temporal folding)
- `time_cascade_recipe(src)` — multi-scale temporal cascade: echo → feedback → drift → pingpong → scrub → slit_scan
- `temporal_geology_recipe(src)` — alternating time-shader strata: drift+smear → shader → echo+tile → shader → extrema+slip
- `hybrid_composite_recipe(src)` — footage + generator lanes, motion mask composite
- `codec_spectrum_recipe(src)` — mpeg2 → mpeg4 → x264 multi-codec crush cascade
- `breathing_wall_recipe(src)` — 3 lanes at different ping-pong rates, screen-blended polyrhythm
- `erosion_recipe(src)` — progressive downscale crush (2x → 4x → 8x)
- `palimpsest_recipe(src)` — same footage, two recipes, edge mask composite (overwritten memory)
- `generator_stooges_recipe()` — stooges but all-generator, no source (alien TV station)
- `gradient_dissolve_recipe(src)` — footage + generator via gradient mask (portal effect)
- `accretion_recipe(src)` — 4 lanes at escalating destruction, screen-blended (geological layering)
- `stutter_recipe(src)` — rapid short segments, hard cuts, channel-surfing
- `echo_chamber_recipe(src)` — stacked echo effects at increasing delays
- `warp_focus_recipe()` — generator + heavy warps, minimal processing
- `edge_poster_recipe(src)` — posterize + edge_glow, footage or generators

**Procedural recipe generator** (`random_recipe()`): picks a structural **archetype**, then fills in with complexity-scaled parameters.

12 archetypes:
| Archetype | Structure |
|-----------|-----------|
| `crush_sandwich` | Alternating crush/shader pairs (C-S-C-S), optional codec cascade |
| `deep_time` | 5–8 stacked time effects, no shaders (pure temporal destruction) |
| `temporal_sandwich` | Alternating time/boutique shader pairs (T-S-T-S) |
| `escalation` | Progressive parameter increase (within-lane crush/downscale, or cross-lane intensity) |
| `polyrhythm` | 2–4 lanes at harmonically-related temporal rates, brightness-neutral blend |
| `palimpsest` | 2 lanes, same source, contrasting treatments (crush vs time), masked composite |
| `hybrid` | Footage + generator lane, masked composite |
| `grab_bag` | Original behavior — independent step draws from pool |
| `stutter` | Rapid-fire short segments, hard cuts |
| `echo_chamber` | Stacked echo effects at increasing delays |
| `warp_focus` | Generator with heavy warp chain, minimal post |
| `edge_poster` | Posterize + edge_glow pairing, single or two-lane |

Auto-selected from eligible set based on context (src, n_lanes, use_generators). Force via `archetype="deep_time"` etc. Complexity still scales all parameters within the chosen archetype.

**Centralized via boutique stacks**: All archetypes use `_shader_step()` which picks a random boutique stack from `boutique_stacks.yaml` instead of assembling random shader combinations. This means developing new boutique stacks automatically enriches every archetype and show reel.

**Blend modes**: brightness-neutral only (overlay, softlight, difference, multiply, normal). Screen and add removed — they compound generator brightness problems.

- **Performance impact**: complexity 0.2 renders at ~2–3x realtime; complexity 0.9 at ~11x realtime (for 3 min video)
- **Overrides**: `n_lanes`, `n_steps`, `n_segments`, `use_transitions`, `use_generators`, `seed`, `target_dur`, `archetype`

**Recipe utilities**: `print_recipe()` pretty-prints, `hash_recipe()` returns 8-char hex hash for output naming.

## Boutique Shader Stacks

`boutique_stacks.yaml` in project root — hand-curated shader combinations with per-shader parameter randomization. This is the single source of visual identity for the pipeline: all archetypes draw from these stacks.

17 stacks: thermal_scan, bruised_film, neon_circuit, cyanotype_ghost, corroded_signal, halftone_punk, dither_print, vhs_archive, deep_fringe, data_destruction, crosshatch_noir, stained_glass, signal_decay, edge_poster, terrain_scan, terrain_voxel, terrain_abyss.

**Parameter resolution** (via `_resolve_shader_params()`):
- Scalar (int/float/string) → used as-is
- 2-element list `[min, max]` → `rng.uniform(min, max)`
- Object `{choice: [...]}` → `rng.choice(list)`

**Pipeline path**: `load_boutique_stacks()` → `_shader_step()` picks random stack → `ShaderStep(shader_paths=..., param_overrides=...)` → `_resolve_shaders_for_step()` passes overrides → `apply_shader_stack(param_overrides=...)` → moderngl GPU uniforms.

**Design**: Stacks are pure shader chains (ShaderStep + NormalizeStep). Time effects come from archetypes, not from stacks. Intensity params pinned at defaults; only character-defining params (exposure, hue, cell_size, etc.) are randomized.

**Adding a new boutique stack**: Add an entry to `boutique_stacks.yaml` with `shaders` list and optional `shader_params`. It will automatically be available to all archetypes and show reels.

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

**Batch generation**: Generate multiple random show reels for curation, each with a unique seed.

```
python -m pipeline.flows.show_reel batch 10 -n 8 --src source.mp4 --footage-ratio 0.5 --min-complexity 0.5 --max-complexity 0.9
```

Brain wipe flows use `ConcurrentTaskRunner(max_workers=3)`. Show reel and stooges flows use `max_workers=4`.

## Stooges Flow

`pipeline/flows/stooges.py` — multi-channel CRT TV content for OBS composite. Each channel draws random segments from source, runs a crush sandwich (crush → 3 shaders → crush → 3 shaders → normalize) per segment, shuffles, interleaves with TV static, and concatenates. Uses `ConcurrentTaskRunner(max_workers=4)` — all segment pipelines across all channels run concurrently with future-based dependency chaining. Confirmed thread-safe for moderngl shader tasks.

```
python -m pipeline.flows.stooges input/footage.mp4 \
    --n-channels 5 --segment-counts 8,10,12,14,16 --seed 42
```

## Show Reel Flow (Production Ready)

`pipeline/flows/show_reel.py` — channel-surfing through heterogeneous "shows". Each show is a short (5–18s) clip rendered at a random complexity level via `random_recipe` + `brain_wipe` subflow, so some are raw warped generators and others have crush, shaders, time effects, etc. Shows are joined with random per-pair transitions. **Output is consistently usable for stream content as of 2026-02-28.**

Supports optional source footage — when `--src` is provided, each show independently flips a coin at `--footage-ratio` probability to decide whether it uses footage or a generator. `--footage-ratio 1.0` guarantees all-footage (no generator fallback). `--src input/` (directory) picks random .mp4 files per show.

```
# Generator-only reel
python -m pipeline.flows.show_reel run -n 15 --min-dur 10 --max-dur 15 --seed 777

# Mixed footage + generator reel (70% footage, random files from directory)
python -m pipeline.flows.show_reel run -n 9 --src input/ --footage-ratio 0.7 --seed 42

# Force a specific archetype for all shows
python -m pipeline.flows.show_reel run -n 8 --src input/footage.mp4 --footage-ratio 1.0 --archetype deep_time --seed 42

# Batch: 9 reels at once
python -m pipeline.flows.show_reel batch 9 -n 9 --seed 8400 --src input/ --footage-ratio 0.7 --min-complexity 0.3 --max-complexity 0.9 --min-dur 10 --max-dur 18

# Target reel duration instead of show count (auto-calculates n_shows)
python -m pipeline.flows.show_reel batch 20 --reel-dur 120 --src input/ --seed 6060

# Human-in-the-loop: plan → edit manifest → render
python -m pipeline.flows.show_reel plan -n 8 --seed 777 --src input/footage.mp4
python -m pipeline.flows.show_reel render output/show_reel_777_manifest.json
```

Key parameters: `n_shows` (number of segments), `reel_dur` (target reel duration in seconds — auto-calculates `n_shows` from avg show duration), `min_dur`/`max_dur` (per-show duration range), `min_complexity`/`max_complexity` (complexity range per show), `transition_dur`, `width`/`height`, `src` (optional source footage or directory), `footage_ratio` (0.0–1.0, default 0.4), `archetype` (force all shows to use a specific archetype), `seed`. When both `n_shows` and `reel_dur` are omitted, defaults to 20 shows.

Note: connect to persistent Prefect server via `PREFECT_API_URL=http://127.0.0.1:4200/api` for UI visibility. Without it, flows start ephemeral servers.

## Shader Library

Shaders in `shaders/` directory, organized by function. Shaders use ISF v2 format with `CATEGORIES` tags in the JSON header for filtering.

### Glitch/color shaders (in `shaders/`)

**Spatial/distortion**: warp, echo, pixel_sort, databend, scan_tear, bit_crush, feedback_zoom, block_shift, stereo_project, posterize, lens_warp, shear, video_heightfield
**Color/tone**: duotone, chromatic_shift, gradient_map, edge_glow, cyanotype, heat_sig, oxide, neon_bleed, bruise
**3D/visualization**: video_heightfield — terrain/heightfield renderer (params: MAXHEIGHT, SLICES, OVERDRIVE, MOVEX/Y/Z, XROTATE, YROTATE)
**Additional** (from `shaders-to-eval/` ISF community): ~89 shaders across blur, color, distortion, film, generator, geometry, glitch, halftone, stylize, tile categories. Plus additional shaders like crosshatch, duotone_isf, kaleidoscope, pixellate, rgb_halftone, solarize, bad_tv, convergence, sepia_tone, dither_bayer used in boutique stacks.

Deleted shaders (for reference): chromawave, false_color, plasma_tint, palette_cycle, interference, solarize_color, mirror_fracture, droste, swirl, tunnel, ulp-warp-kaleidoscope — removed for aesthetic or functional reasons.

### Brain wipe / warp shaders (in `brain-wipe-shaders/`)

Separate shader directory from the glitch/color library. Use `--shader-dir brain-wipe-shaders` with the brain wipe flows, or pass the path as `shader_dir` in Python.

**6 video warpers (have `inputImage`):**
- `ulp-warp-fbm` — fractal domain warp (iterative FBM noise, 1–3 passes)
- `ulp-warp-curl` — curl noise flow (divergence-free, no tearing)
- `ulp-warp-gravitational` — multi-point gravitational lensing (up to 5 orbiting masses)
- `ulp-warp-voronoi` — Voronoi cell refraction (convex/concave/edge-push modes)
- `ulp-warp-ripple` — concentric sine-wave displacement (amplitude, frequency, decay)
- `ulp-warp-twist` — radial spiral distortion (twist amount, radius, falloff)

**12 generators (no `inputImage`):** All tuned for dynamic range (mean 60–140, p5–p95 range >80).
- `ulp-brain-wipe-plasma` — sinusoidal plasma field
- `ulp-brain-wipe-tunnel` — infinite geometric tunnel (smoothstep soft edges)
- `ulp-brain-wipe-chladni` — vibrating plate resonance figures (exponential glow halo)
- `ulp-brain-wipe-rd` — reaction-diffusion simulator (smoothstep contrast expansion)
- `ulp-infernal-drift` — domain-warped FBM fire (counterpart to abyssal_drift)
- `ulp-bioluminescent-field` — deep-sea bioluminescence (ambient scatter + wide glow)
- `abyssal_drift` — deep-sea ambient drift
- `ulp-curl-flow` — curl noise flow field visualization
- `ulp-domain-warp-cascade` — cascading domain warp layers
- `ulp-moire-interference` — optical moire interference patterns
- `ulp-rotating-geometry` — rotating geometric forms
- `ulp-voronoi-flow` — animated Voronoi cell flow

All are tagged `"Warp"`, `"Brain Wipe"`, or `"Generator"` in ISF `CATEGORIES` and compatible with Magic Music Visuals for live use.

### Shader categories and filtering

Shaders declare categories in their ISF `CATEGORIES` header array. The `filter_shaders()` utility in `brain_wipe.py` filters by category and/or by whether the shader has an `inputImage` input (warper vs generator). The `ISFShader` dataclass exposes `.categories` (list[str]) and `.image_inputs` (list of inputs with type `"image"`).

## Filter Chain Merging

When processing recipes, consecutive steps of the same kind are fused to eliminate intermediate encode/decode cycles. Implemented in `brain_wipe.py` via `_group_steps()` which recognizes three categories:

**1. FFmpeg filter chain** — consecutive pure-ffmpeg steps merged into a single `-vf "filter1,filter2,..."` command via `_apply_filter_chain` task.
- Mergeable types: `MirrorStep`, `ZoomStep`, `InvertStep`, `HueShiftStep`, `SaturateStep`, `NormalizeStep`.

**2. Fused time effect chain** — consecutive time effect steps processed on a single in-memory frame buffer via `fused_time_chain` task. One decode, N effects, one encode.
- Fuseable types: all time step types (`ScrubStep`, `DriftStep`, `PingPongStep`, `EchoStep`, `PatchStep`, `SlitScanStep`, `TemporalTileStep`, `QuadLoopStep`, `SmearStep`, `BloomStep`, `StackStep`, `SlipStep`, `FlowWarpStep`, `TemporalSortStep`, `ExtremaHoldStep`, `FeedbackTransformStep`).
- Each effect has a `_process_*` helper that operates on `list[np.ndarray]` — no file I/O. The `@task` wrappers are thin I/O shells.
- `_apply_time_effect()` dispatches step dataclasses to helpers via `match`/`case`.
- Seed consumption matches `_submit_step` exactly: seedless steps (`EchoStep`, `SmearStep`, `BloomStep`, `StackStep`) get `None`.
- Deep time recipes (5–8 stacked time effects) benefit most — ~5x fewer encode/decode cycles.

**3. Singleton** — everything else (`CrushStep`, `ShaderStep`) processed individually via `_submit_step()`.

Single-step groups still go through `_submit_step()` — merging/fusing only activates for runs of 2+.

## Concurrency & Parallelism

The pipeline supports parallel execution at multiple levels, controlled by `Config` fields and Prefect's tag-based concurrency limits.

**Flow-level parallelism** (via `ThreadPoolExecutor`):
- `max_parallel_lanes` (default: 2) — lanes within a single `brain_wipe` run process concurrently. Single-lane recipes skip threading entirely.
- `max_parallel_shows` (default: 2) — shows within a single `show_reel_render` run render concurrently. Set to 1 for sequential (legacy) behavior.
- Each thread gets its own `contextvars.copy_context()` snapshot so Prefect task submissions find the flow's task runner.

**Task-level parallelism** (via Prefect `ConcurrentTaskRunner`):
- `brain_wipe`: `max_workers=3` — segments within a lane process concurrently.
- `show_reel` / `stooges`: `max_workers=4`.

**Global resource limits** (via Prefect tag-based concurrency limits):
- Tasks tagged `"gpu"`: `apply_shader`, `apply_shader_stack`, `apply_random_shader_stack` — limited by VRAM.
- Tasks tagged `"ram-heavy"`: all 17 time effect tasks + `fused_time_chain` — limited by system RAM.
- Create limits via: `prefect concurrency-limit create gpu 3` and `prefect concurrency-limit create ram-heavy 3`.
- Without a Prefect server or without limits created, tags are inert — no behavioral change.

**Hierarchy**:
```
show_reel_batch (sequential reels)
  show_reel_render (ThreadPoolExecutor, max_parallel_shows)
    brain_wipe per show (ConcurrentTaskRunner, max_workers=3)
      ThreadPoolExecutor for lanes (max_parallel_lanes)
        segments processed concurrently via Prefect futures
          individual tasks throttled by gpu/ram-heavy tags
```

## GPU Encoding (VideoToolbox)

`Config.gpu_encode` controls whether intermediate encodes use `h264_videotoolbox` (macOS hardware encoder) or `libx264`. **Default: disabled** (`gpu_encode=False`).

VideoToolbox is ~9x faster per-encode but unreliable on long batches — the hardware encoder crashes with SIGBUS after ~16 reels despite `-allow_sw 1`. Since encoding is a small fraction of total render time, the reliability loss outweighs the speed gain.

## Known Issues / TODO

- No audio passthrough — all tasks strip audio (`-an`). This is intentional for now (visual processing only) but could be added via ffmpeg's `-c:a copy` flag.
- Work directory cleanup is not automatic. Intermediate files accumulate in `work/`.

## Style

Python 3.10+. Type hints throughout. `from __future__ import annotations` in every module. `Path` objects for all file references (never raw strings). Prefect 3.x API only (no Prefect 2 patterns).
