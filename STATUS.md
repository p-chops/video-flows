# Video Pipeline — Session Status

## Stooges Flow: Complete

`pipeline/flows/stooges.py` — the production workflow for building multi-channel CRT TV content for OBS composite. Each channel is a sequence of crush-sandwiched segments interleaved with TV static bursts.

### Workflow per channel

1. Draw N random segments from source (random start, 1–30s duration)
2. Crush sandwich each segment: crush → 3 shaders → crush → 3 shaders → normalize
3. Shuffle segment order (per-channel RNG)
4. Interleave with static bursts
5. Re-encoding concat to final channel file

### Parallelism

Uses Prefect `ConcurrentTaskRunner` with 4 workers. All segment pipelines across all channels are submitted as futures with automatic dependency chaining (extract → crush1 → shade1 → crush2 → shade2 → normalize). Segments across channels run concurrently; the sequential chain within each segment is enforced by future dependencies. Shader tasks (moderngl) are confirmed thread-safe — no OpenGL concurrency issues observed.

### Seeking / playback fixes (hard-won)

Three bugs were found and fixed during development:

1. **Timestamp gaps in concat** — `concat -c copy` with mixed-fps segments caused gaps. Fixed by re-encoding concat instead of stream-copy.
2. **Moov atom at end + sparse keyframes** — added `-movflags +faststart`, `-g 48`, `-bf 0` to concat.
3. **Frozen video from fps truncation** — `extract_segment` used `f"{fps:.3f}"` which truncated 24000/1001 to 2997/125. The mismatch caused ffmpeg frame rate conversion with massive `dup`/`drop`. Fixed by using `str(fps)` for full precision.

### Prefect observability

All tasks are tracked in the Prefect UI (direct task calls, not `.fn()`). Run `prefect server start` and set `PREFECT_API_URL=http://localhost:4200/api` before running.

### CLI

```
python -m pipeline.flows.stooges input/footage.mp4 \
    --n-channels 5 \
    --segment-counts 8,10,12,14,16 \
    --seed 42 \
    -o output/channels_v4
```

### Existing output

- `output/channels_v3/channel_00–04.mp4` — 5-channel production run (2:03–4:49, varying lengths)
- `output/channels_v4/channel_00–05.mp4` — 6-channel production run (segment counts 8,10,12,14,16,9, seed 500)
- `output/channels_parallel/channel_00–02.mp4` — 3-channel concurrency test run

## Brain Wipe Flows: Working with Multi-Pass Support

`pipeline/flows/brain_wipe.py` — two flows for warp and distortion shader processing, with category-aware filtering.

### `warp_chain`

Apply a chain of warp shaders to source footage. Single input → single output. Filters to `["Warp", "Brain Wipe"]` ISF categories by default, restricting to video warpers (shaders with `inputImage`). Supports explicit shader paths or random selection. Normalizes output levels by default.

```
python -m pipeline.flows.brain_wipe warp-chain source/footage.mp4 --n-shaders 2 --seed 7
```

### `brain_wipe_render`

Pre-render a long-form brain wipe sequence for streaming. No source footage needed — generator shaders synthesize all content from scratch. Each segment gets a random generator followed by a random number of warp shaders (configurable depth range). Supports configurable resolution, frame rate, and segment count. Outputs per-segment previews.

```
python -m pipeline.flows.brain_wipe brain-wipe-render \
    --shader-dir brain-wipe-shaders \
    -n 12 --segment-dur 20 --min-shaders 2 --max-shaders 6 \
    --width 1280 --height 720 --fps 60 --seed 42
```

Key parameters: `--width`/`--height`/`--fps` (default 1920x1080@30), `--min-shaders`/`--max-shaders` (default 2/6, total stack depth including generator), `--no-normalize`, `--no-shuffle`.

### Multi-pass ISF shader support

`_apply_shader_stack` in `shader.py` now supports multi-pass ISF shaders with persistent buffers. This enables reaction-diffusion, feedback, and other stateful shaders that accumulate state across frames.

Implementation: for each shader with `PASSES` containing persistent targets, a ping-pong FBO pair is created per buffer. On each frame, passes execute sequentially — buffer passes render to their persistent FBO (then swap read/write), and the final output pass renders to the chain FBO. Float textures (`FLOAT: true`) are supported for high-precision simulation buffers.

Single-pass shaders continue to work exactly as before — multi-pass logic is only activated when a shader declares persistent buffer passes.

### Shader timing fix

`_apply_shader_stack` uses deterministic frame-based timing (`frame_idx / fps`) instead of wall-clock `time.monotonic()`. This eliminates stutter from variable render times — every frame gets an exact `1/fps` time step.

### Level-control param pinning

Warp shader params `brightness` and `desaturate`, and generator params `brightness` and `bg_brightness`, are pinned to their shader-declared defaults during randomization. This prevents warp chains from crushing output to near-black. Creative params (warp_strength, scale, flow_speed, etc.) are still fully randomized.

### `filter_shaders` utility

Filters a `load_shader_dir` dict by ISF `CATEGORIES` tags and/or `inputImage` presence. Used to partition the shader library into warpers (have `inputImage`) vs generators (no `inputImage`). Exposed from `pipeline.flows.brain_wipe`.

### Brain wipe shaders (in `brain-wipe-shaders/`)

**4 warpers**: `ulp-warp-fbm`, `ulp-warp-curl`, `ulp-warp-gravitational`, `ulp-warp-voronoi`

**7 generators**: `ulp-brain-wipe-plasma`, `ulp-brain-wipe-tunnel`, `ulp-brain-wipe-chladni`, `ulp-brain-wipe-rd`, `ulp-abyssal-jellies-v4`, `ulp-bioluminescent-field`, `abyssal_drift`

Use `--shader-dir brain-wipe-shaders` with the brain wipe flows. Default category filter is `["Warp", "Brain Wipe", "Generator"]`.

### Generator brightness levels

Without normalization, generators vary widely in brightness:

| Generator | Avg Y | Notes |
|-----------|-------|-------|
| `ulp-brain-wipe-rd` | 193–232 | Brightest — reaction-diffusion, purple/cyan/white palette |
| `ulp-brain-wipe-tunnel` | 80–110 | Good mid-range brightness |
| `ulp-brain-wipe-plasma` | 60–90 | Acceptable |
| `ulp-brain-wipe-chladni` | 40–60 | Dark unless bg_brightness is raised |
| `abyssal_drift` | ~38 | Dark — deep-sea palette |
| `ulp-abyssal-jellies-v4` | 17–20 | Very dark — tiny bright features |
| `ulp-bioluminescent-field` | 20–30 | Very dark |

The RD shader significantly improves the unnormalized mix. More bright generators would further reduce dependence on normalization.

### RD inject_density tuning

The RD shader's `inject_density` parameter controls how often new B-chemical patches are seeded into the simulation. Finding the sweet spot required a parameter sweep and a shader fix.

**Hash ceiling bug**: The original injection used a single hash evaluation per patch per frame: `hash2(patch * 7.3 + hash1(frame) * 97.3) > (1.0 - density)`. With 625 patches (25x25 grid) x 900 frames = 562,500 evaluations, the maximum hash value creates a hard ceiling. Below `1.0 - max_hash`, ZERO injections fire. Above it, many fire at once. This produced a binary step — no injection vs. too much — with no intermediate behavior.

**Fix**: Combined two independent hash evaluations via `fract(roll1 + roll2)`. This breaks the single-hash ceiling and produces a smooth probability distribution, allowing very low inject_density values to produce rare events.

**Rescaled parameter range**: `inject_density` was remapped from the internal 0–0.002 range to a user-facing 0–1 range (internal: `inject_density * 0.002`). This makes it usable as a single knob in MMV. The sweet spot for gradual pattern formation (blocks lighting up occasionally, patterns growing and connecting) is around 0.15–0.25.

Sweep results (30s renders, 1280x720@30fps, all other params at defaults):

| inject_density | avg Y | last 5s Y | character |
|---|---|---|---|
| 0.0 | 27.6 | 21.0 | No injection — seed decays |
| 0.10 | 28.2 | 22.1 | First hints of activity |
| 0.15 | 29.4 | 24.7 | Occasional blocks |
| 0.25 | 38.6 | 44.9 | Patterns forming & connecting |
| 0.50 | 50.5 | 58.4 | Active pattern formation |
| 1.0 | 60.5 | 60.8 | Full density (old default) |

Sweep script: `sweep_rd_inject.py`. Output: `output/rd_inject_sweep_rescaled/`.

### ISF bool/event fix

ISF `"TYPE": "bool"` and `"event"` inputs become `uniform float` in translated GLSL. Four shaders had bare `if (var)` which fails in `#version 330` — fixed to `if (var > 0.5)`: `ulp-brain-wipe-tunnel.fs`, `ulp-brain-wipe-plasma.fs`, `ulp-brain-wipe-chladni.fs`, `ulp-brain-wipe-rd.fs`.

### `abyssal_drift.fs` header fix

Had comment lines before the ISF `/*{` header. The ISF parser uses `re.match` (anchored to start of file), so the comments prevented parsing. Fixed by removing the leading comments.

## Compositing Lab: Dense Random Compositing

`pipeline/flows/compositing_lab.py` — generates N independent composite samples, each built from a random recipe of source layers and compositing operations. Runs samples concurrently as Prefect subflows via ThreadPoolExecutor for full UI observability.

### Recipe structure per sample

1. **Layer generation** (3–6 layers): raw footage segment, shader-processed segment, generator shader render, TV static, or solid colour
2. **Layer pre-processing** (50% chance per layer): warp through 1–2 spatial shaders (from `shaders/` spatial subset + brain-wipe warp collection)
3. **Compositing operations** (2–5 ops, weighted random selection):
   - **blend** — random blend mode (screen/add/multiply/overlay/difference/softlight), random opacity 0.2–0.8
   - **masked** — masked merge using randomly-generated mask (luma/edge/motion/gradient) from one of the two input layers
   - **self_keyed** — mask derived from the overlay's own content (luma/edge/motion/gradient/chroma), creating content-dependent transparency
   - **chromakey** — HSV colour removal using random hue target, revealing base through transparent regions
   - **pip** — picture-in-picture at random position and scale
   - **multi_pip** — 2–3 scaled overlays placed onto a single base
4. **Post-processing** (50% chance each): random shader, bitrate crush 0.7–1.0, normalize levels

### Operation weights

Self-keyed (4) and chromakey (3) are weighted higher than blend (3), masked (2), pip (2), multi_pip (2) — transparency-based operations are the core capability for dense layered compositions.

### CLI

```
python -m pipeline.flows.compositing_lab input/footage.mp4 \
    -n 8 --seed 42 --segment-dur 10 \
    --shader-dir shaders --brain-wipe-dir brain-wipe-shaders \
    -o output/comp_lab
```

### Tasks added

- `picture_in_picture` in `composite.py` — ffmpeg overlay filter, scales overlay to fraction of base width, positions at (x, y)

## Crush Task: Fixed and Tuned

Built `bitrate_crush` in `pipeline/tasks/glitch.py` — a codec-abuse effect that encodes at punishingly low quality to introduce compression artifacts, then re-encodes clean to bake them in.

Went through several iterations to eliminate a "flash and decay" artifact where the video would periodically flash bright and gradually recover:

1. **CBR bitrate mode** — flashed due to VBV buffer oscillation and keyframe starvation
2. **2-pass CBR** — still flashed, encoder can't represent scene changes at 50kbps
3. **CRF mode** — still flashed, CRF varies QP per frame based on complexity
4. **Fixed QP + scene detection disabled** — fixed it. The real culprits were x264's auto-inserted scene-change keyframes (which look brighter than P-frames at extreme QP) and CRF's per-frame QP variation.

Final design: fixed QP, single keyframe at start, `-sc_threshold 0`, `-keyint_min` = total frames.

Added `downscale` parameter — shrinks before crushing, nearest-neighbor upscale baked into the dirty pass for bigger, blockier artifacts.

`crush` parameter: 0.0 (mild) to 1.0 (maximum destruction). x264 at 0.9–1.0 is the sweet spot for this project's aesthetic.

## Shader Library: 21 + 11

**Glitch/color (21)**: Original library in `shaders/`. Added previously: `lens_warp`, `shear`. Deleted previously: `swirl`, `tunnel`.

**Brain wipe (11)**: In `brain-wipe-shaders/`. 4 warpers + 7 generators. See brain wipe section for details.

## Datamosh: Dropped

Attempted an AVI-based datamosh task (strip I-frame keyframe flags to create motion-vector hallucination). The approach of flipping idx1 index flags doesn't work — decoders read actual VOP headers in the bitstream, not the index. Removed entirely.

## Flows Updated

- `crush_lab` and `deep_color` flows updated to use `crush` float parameter (replaces old `bitrate`/`gop` params)
- `stooges` — multi-channel CRT TV content (production flow)
- `warp_chain` + `brain_wipe_render` — warp/distortion shader flows
- `compositing_lab` — dense random compositing with keying, masking, warping, and multi-PIP
- All production flows use `ConcurrentTaskRunner(max_workers=4)` or `ThreadPoolExecutor` for parallel execution
- All flows registered and exported from `flows/__init__.py`
