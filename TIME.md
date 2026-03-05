# Temporal Effects

The video pipeline treats time as a material — something that can be folded, sorted, sliced, and recombined. Every temporal effect takes a video and produces a same-duration video where time itself has been restructured.

## Architecture

Time effects live in `pipeline/tasks/time.py`. Each effect has two parts:

1. **`_process_*` helper** — a pure function that operates on an in-memory frame sequence. No file I/O.
2. **`@task` wrapper** — a thin shell that decodes input, calls the helper, encodes output.

This separation enables **fused time chains**: when a recipe stacks multiple time effects, the pipeline decodes once, runs all the `_process_*` helpers in sequence on the in-memory frames, and encodes once. A 5-effect deep_time recipe does 1 decode + 1 encode instead of 10.

### Memory models

Each effect uses a memory strategy appropriate to its access pattern:

| Model | How it works | Used by |
|-------|-------------|---------|
| **All frames in RAM** | Load entire clip as `list[ndarray]` | scrub, drift, ping-pong, slit_scan, slip, temporal_tile, temporal_sort |
| **Streaming** | Process frame-by-frame, fixed-size state | echo (ring buffer), smear (1 canvas), flow_warp (prev frame), feedback_transform (prev output), scan_refresh (float32 canvas) |
| **Volumetric** | Full `(T,H,W,3)` numpy array | temporal_fft, temporal_gradient, axis_swap, depth_slice |

`FrameBuffer` manages memory automatically — small clips stay in RAM, large ones spill to `np.memmap` temp files, and clips exceeding `max_ram_mb` are rejected before decoding starts.

### Recipe integration

The `random_recipe()` generator picks time effects from a weighted pool. Each archetype uses them differently:

- **deep_time** — 3–5 stacked time effects, pure temporal destruction
- **cascade** — 2–3 domains from {shaders, time, crush} in random order; the time domain contributes 1–3 effects

## Effects

### Playhead effects

These manipulate *when* the playhead reads from the source timeline.

**scrub** — The playhead wanders through the source at varying speeds, sometimes reversing. A smoothed random walk controls position. High intensity = wild jumps; low intensity = gentle drift. The signature "stuttering VHS" look.
- Key params: `smoothness` (0.01–0.2), `intensity` (0.3–0.9)

**drift** — A fixed-length loop window slides gradually through the source. Each cycle plays the same material, but the window drifts forward by a small amount. Creates a meditative, slowly-evolving repetition.
- Key params: `loop_dur` (1.0–4.0s), `drift` (0.1–0.5)

**ping_pong** — A subsegment plays forward then backward in a breathing rhythm. The window length and position are randomized. Creates a "breathing" or "pulsing" quality.
- Key params: `window` (1.0–4.0s)

**quad_loop** — Four polyrhythmic loops at different rates composited into a 2×2 grid or horizontal/vertical bands. Each quadrant loops a different subsegment at a different speed, creating cross-rhythms.
- Key params: `layout` (grid_2x2 / vertical_bands / horizontal_bands), `loop_durs` (4 floats)

### Feedback and accumulation

These build up visual material over time through recursive processes.

**echo** — Motion trails via ring-buffer blending. Each frame is mixed with frames from N steps ago. Moving objects leave ghostly afterimages that fade. High trail values make new frames barely register.
- Key params: `delay` (0.03–0.15s), `trail` (0.3–0.6)

**feedback_transform** — Each frame is blended with a spatially-transformed version of the previous output. The transform accumulates recursively: zoom creates infinite tunnels, rotate creates spirals, translate creates smears.
- Key params: `transform` (zoom / rotate / translate / spiral), `amount`, `mix` (0.3–0.55)

**scan_refresh** — CRT phosphor beam simulation. A scan line sweeps across the frame; pixels boost at the beam then exponentially decay. Creates a visible refresh sweep. Brightness-preserving: the boost factor compensates for decay so the time-average matches the original.
- Key params: `speed` (0.5–2.0), `decay` (1.0–4.0), `beam_width` (0.02–0.1), `axis` (horizontal / vertical)

### Spatial-temporal hybrids

These treat video as a 3D volume (x, y, time) and rearrange the mapping between space and time.

**slit_scan** — Each row (or column) of the output samples from a different moment in time. The scan position sweeps through the source, so the top of the frame might show 2 seconds ago while the bottom shows now. A classic video art technique — creates flowing, time-smeared imagery.
- Key params: `axis` (horizontal / vertical), `scan_speed` (0.5–2.0)

**temporal_tile** — The frame is divided into a grid; each tile shows a different moment from the source. Like a surveillance wall where every monitor is slightly out of sync.
- Key params: `grid` (2–6), `offset_scale` (0.1–0.5)

**slip** — The frame is divided into horizontal or vertical bands, each offset independently in time. Adjacent bands show different moments, creating an interlacing/tearing glitch.
- Key params: `n_bands` (3–12), `max_slip` (0.1–0.5), `axis` (horizontal / vertical)

**depth_slice** — An angled plane cuts through the spacetime volume. The slice angle determines how much space vs time you see — shallow angles show mostly spatial content with slight temporal offset; steep angles show dramatic time displacement across the frame.
- Key params: `angle` (10–80°), `axis` (horizontal / vertical)

**axis_swap** — Swaps the time axis with a spatial axis, literally viewing the video from the side. The output shows the temporal evolution of a single row or column stretched across the frame. Produces abstract streak imagery.
- Key params: `axis` (horizontal / vertical), `position` (0.0–1.0)

### Motion-reactive

These respond to what's actually moving in the frame.

**flow_warp** — Computes optical flow between frames and uses it to displace pixels. Motion becomes *more* motion — movement exaggerates itself. Needs real motion in the source to do anything; it's a no-op on static content.
- Key params: `amplify` (2.0–8.0), `smooth` (5–21)

**datamosh** — Simulates I-frame loss by propagating motion vectors from optical flow without refreshing the reference frame. The image gradually drifts apart from reality as motion accumulates. Set `refresh_interval` high for continuous drift.
- Key params: `refresh_interval` (20–99999 frames), `flow_scale` (0.8–1.2)

**temporal_displace** — Each pixel's brightness determines *when* it samples from. Bright pixels pull from the future; dark pixels pull from the past. The image becomes self-referential — its content determines its own temporal structure.
- Key params: `max_offset` (0.1–0.5), `channel` (luma / red / green / blue)

### Sorting and quantization

These reorder or reduce frames based on pixel values.

**temporal_sort** — For each pixel position, collects all values across time and sorts them by luminance. The output plays from darkest to brightest (or vice versa). Dark regions emerge first, then midtones, then highlights. Creates a geological stratification effect.
- Key params: `mode` (luminance / hue / saturation / red / green / blue), `direction` (ascending / descending)

**temporal_morph** — Morphological operations (min/max/open/close) applied along the time axis with a sliding window. Min = temporal erosion (darkest survives), max = temporal dilation (brightest survives). Open/close combine both for noise removal.
- Key params: `operation` (dilate / erode / open / close), `kernel_size` (3–15)

### Frequency domain

These operate on the video's temporal frequency content via FFT.

**temporal_fft** — Transforms each pixel's timeline into the frequency domain, applies a filter, then transforms back. Five modes: low_pass (smooth/dreamy), high_pass (only motion), band_pass (specific rhythms), band_stop (remove rhythms), spectral_gate (threshold-based).
- Key params: `mode` (low_pass / high_pass / band_pass / band_stop / spectral_gate), `cutoff` (0.05–0.5)

### Removed from random pool

These effects are still available for direct recipe construction, but are excluded from automatic recipe generation because they tend toward stasis — they compound with other effects to freeze the output.

**extrema_hold** — Tracks per-pixel min or max over time. Mode='max' drives to white; mode='min' drives to black. With decay=0, the effect is permanent.

**smear** — Pixels freeze in place unless inter-frame change exceeds a threshold. Compounds with other motion-dampening effects.

**bloom** — Temporal edge detection — only passes pixels that changed significantly between frames. Static areas go black.

**patch** — Random rectangular patches overlay frozen frames from the past. A temporal collage.

**frame_quantize** — K-means clustering reduces the clip to K representative frames. High K preserves more; low K creates a posterized-in-time effect.

**temporal_gradient** — Per-pixel temporal derivative. First derivative shows velocity (motion edges); second derivative shows acceleration.

**temporal_equalize** — Per-pixel histogram equalization across time. Redistributes brightness values to maximize contrast at each pixel position.

**spectral_remix** — Rearranges FFT frequency bins (reverse, shuffle, rotate, interleave). Preserves magnitudes but reorders temporal frequencies.

**phase_scramble** — Randomizes temporal phases while preserving magnitudes. Each pixel's timeline has the same frequency content but scrambled timing.

**frame_stack** — Sliding window reduction (mean/max/min) across frames. Mean = motion blur; max/min = temporal extrema.

## Using time effects

Force the `deep_time` archetype to get pure temporal processing:

```bash
python -m pipeline.flows.show_reel run -n 8 --archetype deep_time --src input/clip.mp4 --footage-ratio 1.0 --seed 42
```

Or use `cascade` for time effects mixed with shaders and/or crush:

```bash
python -m pipeline.flows.show_reel run -n 8 --archetype cascade --src input/clip.mp4 --footage-ratio 1.0 --seed 42
```
