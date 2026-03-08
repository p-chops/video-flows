# Starter Pack

The starter pack ships with the video-flows repository and provides everything needed to generate output on first run — no additional shaders or configuration required.

**30 shaders** (13 processors, 17 generators) and **24 evolved stacks**.

## Processors

Processors take video input (`inputImage`) and transform it.

| Shader | Description | Key params |
|--------|-------------|------------|
| `starter_duotone` | Map luminance to two colors with contrast curve | `shadow_hue`, `highlight_hue`, `saturation`, `contrast`, `mix_amount` |
| `starter_edges` | Sobel edge detection with adjustable threshold, line width, and 3 color modes (white-on-black, source color, luminance) | `threshold`, `line_width`, `mix_amount`, `invert_bg`, `color_mode` |
| `starter_feedback` | Temporal feedback loop — composites current frame over a zoomed/rotated echo of the previous output. Dark regions become transparent, revealing spiraling trails | `feedback_mix`, `zoom_amount`, `rotate_amount`, `dark_threshold`, `color_drift` |
| `starter_glow` | Edge-detecting glow with color tinting via Sobel filter | `glow_strength`, `hue`, `brightness` |
| `starter_invert_luma` | Selective luminance inversion with mix control — inverts brightness while preserving hue | `mix_amount`, `threshold` |
| `starter_kaleido` | Kaleidoscope — polar mirror symmetry with rotation | `segments`, `spin`, `zoom` |
| `starter_macroblocks` | Simulates H.264 macroblock compression artifacts: stale blocks frozen on old content, DCT quantisation, block-edge ringing | `block_size`, `stale_prob`, `ringing`, `quantize` |
| `starter_mosaic` | Pixelation mosaic with color quantization — chunky blocks with limited palette | `block_size`, `color_levels` |
| `starter_scanline` | CRT scanlines with phosphor glow, barrel distortion, and rolling sync bar | `line_count`, `line_darkness`, `barrel`, `phosphor`, `roll_speed` |
| `starter_shift` | RGB channel offset — splits channels apart for chromatic aberration | `amount`, `angle` |
| `starter_warp` | Sine-wave domain warping — animated layered displacement field | `amplitude`, `frequency`, `speed`, `layers` |
| `ulp-warp-fbm` | FBM domain warp — iterative fractal noise distortion (Inigo Quilez-style). Three warp passes for deeply structured organic distortion | `warp_strength`, `warp_passes`, `scale`, `octaves`, `drift_speed` |
| `ulp-warp-gravitational` | Gravitational lensing — orbiting point masses bend video like light around black holes | `warp_strength`, `num_masses`, `orbit_speed`, `orbit_radius`, `attraction`, `falloff` |

## Generators

Generators produce visuals from math — no video input needed. Used as source content for generator-mode shows and warp chains.

| Shader | Description | Key params |
|--------|-------------|------------|
| `starter_cells` | Animated Voronoi cells with drifting seed points — organic, cell-like textures | `cell_count`, `speed`, `edge_width`, `brightness` |
| `starter_drift` | Animated FBM noise flow with cosine palette — continuously evolving abstract patterns | `flow_speed`, `complexity`, `palette_shift`, `brightness` |
| `starter_plasma` | Classic plasma — layered sine waves with cosine palette | `scale`, `speed`, `palette_shift`, `warp`, `brightness` |
| `starter_rings` | Concentric rings with radial pulse and rotation | `ring_count`, `speed`, `palette_shift`, `brightness` |
| `starter_tunnel` | Infinite tunnel — raymarched concentric geometry with color banding | `speed`, `twist`, `shape`, `palette_shift`, `brightness` |
| `starter_weave` | Interlocking sine waves producing a woven textile pattern with diagonal scrolling | `scale`, `speed`, `color_cycle`, `brightness` |
| `ulp-bioluminescent-field` | Deep-sea bioluminescence — drifting organisms pulsing with cold light, depth parallax, propagating bloom | `density`, `drift_speed`, `pulse_rate`, `glow_size`, `depth_layers`, `bloom_spread` |
| `ulp-brain-wipe-chladni` | Chladni resonance figures — standing wave nodal patterns from vibrating plate math | `mode_m`, `mode_n`, `morph_speed`, `line_width`, `scale` |
| `ulp-brain-wipe-plasma` | Plasma field — overlapping sinusoidal color pools, silky and endlessly morphing | `speed`, `scale`, `complexity`, `color_hue`, `brightness` |
| `ulp-brain-wipe-rd` | Gray-Scott reaction-diffusion — self-organizing chemical patterns (two-pass: persistent RD simulation + colorization) | `feed`, `kill`, `diff_a`, `diff_b`, `inject_density`, `sim_scale` |
| `ulp-brain-wipe-tunnel` | Hypnotic tunnel — infinite geometric corridor with circular/polygonal cross-sections | `speed`, `rotation_speed`, `sides`, `depth_freq`, `angular_freq` |
| `ulp-curl-flow` | Curl noise flow field — color advected through a velocity field producing flowing paint-like streams | `flow_speed`, `turbulence`, `trail_length`, `vortex_scale`, `brightness` |
| `ulp-domain-warp-cascade` | Recursive FBM where each layer warps the next — alien landscape formations that slowly morph | `warp_depth`, `morph_speed`, `terrain_scale`, `palette_mode`, `ridge_enhance` |
| `ulp-infernal-drift` | Domain-warped FBM rising through fire — turbulent flame structures, floating embers, heat shimmer | `burn_rate`, `turbulence`, `heat`, `ember_density`, `flame_scale` |
| `ulp-moire-interference` | Overlapping rotating line/circle grids producing complex emergent beating patterns | `pattern_type`, `density`, `rotation_speed`, `n_layers`, `brightness` |
| `ulp-rotating-geometry` | Concentric rotating polygons and spirals — hard-edged, rhythmic, fast geometric motion | `n_rings`, `spin_speed`, `shape_complexity`, `spiral_twist`, `brightness` |
| `ulp-voronoi-flow` | Animated Voronoi cells with drifting seeds — organic cellular topology that splits, merges, and flows | `cell_scale`, `flow_speed`, `edge_glow`, `color_mode`, `warp_amount` |

## Stacks

Stacks are curated shader chains with per-shader parameter randomization ranges, evolved via `vf pack evolve` for visual quality and diversity. Each stack is applied as a `ShaderStep` followed by a `NormalizeStep`.

| Stack | Chain |
|-------|-------|
| `shadow_fade` | mosaic → fbm warp |
| `burnt_prism` | invert_luma → gravitational → feedback → edges → scanline |
| `frozen_mesh` | warp → invert_luma → glow |
| `distant_edge` | kaleido → edges → glow → fbm warp |
| `signal_drain` | duotone → edges |
| `rich_tomb` | invert_luma → mosaic → feedback → warp → edges |
| `cold_burn` | scanline → warp → fbm warp → feedback → gravitational |
| `dark_loop` | feedback → shift → edges → warp |
| `fever_bone` | duotone → kaleido → fbm warp → invert_luma → glow |
| `raw_hum` | edges → duotone → invert_luma → feedback |
| `slow_tide` | scanline → duotone → warp → glow |
| `fever_tunnel` | edges → scanline → warp |
| `rich_pulse` | gravitational → edges → duotone → kaleido |
| `strange_beam` | mosaic → glow → duotone → gravitational |
| `static_frost` | feedback → edges → fbm warp |
| `silent_tomb` | edges → shift |
| `hollow_pulse` | warp → mosaic → edges → duotone → fbm warp |
| `dream_loop` | duotone → shift → edges |
| `static_slice` | fbm warp → glow → scanline → invert_luma → edges |
| `deep_drift` | scanline → mosaic → edges → gravitational → shift |
| `liquid_dust` | feedback → edges |
| `hot_smear` | edges → scanline |
| `rust_plume` | feedback → kaleido → gravitational → shift → edges |
| `strange_circuit` | glow → mosaic → edges → fbm warp → shift |

Stack sizes range from 2 to 5 shaders. Parameter randomization ranges are defined in `stacks.yaml` — each render produces a unique variant within the stack's character.
