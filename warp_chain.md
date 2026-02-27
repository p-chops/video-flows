# Brain Wipe / Warp Chain Flows

New flows and utilities for applying the brain wipe and warp distortion shaders
through the Prefect pipeline. Mirrors the structure of `stooges.py` and the
example flows, adding category awareness so the warp/brain-wipe shaders can be
targeted independently from the glitch and color shaders.

---

## New file: `pipeline/flows/brain_wipe.py`

### Utility: `filter_shaders`

```python
filter_shaders(shaders, categories=None, has_image_input=None)
```

Filters a `load_shader_dir` dict by ISF category tags and/or image-input
presence. Key distinctions for the new shader set:

- **Warpers** — have an `inputImage` input; displace or warp video frame-by-frame.
  (`ulp-warp-fbm`, `ulp-warp-curl`, `ulp-warp-gravitational`, `ulp-warp-voronoi`,
  `ulp-brain-wipe-chladni-video`, `ulp-brain-wipe-tunnel-video`)
- **Generators** — no `inputImage`; synthesize content from scratch, ignoring
  source video. (`ulp-brain-wipe-plasma`, `ulp-brain-wipe-tunnel`,
  `ulp-brain-wipe-chladni`)

Both sets are tagged with categories `"Warp"` or `"Brain Wipe"` in their ISF
headers, so `filter_shaders(..., categories=["Warp", "Brain Wipe"])` isolates
them from the rest of the library (glitch, color, etc.).

---

### Flow 1: `warp_chain`

Apply a chain of warp shaders to source footage. Single input → single output.

```
python -m pipeline.flows.brain_wipe warp-chain source/footage.mp4 \
    --n-shaders 2 \
    --seed 7
```

**What it does:**
1. Loads shader library, filters to `["Warp", "Brain Wipe"]` category, warpers only
2. Randomly picks `n_shaders` from the pool, randomises their float params
3. Applies as a single chained stack via `apply_shader_stack`
4. Normalizes levels on output

**Key params:**

| param | default | notes |
|---|---|---|
| `--n-shaders` | 2 | shaders to chain |
| `--categories` | `Warp` `Brain Wipe` | ISF category filter |
| `--all-shaders` | off | include generators (no inputImage) |
| `--shaders` | — | explicit shader paths; skips random selection |
| `--seed` | — | reproducibility |
| `--no-normalize` | off | skip level normalization |

Prints the full recipe (shader names + randomised param values) for every run.

---

### Flow 2: `brain_wipe_render`

Pre-render a long-form brain wipe sequence suitable for use as a stream asset
or OBS media source.

```
python -m pipeline.flows.brain_wipe brain-wipe-render source/footage.mp4 \
    -n 12 \
    --segment-dur 20 \
    --n-warp-shaders 2 \
    --use-generators \
    --seed 42
```

**What it does:**
1. Pulls `n_segments` random clips from source (each `segment_dur` seconds)
2. Per segment: optionally prepends a random generator shader, then chains
   `n_warp_shaders` random warpers
3. Normalizes each segment; saves per-segment previews to `output/brain_wipe_segments/`
4. Shuffles and concatenates into a single output file

**Key params:**

| param | default | notes |
|---|---|---|
| `-n` / `--n-segments` | 8 | number of segments |
| `--segment-dur` | 20.0 | seconds per segment |
| `--n-warp-shaders` | 2 | warp shaders per segment |
| `--use-generators` | off | prepend a generator (plasma/tunnel/chladni) before warping — replaces video content |
| `--no-shuffle` | off | preserve segment order |
| `--output-dir` | — | directory for per-segment preview files |
| `--seed` | — | master seed; per-segment seeds derived from it |

With `--use-generators`, each segment's chain is: generator → warp → warp.
The generator ignores the source video and synthesizes imagery from scratch;
the warp shaders then distort it. Without it, the warp shaders run directly
on source footage.

Segment previews appear in `output/brain_wipe_segments/` as they finish,
named `seg_NNN_<hash>.mp4`, so you can monitor results while the rest is processing.

---

## Registration

Both flows are registered in `pipeline/flows/examples.py`:
- Added to the `FLOWS` dict
- Added as subcommands to the examples CLI (`python -m pipeline.flows.examples`)

---

## New shaders (in `shaders/`)

The following shaders were written to feed these flows. Copy them into
`video_pipeline/shaders/` (or point `--shader-dir` at `memory/shaders/`) to
make them available to the pipeline.

### Generators (no inputImage — synthesize content)
| file | description |
|---|---|
| `ulp-brain-wipe-plasma.fs` | Sinusoidal plasma field — silky morphing color pools |
| `ulp-brain-wipe-tunnel.fs` | Infinite geometric tunnel — circular or polygonal cross-sections |
| `ulp-brain-wipe-chladni.fs` | Vibrating plate resonance figures — nodal line patterns |

### Video warpers (have inputImage — warp source footage)
| file | description |
|---|---|
| `ulp-brain-wipe-tunnel-video.fs` | Maps video onto tunnel surface using polar UV coordinates |
| `ulp-brain-wipe-chladni-video.fs` | Chladni gradient field as video displacement; optional nodal line overlay |
| `ulp-warp-fbm.fs` | Fractal domain warp — iterative FBM noise warping noise (1–3 passes) |
| `ulp-warp-curl.fs` | Curl noise flow — divergence-free field; no tearing or compression |
| `ulp-warp-gravitational.fs` | Multi-point gravitational lensing — up to 5 animated orbiting masses |
| `ulp-warp-voronoi.fs` | Voronoi cell refraction — convex lens, concave, or edge-push modes |

All shaders are tagged `"Brain Wipe"` or `"Warp"` in their ISF `CATEGORIES`
header and are compatible with Magic Music Visuals for live use.
