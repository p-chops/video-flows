# ULP Video Pipeline

Automated video processing pipeline for the Undersea Lair Project. Cuts, sequences, applies GLSL shaders to, time-warps, and composites video into dense, textural, otherworldly visuals for livestream content.

Built on [Prefect](https://www.prefect.io/) for orchestration, [ModernGL](https://github.com/moderngl/moderngl) for headless GPU shader rendering, and ffmpeg for all video I/O.

## What it does

The pipeline takes source footage (or generates visuals from scratch via GLSL generator shaders) and processes it through combinations of:

- **Shader stacks** — chains of ISF v2 GLSL shaders applied via headless OpenGL (color effects, distortion, glitch, stylization)
- **Time effects** — 21 temporal manipulations (scrub, drift, ping-pong, echo, slit scan, temporal FFT, axis swap, and more)
- **Codec crush** — intentional quality destruction via fixed-QP encoding (x264, mpeg2, mpeg4)
- **Compositing** — multi-lane blend, masked composite, picture-in-picture, chromakey
- **Transitions** — crossfade, luma wipe, whip pan, static burst, flash

Processing is described by **recipes** — declarative data structures that specify sources, processing steps, lane compositing, and post-processing. Recipes can be hand-crafted, built from presets, or procedurally generated.

## Requirements

- Python 3.10+
- ffmpeg and ffprobe on PATH
- OpenGL 3.3+ capable GPU (runs headless — no window needed)

## Installation

```bash
pip install -e .
```

This installs the `pipeline` Python package and its dependencies (prefect, moderngl, numpy, opencv-python-headless, PyYAML).

Shaders, boutique stack definitions, and source footage live outside the package and are resolved relative to the project root (override with `VP_PROJECT_ROOT` env var).

## Project layout

```
pipeline/               # Python package
├── config.py           # Config dataclass — paths, ffmpeg settings, defaults
├── recipe.py           # Recipe data model — sources, steps, lanes, compositing
├── ffmpeg.py           # Video I/O — probe, decode, encode, concat
├── gl.py               # Headless OpenGL context, FBO ping-pong
├── isf.py              # ISF v2 shader parser + GLSL translator
├── tasks/              # Prefect @task functions (atomic operations)
│   ├── cut.py          # Scene detection, segment extraction
│   ├── sequence.py     # Concat, shuffle, interleave, static generation
│   ├── shader.py       # Apply ISF shaders via ModernGL
│   ├── composite.py    # Blend, masked composite, PIP, chromakey
│   ├── mask.py         # Luma, edge, motion, chroma, gradient masks
│   ├── color.py        # Level normalization, quality probing
│   ├── glitch.py       # Codec-based compression artifacts
│   ├── time.py         # 21 temporal effects
│   ├── transform.py    # Mirror, zoom, invert, hue shift, saturate
│   └── transition.py   # Crossfade, wipe, whip pan, flash
└── flows/              # Prefect @flow compositions (pipelines)
    ├── brain_wipe.py   # Recipe executor, warp chains, generator renders
    ├── show_reel.py    # Channel-surfing show reel generator
    ├── stooges.py      # Multi-channel CRT TV content
    ├── time_lab.py     # Standalone time effect testing
    ├── transition_lab.py
    ├── compositing_lab.py
    └── examples.py

shaders/                # ISF v2 glitch/color/distortion shaders
brain-wipe-shaders/     # Generator + warp shaders
boutique_stacks.yaml    # Hand-curated shader stack definitions
input/                  # Source footage (not tracked)
work/                   # Intermediate files (not tracked)
output/                 # Final renders
```

## Quick start

### Show reel (recommended starting point)

The show reel generates a sequence of short "shows" — each with a randomly chosen archetype, complexity level, and shader stack — joined with random transitions.

```bash
# Generator-only reel (no source footage needed)
python -m pipeline.flows.show_reel run -n 8 --seed 42

# Mixed footage + generators (70% footage)
python -m pipeline.flows.show_reel run -n 10 --src input/footage.mp4 --footage-ratio 0.7 --seed 42

# Pick random files from a directory
python -m pipeline.flows.show_reel run -n 10 --src input/ --footage-ratio 0.7 --seed 42

# Control duration and complexity
python -m pipeline.flows.show_reel run --reel-dur 120 --min-dur 8 --max-dur 15 \
    --min-complexity 0.3 --max-complexity 0.8 --seed 42

# Force all shows to use a specific archetype
python -m pipeline.flows.show_reel run -n 8 --archetype deep_time --seed 42

# Batch: generate 10 reels at once
python -m pipeline.flows.show_reel batch 10 --reel-dur 60 --src input/ --seed 7070
```

### Human-in-the-loop workflow

Plan a reel (preview recipes + save manifest), optionally edit the manifest JSON, then render:

```bash
python -m pipeline.flows.show_reel plan -n 8 --seed 777 --src input/footage.mp4
# edit work/reel_777_manifest.json if desired
python -m pipeline.flows.show_reel render work/reel_777_manifest.json
```

### Recipe presets

Run a specific recipe directly:

```bash
# Temporal destruction — stacked time effects, no shaders
python -m pipeline.flows.brain_wipe brain-wipe --preset deep-time input/footage.mp4 --seed 42

# Generator render — GLSL generators + warp shaders, no source needed
python -m pipeline.flows.brain_wipe brain-wipe --preset generator-render -n 12 --seed 42

# Multi-channel CRT content
python -m pipeline.flows.stooges input/footage.mp4 --n-channels 5 --seed 42

# Warp chain — apply warp shaders to footage
python -m pipeline.flows.brain_wipe warp-chain input/footage.mp4 --n-shaders 3 --seed 42
```

Available presets: `crush-sandwich`, `stooges`, `generator-render`, `temporal-sandwich`, `deep-time`, `hybrid-composite`, `codec-spectrum`, `breathing-wall`, `erosion`, `palimpsest`, `generator-stooges`, `gradient-dissolve`, `accretion`, `time-cascade`, `temporal-geology`.

### Time effect lab

Test individual time effects on a clip:

```bash
python -m pipeline.flows.time_lab scrub input/clip.mp4 --intensity 0.7 --seed 42
python -m pipeline.flows.time_lab temporal-fft input/clip.mp4 --filter-type phase_scramble --seed 42
python -m pipeline.flows.time_lab axis-swap input/clip.mp4 --axis horizontal
```

Available effects: `scrub`, `drift`, `pingpong`, `echo`, `patch`, `temporal-fft`, `temporal-gradient`, `temporal-median`, `axis-swap`.

## Prefect UI

The pipeline uses Prefect for orchestration. For flow visibility in the Prefect UI, start a server and prefix commands with the API URL:

```bash
prefect server start &
PREFECT_API_URL=http://127.0.0.1:4200/api python -m pipeline.flows.show_reel run -n 8 --seed 42
```

Without a server, flows run with an ephemeral backend — everything still works, you just don't get the UI.

## Archetypes

The procedural recipe generator (`random_recipe()`) chooses from 11 structural archetypes that define how processing steps and lanes relate to each other:

| Archetype | What it does |
|-----------|-------------|
| `deep_time` | 5-8 stacked time effects, no shaders — pure temporal destruction |
| `temporal_sandwich` | Alternating time effect / shader stack pairs |
| `escalation` | Progressive parameter increase across steps or lanes |
| `polyrhythm` | 2-4 lanes at harmonically-related temporal rates, blended |
| `palimpsest` | Same source, two contrasting treatments, masked composite |
| `hybrid` | Footage + generator lane, masked composite |
| `grab_bag` | Independent random step draws |
| `stutter` | Rapid-fire short segments, hard cuts |
| `echo_chamber` | Stacked echo effects at increasing delays |
| `warp_focus` | Generator with heavy warp chain, minimal post |
| `edge_poster` | Posterize + edge glow pairing |

Complexity (0.0-1.0) scales parameters within the chosen archetype.

## Shaders

Shaders use [ISF v2](https://isf.video/) format — a JSON metadata header followed by GLSL fragment shader code. The pipeline's ISF translator handles the conversion to standard `#version 330` GLSL for ModernGL.

Two shader directories:
- `shaders/` — color, distortion, glitch, stylization effects (applied to existing video)
- `brain-wipe-shaders/` — generators (produce visuals from nothing) and warp shaders (distort existing video)

### Boutique stacks

`boutique_stacks.yaml` defines hand-curated shader combinations with per-shader parameter randomization. All archetypes draw from this file — it's the single source of visual identity for the pipeline.

```yaml
terrain_scan:
  shaders:
    - video_heightfield
    - gradient_map
  shader_params:
    video_heightfield:
      MAXHEIGHT: [0.3, 0.8]        # uniform random in range
      SLICES: 128
      OVERDRIVE: [1.0, 2.5]
    gradient_map:
      intensity: {choice: [0.5, 0.7, 0.9]}  # pick one
```

Parameter syntax: scalar (fixed value), `[min, max]` (uniform random), `{choice: [...]}` (pick one).

Adding a new stack to this file automatically makes it available to every archetype and show reel.

## Python API

```python
from pathlib import Path
from pipeline.config import Config
from pipeline.recipe import random_recipe, crush_sandwich_recipe
from pipeline.flows.brain_wipe import brain_wipe

# Run a preset recipe
recipe = crush_sandwich_recipe(Path("input/footage.mp4"), seed=42)
brain_wipe(recipe)

# Procedural recipe
recipe = random_recipe(
    src=Path("input/footage.mp4"),
    seed=42,
    complexity=0.6,
    archetype="deep_time",
    target_dur=15.0,
)
brain_wipe(recipe)
```

Output goes to `output/` by default. Intermediate files accumulate in `work/` and are not automatically cleaned up.

## License

MIT License. See [LICENSE](LICENSE).
