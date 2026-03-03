# ULP Video Pipeline

Automated video processing pipeline for dense, textural, otherworldly visuals. Takes source footage or generates visuals from scratch via GLSL shaders, then processes through shader stacks, temporal effects, codec crush, compositing, and transitions — all orchestrated with [Prefect](https://www.prefect.io/).

## Getting started

```bash
pip install -e .

# Generate a show reel — works immediately, no source footage needed
python -m pipeline.flows.show_reel run -n 8 --seed 42
```

Output lands in `output/`. That's it — the included **starter pack** has everything needed to produce output on first run.

## Shader packs

Shader packs are self-contained bundles of ISF shaders and curated combinations ("stacks") that live under `packs/`. The pipeline auto-discovers all installed packs at runtime.

```
packs/
├── starter/          ← included — 8 shaders, 5 stacks
│   ├── shaders/
│   └── stacks.yaml
└── my_pack/          ← add your own
    ├── shaders/
    └── stacks.yaml
```

Use `--pack` to restrict which packs are used:

```bash
# Use only the starter pack
python -m pipeline.flows.show_reel run -n 8 --pack starter --seed 42

# Combine specific packs
python -m pipeline.flows.show_reel run -n 8 --pack starter --pack my_effects --seed 42

# All installed packs (default)
python -m pipeline.flows.show_reel run -n 8 --seed 42
```

Creating a new pack is straightforward — write ISF v2 shaders, define stacks in YAML, drop them in `packs/`. See [PACKS.md](PACKS.md) for the full guide.

## Show reel

The show reel generates a sequence of short "shows" — each with a randomly chosen structure, complexity level, and shader stack — joined with random transitions.

```bash
# Generator-only (no source footage)
python -m pipeline.flows.show_reel run -n 8 --seed 42

# Process source footage (70% footage, 30% generators)
python -m pipeline.flows.show_reel run -n 10 --src input/footage.mp4 --footage-ratio 0.7 --seed 42

# Pick random files from a directory
python -m pipeline.flows.show_reel run -n 10 --src input/ --footage-ratio 0.7 --seed 42

# Control duration and complexity
python -m pipeline.flows.show_reel run --reel-dur 120 --min-dur 8 --max-dur 15 \
    --min-complexity 0.3 --max-complexity 0.8 --seed 42

# Force a specific archetype for all shows
python -m pipeline.flows.show_reel run -n 8 --archetype deep_time --seed 42

# Batch: generate 10 reels at once
python -m pipeline.flows.show_reel batch 10 --reel-dur 60 --src input/ --seed 7070
```

### Human-in-the-loop workflow

Plan a reel (preview recipes + save manifest), optionally edit the JSON, then render:

```bash
python -m pipeline.flows.show_reel plan -n 8 --seed 777 --src input/footage.mp4
# edit work/reel_777_manifest.json if desired
python -m pipeline.flows.show_reel render work/reel_777_manifest.json
```

## What's in the box

### Processing capabilities

- **Shader stacks** — chains of ISF v2 GLSL shaders applied via headless OpenGL (color, distortion, glitch, stylization)
- **Time effects** — 21 temporal manipulations (scrub, drift, ping-pong, echo, slit scan, temporal sort, feedback transform, and more)
- **Codec crush** — intentional quality destruction via fixed-QP encoding (x264, mpeg2, mpeg4)
- **Compositing** — multi-lane blend, masked composite, picture-in-picture, chromakey
- **Transitions** — crossfade, luma wipe, whip pan, static burst, flash

### Recipe system

Processing is described by **recipes** — declarative structures that specify sources, steps, lane compositing, and post-processing. The procedural generator (`random_recipe()`) chooses from 11 structural archetypes:

| Archetype | Structure |
|-----------|-----------|
| `deep_time` | 5-8 stacked time effects, pure temporal destruction |
| `temporal_sandwich` | Alternating time effect / shader stack pairs |
| `escalation` | Progressive parameter increase across steps or lanes |
| `polyrhythm` | 2-4 lanes at different temporal rates, blended |
| `palimpsest` | Same source, two contrasting treatments, masked composite |
| `hybrid` | Footage + generator lane, masked composite |
| `grab_bag` | Independent random step draws |
| `stutter` | Rapid-fire short segments, hard cuts |
| `echo_chamber` | Stacked echo effects at increasing delays |
| `warp_focus` | Generator with heavy warp chain |
| `edge_poster` | Posterize + edge glow pairing |

Complexity (0.0-1.0) scales parameters within the chosen archetype.

### Quality gate

After each show renders, a 5-feature quality classifier (brightness, contrast, motion, temporal variance, spatial entropy) checks the output. Dead or static clips are automatically rerolled with a new recipe.

## Recipe presets

Run specific recipe types directly:

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

## Time effect lab

Test individual time effects on a clip:

```bash
python -m pipeline.flows.time_lab scrub input/clip.mp4 --intensity 0.7 --seed 42
python -m pipeline.flows.time_lab echo input/clip.mp4 --delay 0.1 --trail 0.8
python -m pipeline.flows.time_lab drift input/clip.mp4 --loop-dur 2.0 --seed 42
```

## Python API

```python
from pathlib import Path
from pipeline.recipe import random_recipe
from pipeline.flows.brain_wipe import brain_wipe

recipe = random_recipe(
    src=Path("input/footage.mp4"),
    seed=42,
    complexity=0.6,
    archetype="deep_time",
    target_dur=15.0,
    packs=["starter"],
)
brain_wipe(recipe)
```

## Requirements

- Python 3.10+
- ffmpeg / ffprobe on PATH
- OpenGL 3.3+ GPU (headless — no window needed)

## Prefect UI (optional)

For flow visibility, start a Prefect server and prefix commands:

```bash
prefect server start &
PREFECT_API_URL=http://127.0.0.1:4200/api python -m pipeline.flows.show_reel run -n 8 --seed 42
```

Without a server, everything still works — you just don't get the dashboard.

## License

MIT License. See [LICENSE](LICENSE).
