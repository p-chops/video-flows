# Video Flows

Automated video processing pipeline for dense, textural, otherworldly visuals. Takes source footage or generates visuals from scratch via GLSL shaders, then processes through shader stacks, temporal effects, codec crush, compositing, and transitions — all orchestrated with [Prefect](https://www.prefect.io/).

## Getting started

Download the release zip or clone the repository. Then, in that directory, create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Now install and test:

```bash
pip install .

# Generate a show reel — works immediately, no source footage needed
vf reel -n 8 --seed 42
```

Output lands in `output/`. That's it — the included **starter pack** has everything needed to produce output on first run.

### Bring your own shaders

Have ISF shaders? Create a pack and start rendering in two commands:

```bash
vf pack create my_effects ~/Downloads/cool_shaders/
vf reel -n 8 --pack my_effects --seed 42
```

`vf pack create` validates your shaders, removes any that don't compile, and auto-generates `stacks.yaml` with randomized shader combinations. See [PACKS.md](PACKS.md) for details.

## Shader packs

Shader packs are self-contained bundles of ISF shaders and curated combinations ("stacks") that live under `packs/`. The pipeline auto-discovers all installed packs at runtime.

```
packs/
├── starter/          ← included — 29 shaders, 24 stacks
│   ├── shaders/
│   └── stacks.yaml
└── my_pack/          ← add your own (vf pack create)
    ├── shaders/
    └── stacks.yaml
```

Use `--pack` to restrict which packs are used:

```bash
# Use only the starter pack
vf reel -n 8 --pack starter --seed 42

# Combine specific packs
vf reel -n 8 --pack starter --pack my_effects --seed 42

# All installed packs (default)
vf reel -n 8 --seed 42
```

See [PACKS.md](PACKS.md) for the full guide on creating and curating packs.

## CLI

The `vf` command is the main entry point. Six subcommands: `reel`, `show`, `shows`, `join`, `stack`, `pack`.

### Show reels

The show reel generates a sequence of short "shows" — each with a randomly chosen structure, complexity level, and shader stack — joined with random transitions.

```bash
# Generator-only (no source footage)
vf reel -n 8 --seed 42

# Process source footage (70% footage, 30% generators)
vf reel -n 10 --src input/footage.mp4 --footage-ratio 0.7 --seed 42

# Pick random files from a directory
vf reel -n 10 --src input/ --footage-ratio 0.7 --seed 42

# Control duration and complexity
vf reel --reel-dur 120 --min-dur 8 --max-dur 15 \
    --min-complexity 0.3 --max-complexity 0.8 --seed 42

# Force a specific archetype for all shows
vf reel -n 8 --archetype deep_time --seed 42

# Batch: generate 10 reels at once
vf reel batch 10 --reel-dur 60 --src input/ --seed 7070
```

#### Human-in-the-loop workflow

Plan a reel (preview recipes + save manifest), optionally edit the JSON, then render:

```bash
vf reel plan -n 8 --seed 777 --src input/footage.mp4
# edit work/reel_777_manifest.json if desired
vf reel render work/reel_777_manifest.json
```

### Single clips

Render a single clip via `vf show`:

```bash
# Random archetype, generator-only
vf show --seed 42

# Force an archetype with source footage
vf show input/footage.mp4 --archetype deep_time --seed 42

# Named recipe preset
vf show input/footage.mp4 --preset stooges --seed 42

# Control complexity and duration
vf show --archetype cascade --complexity 0.7 --duration 15 --seed 42
```

### Batch shows (generate → curate → join)

Generate individual clips for curation, then join the keepers into a reel:

```bash
# Generate 10 short deep_time clips from source footage
vf shows input/footage.mp4 -n 10 --duration 10 --archetype deep_time

# Output to a specific directory
vf shows input/footage.mp4 -n 20 --duration 8 -o curated/

# Watch, delete duds, then join survivors with random transitions
vf join curated/ -o final_reel.mp4 --transition random --transition-dur 0.5

# Join with a specific transition type
vf join curated/ -o final_reel.mp4 --transition crossfade

# Shuffle clip order before joining
vf join curated/ --shuffle --seed 42

# Join clips listed in a text file (one path per line)
vf join playlist.txt -o final_reel.mp4
```

`vf shows` defaults to sequential rendering (`--max-workers 1`). Increase for parallel rendering if you have RAM headroom.

### Named stacks

Run a specific shader stack by name via `vf stack`:

```bash
# Run a stack on source footage
vf stack crt_mosaic input/footage.mp4 --seed 42

# Run a stack with a generator (no source needed)
vf stack terrain_scan --seed 42

# Restrict to a specific pack
vf stack crt_mosaic --pack my_effects --duration 15 --seed 42
```

### Pack management

```bash
# List installed packs
vf pack list
vf pack list -v                  # include stack names

# Inspect a pack
vf pack info starter
vf pack info my_effects -v       # full stack chains and params

# Create a pack from a folder of ISF shaders
vf pack create my_effects ~/Downloads/cool_shaders/

# Regenerate stacks.yaml for an existing pack
vf pack stacks packs/my_effects/

# Evolve stacks via diversity-weighted random search
vf pack evolve packs/my_effects/ --candidates 2000 -n 20 --seed 42
```

## What's in the box

### Processing capabilities

- **Shader stacks** — chains of ISF v2 GLSL shaders applied via headless OpenGL (color, distortion, glitch, stylization)
- **Time effects** — 27 temporal manipulations (scrub, drift, ping-pong, echo, slit scan, temporal sort, feedback transform, datamosh, and more)
- **Codec crush** — intentional quality destruction via fixed-QP encoding (x264, mpeg2, mpeg4)
- **Compositing** — multi-lane blend, masked composite, picture-in-picture, chromakey
- **Transitions** — crossfade, luma wipe, whip pan, static burst, flash, slide, dissolve, zoom, pixelate, melt, interlace, squeeze

### Recipe system

Processing is described by **recipes** — declarative structures that specify sources, steps, lane compositing, and post-processing. The procedural generator (`random_recipe()`) chooses from 4 structural archetypes — three pure-domain poles plus a multi-domain mixer:

| Archetype | Weight | Structure |
|-----------|--------|-----------|
| `deep_space` | 3.0 | Pure shader stack — hand-curated boutique stacks, no time effects |
| `cascade` | 3.0 | 2–3 domains from {shaders, time, crush} in random order |
| `deep_time` | 2.0 | 3–5 stacked time effects, pure temporal destruction |
| `codec_crush` | 0.5 | Single crush pass → shader stack (rare, expensive) |

Complexity (0.0–1.0) scales parameters within the chosen archetype.

### Quality gate

After each show renders, a 5-feature quality classifier (brightness, contrast, motion, temporal variance, spatial entropy) checks the output. Dead or static clips are automatically rerolled with a new recipe.

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

For flow visibility, start a Prefect server:

```bash
prefect server start &
PREFECT_API_URL=http://127.0.0.1:4200/api vf reel -n 8 --seed 42
```

Without a server, everything still works — you just don't get the dashboard.

## License

MIT License. See [LICENSE](LICENSE).

## Note on use of AI

This codebase was developed using Claude Code and Opus 4.6.
