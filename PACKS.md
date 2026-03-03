# Shader Packs

Shader packs are self-contained bundles of ISF shaders and curated shader stacks. They're the visual building blocks of the video pipeline — every archetype, show reel, and recipe draws its shader combinations from whatever packs are installed.

A pack is just a directory under `packs/` with two things inside: shaders and a recipe for combining them.

```
packs/
├── starter/          ← ships with the repo (14 shaders, 12 stacks)
│   ├── shaders/
│   │   └── *.fs
│   └── stacks.yaml
└── my_pack/          ← your personal pack (gitignored)
    ├── shaders/
    │   ├── swirl.fs
    │   ├── grain.fs
    │   └── color_wash.fs
    └── stacks.yaml
```

When the pipeline runs, it auto-discovers every `packs/*/stacks.yaml` and loads them all into the shared pool. Every archetype (deep_time, polyrhythm, temporal_sandwich, etc.) randomly picks from this pool, so adding a new pack automatically enriches all generated output.

## Creating a pack

### 1. Make the directory structure

```bash
mkdir -p packs/my_pack/shaders
```

### 2. Add your ISF shaders

Drop `.fs` files into `packs/my_pack/shaders/`. Each shader must be a valid ISF v2 file: a JSON header inside `/*{ }*/` followed by GLSL body.

Minimal ISF header for a **processor** (takes video input):

```glsl
/*{
    "ISFVSN": "2",
    "DESCRIPTION": "What this shader does",
    "CATEGORIES": ["Color"],
    "INPUTS": [
        { "NAME": "inputImage", "TYPE": "image" },
        {
            "NAME": "amount",
            "TYPE": "float",
            "DEFAULT": 0.5,
            "MIN": 0.0,
            "MAX": 1.0
        }
    ]
}*/

void main() {
    vec2 uv = isf_FragNormCoord;
    vec4 color = IMG_NORM_PIXEL(inputImage, uv);
    // ... your effect ...
    gl_FragColor = color;
}
```

Minimal ISF header for a **generator** (no video input — produces visuals from math):

```glsl
/*{
    "ISFVSN": "2",
    "DESCRIPTION": "Procedural pattern generator",
    "CATEGORIES": ["Generator"],
    "INPUTS": [
        {
            "NAME": "speed",
            "TYPE": "float",
            "DEFAULT": 1.0,
            "MIN": 0.0,
            "MAX": 5.0
        }
    ]
}*/

void main() {
    vec2 uv = isf_FragNormCoord;
    // ... procedural visuals using TIME, uv, etc. ...
    gl_FragColor = vec4(color, 1.0);
}
```

The pipeline's ISF translator handles the conversion from ISF builtins to standard GLSL 330. You write ISF; it runs on moderngl.

**ISF builtins available:**

| ISF | What it does |
|-----|-------------|
| `isf_FragNormCoord` | Normalized UV (0–1) |
| `IMG_NORM_PIXEL(tex, uv)` | Sample texture at normalized coords |
| `IMG_PIXEL(tex, px)` | Sample texture at pixel coords |
| `RENDERSIZE` | Output resolution as `vec2` |
| `TIME` | Elapsed time in seconds |
| `FRAMEINDEX` | Current frame number |

**ISF input types:**

| Type | GLSL uniform | Notes |
|------|-------------|-------|
| `float` | `uniform float` | Specify DEFAULT, MIN, MAX |
| `bool` | `uniform float` | 0.0 or 1.0 |
| `long` | `uniform int` | Integer with VALUES/LABELS |
| `point2D` | `uniform vec2` | |
| `color` | `uniform vec4` | RGBA |
| `image` | `uniform sampler2D` | Video input |

### 3. Write stacks.yaml

A stack is a curated combination of shaders from your pack, applied in order. This is where the visual identity lives — stacks define *how* your shaders work together.

Create `packs/my_pack/stacks.yaml`:

```yaml
stacks:
  warm_grain:
    shaders: [color_wash, grain]
    shader_params:
      color_wash:
        warmth: [0.3, 0.8]
        tint: [0.0, 1.0]
      grain:
        amount: [0.1, 0.4]

  heavy_swirl:
    shaders: [swirl, grain, color_wash]
    shader_params:
      swirl:
        intensity: [0.5, 1.0]
        radius: {choice: [0.3, 0.5, 0.7]}
      grain:
        amount: 0.2
```

**Key rules:**

- Shader names in `shaders:` are stems (no `.fs` extension) — they must match files in your pack's `shaders/` directory.
- Shaders are applied in order (first shader processes the input, second processes the first's output, etc.).
- `shader_params` is optional. If omitted, shaders run with their ISF defaults.
- Every stack automatically gets a `NormalizeStep` appended (level-stretches the output so nothing clips to black).

### Parameter randomization syntax

The `shader_params` section supports three value formats:

| Format | Meaning | Example |
|--------|---------|---------|
| Scalar | Fixed value, used as-is | `amount: 0.5` |
| `[min, max]` | Random uniform in range | `amount: [0.2, 0.8]` |
| `{choice: [...]}` | Random pick from list | `mode: {choice: [0, 1, 2]}` |

This works for any type — floats, ints, and even vector values:

```yaml
shader_params:
  my_shader:
    intensity: [0.3, 0.9]           # float range
    cell_size: {choice: [8, 16, 32]} # pick one
    threshold: 0.5                    # fixed
    tint_color: {choice: [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]]}  # pick a vec4
```

**Design tip:** Pin intensity/brightness-related params to sensible defaults. Randomize character-defining params (hue, cell size, pattern choice, etc.). Full randomization of intensity params tends to crush output to near-black.

### 4. Verify it works

```bash
# Check that your stacks load
python -c "from pipeline.recipe import load_boutique_stacks; \
    stacks = load_boutique_stacks(); \
    print(f'{len(stacks)} stacks loaded'); \
    for name, shaders, _, base in stacks: \
        print(f'  {base.parent.name}/{name}: {shaders}')"

# Plan a show reel restricted to your pack
python -m pipeline.flows.show_reel plan -n 3 --seed 42 --pack my_pack

# Short test render
PREFECT_API_URL=http://127.0.0.1:4200/api \
    python -m pipeline.flows.show_reel run -n 2 --min-dur 3 --max-dur 5 \
    --seed 42 --pack my_pack --width 640 --height 360
```

## Using packs

### CLI

All flow CLIs accept `--pack` to restrict which packs are used. Repeatable for multiple packs.

```bash
# Use all installed packs (default)
python -m pipeline.flows.show_reel run -n 8 --seed 42

# Only use the ulp pack
python -m pipeline.flows.show_reel run -n 8 --seed 42 --pack ulp

# Only use starter
python -m pipeline.flows.show_reel run -n 8 --seed 42 --pack starter

# Use both starter and my_pack (exclude ulp)
python -m pipeline.flows.show_reel run -n 8 --seed 42 --pack starter --pack my_pack
```

`--pack` works on all subcommands: `plan`, `render`, `run`, `batch`.

### Python

```python
from pipeline.config import Config
from pipeline.recipe import random_recipe

# All packs
recipe = random_recipe(seed=42, complexity=0.6)

# Restricted to one pack
recipe = random_recipe(seed=42, complexity=0.6, packs=["my_pack"])

# Config object carries the filter everywhere
cfg = Config(packs=["my_pack"])
```

## Git and sharing

By default, `packs/*` is gitignored except `packs/starter/`. Your personal packs stay local.

To track a pack in git, add an exception to `.gitignore`:

```
packs/*
!packs/starter/
!packs/my_shared_pack/
```

## Tips

- **Start with one stack.** You can always add more. A single well-tuned stack is more useful than ten untested ones.
- **Test stacks with `plan` first.** `python -m pipeline.flows.show_reel plan -n 5 --seed 42 --pack my_pack` shows you exactly what recipes will be generated without rendering anything.
- **Shader order matters.** A posterize before an edge detector looks very different from the reverse. Experiment with ordering.
- **Stacks are pure shader chains.** Time effects (echo, drift, scrub, etc.) come from archetypes, not from stacks. Your stacks just define the visual processing — the pipeline wraps them in temporal structure automatically.
- **Mix processors and generators.** A pack can contain both. Processors are used in shader stacks; generators are used as source content for generator-mode shows (warp chains, etc.).
- **Name collisions.** If two packs have a shader with the same stem name, the last one loaded (alphabetical pack order) wins. Use a prefix to avoid this: `my_swirl.fs` instead of `swirl.fs`.
