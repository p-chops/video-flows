#!/usr/bin/env python3
"""
Procedural stacks.yaml generator.

Given a folder of ISF shaders, this script:
1. Filters to shaders that parse and compile successfully
2. Classifies them as processors (have inputImage) vs generators
3. Generates random-slug-named stacks with variety and coverage
4. Writes a stacks.yaml file

Usage:
    python scripts/generate_stacks.py packs/my_pack/shaders/
    python scripts/generate_stacks.py packs/my_pack/shaders/ --n-stacks 20 --seed 42
    python scripts/generate_stacks.py packs/my_pack/shaders/ -o packs/my_pack/stacks.yaml
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import yaml

# Add project root to path
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from pipeline.isf import ISFShader, parse_isf
from pipeline.gl import GLContext, VERT_SHADER


# ─── Slug generation ─────────────────────────────────────────────────────────

_ADJECTIVES = [
    "broken", "burnt", "cold", "corroded", "crushed", "dark", "deep",
    "dirty", "distant", "dream", "drowned", "dusk", "faded", "false",
    "fever", "frozen", "ghost", "glass", "glow", "gold", "half",
    "hollow", "hot", "iron", "liquid", "lost", "low", "melted", "midnight",
    "neon", "night", "null", "oxide", "pale", "phantom", "raw", "red",
    "rich", "rust", "salt", "shadow", "sharp", "sick", "signal", "silent",
    "slow", "soft", "solar", "static", "steel", "stone", "strange",
    "sub", "sun", "swamp", "toxic", "ultra", "void", "warm", "wet", "wild",
]

_NOUNS = [
    "ash", "beam", "bite", "bleed", "bloom", "blur", "bone", "burn",
    "cage", "chain", "chrome", "circuit", "coil", "core", "crash",
    "crust", "crystal", "current", "cut", "dance", "decay", "depth",
    "drain", "drift", "dust", "echo", "edge", "engine", "fade",
    "field", "film", "fire", "flash", "flood", "flow", "flux",
    "fog", "fold", "forge", "fossil", "frost", "furnace", "gate",
    "ghost", "grain", "grid", "grind", "haze", "heat", "hole",
    "hum", "ink", "lab", "lens", "light", "loop", "mask", "mesh",
    "mirror", "moth", "nerve", "noise", "orbit", "phase", "pipe",
    "pit", "plate", "plume", "pool", "press", "prism", "pulse",
    "rain", "rift", "ring", "ruin", "scan", "scorch", "scrap",
    "screen", "seed", "shade", "shell", "shift", "signal", "slab",
    "slice", "slide", "smear", "smoke", "snap", "soak", "spark",
    "spine", "spiral", "spray", "stain", "steam", "sting", "storm",
    "strand", "stripe", "surge", "swirl", "tape", "tear", "thread",
    "tide", "tomb", "tone", "trace", "trail", "trap", "tremor",
    "tunnel", "valve", "vapor", "vein", "vine", "void", "wash",
    "wave", "web", "well", "wire", "worm", "wound", "wreck", "zone",
]


def _random_slug(rng: random.Random, used: set[str]) -> str:
    """Generate a unique adj_noun slug."""
    for _ in range(200):
        slug = f"{rng.choice(_ADJECTIVES)}_{rng.choice(_NOUNS)}"
        if slug not in used:
            used.add(slug)
            return slug
    # fallback: numbered
    n = len(used)
    slug = f"stack_{n:03d}"
    used.add(slug)
    return slug


# ─── Shader validation ───────────────────────────────────────────────────────

def _try_compile(gl: GLContext, shader: ISFShader) -> bool:
    """Try to compile a parsed ISF shader. Returns True on success."""
    try:
        prog = gl.compile(shader.glsl_source, VERT_SHADER)
        prog.release()
        return True
    except Exception:
        return False


def validate_shaders(
    shader_dir: Path,
) -> tuple[list[ISFShader], list[ISFShader], list[tuple[str, str]]]:
    """Parse and compile-test all .fs files in a directory.

    Returns:
        (processors, generators, failures)
        processors: shaders with inputImage (can process video)
        generators: shaders without inputImage (synthesize from scratch)
        failures: list of (filename, error_message)
    """
    fs_files = sorted(shader_dir.glob("*.fs"))
    if not fs_files:
        print(f"No .fs files found in {shader_dir}")
        return [], [], []

    gl = GLContext()
    processors: list[ISFShader] = []
    generators: list[ISFShader] = []
    failures: list[tuple[str, str]] = []

    for fs_path in fs_files:
        # Parse
        try:
            shader = parse_isf(fs_path)
        except Exception as e:
            failures.append((fs_path.name, f"parse error: {e}"))
            continue

        # Compile
        if not _try_compile(gl, shader):
            failures.append((fs_path.name, "compile error"))
            continue

        # Classify
        has_input_image = any(i.name == "inputImage" for i in shader.image_inputs)
        if has_input_image:
            processors.append(shader)
        else:
            generators.append(shader)

    gl.release()
    return processors, generators, failures


# ─── Parameter randomization spec ────────────────────────────────────────────

def _param_spec_for_input(inp) -> Any:
    """Generate a YAML param randomization spec for an ISF input.

    Returns None if the input shouldn't be randomized (e.g., images, events).
    """
    if inp.type == "image":
        return None

    if inp.type == "event":
        return None

    if inp.type == "bool":
        # Randomize as choice
        return {"choice": [0.0, 1.0]}

    if inp.type == "float":
        lo = float(inp.min) if inp.min is not None else 0.0
        hi = float(inp.max) if inp.max is not None else 1.0
        default = float(inp.default) if inp.default is not None else (lo + hi) / 2

        # For intensity-like params, pin near default to avoid crushing output
        name_lower = inp.name.lower()
        intensity_names = {
            "brightness", "bg_brightness", "desaturate", "intensity",
            "opacity", "alpha",
        }
        if name_lower in intensity_names:
            # Pin at default
            return default

        # For other float params, use a range
        # Narrow the range somewhat — full range can be extreme
        spread = (hi - lo) * 0.6
        center = default
        range_lo = max(lo, center - spread / 2)
        range_hi = min(hi, center + spread / 2)
        # Ensure some range exists
        if abs(range_hi - range_lo) < 0.001:
            return default
        return [round(range_lo, 4), round(range_hi, 4)]

    if inp.type == "long":
        lo = int(inp.min) if inp.min is not None else 0
        hi = int(inp.max) if inp.max is not None else 10
        default = int(inp.default) if inp.default is not None else (lo + hi) // 2
        if lo == hi:
            return default
        # Use choice with a few values spanning the range
        n_choices = min(hi - lo + 1, 5)
        step = max(1, (hi - lo) // (n_choices - 1))
        choices = sorted(set(
            [lo] + [lo + step * i for i in range(1, n_choices - 1)] + [hi]
        ))
        if len(choices) == 1:
            return choices[0]
        return {"choice": choices}

    if inp.type == "point2D":
        # Skip — these are usually positional and best left at default
        return None

    if inp.type == "color":
        # Skip — color params are complex and best left at default
        return None

    return None


def _build_shader_params(
    shaders: list[ISFShader],
) -> dict[str, dict[str, Any]]:
    """Build shader_params spec for a stack's shaders."""
    params: dict[str, dict[str, Any]] = {}

    for shader in shaders:
        shader_params: dict[str, Any] = {}
        for inp in shader.param_inputs:
            spec = _param_spec_for_input(inp)
            if spec is not None:
                shader_params[inp.name] = spec
        if shader_params:
            params[shader.path.stem] = shader_params

    return params


# ─── Stack generation ─────────────────────────────────────────────────────────

def generate_stacks(
    processors: list[ISFShader],
    generators: list[ISFShader],
    *,
    n_stacks: int | None = None,
    seed: int = 42,
) -> dict[str, dict]:
    """Generate random stacks from validated shaders.

    Strategy:
    - Each processor appears in at least 1 stack (coverage)
    - Stack sizes: 1–4 processors
    - Generators are NOT included in stacks (they're used as sources)
    - Minimize repeat appearances until coverage is met
    - After coverage, fill remaining stacks with varied combinations

    Returns a dict of stack_name → {shaders: [...], shader_params: {...}}
    """
    if not processors:
        print("No processor shaders available — cannot generate stacks.")
        return {}

    rng = random.Random(seed)

    # Default: 2.5 * sqrt(n_processors), minimum 4.
    # Birthday-problem scaling — keeps expected repeat rate per reel
    # roughly constant as the shader library grows.
    if n_stacks is None:
        import math
        n_stacks = max(4, round(2.5 * math.sqrt(len(processors))))

    used_slugs: set[str] = set()
    stacks: dict[str, dict] = {}

    # Track how many times each shader has been used
    usage_count: dict[str, int] = {s.path.stem: 0 for s in processors}
    shader_by_stem: dict[str, ISFShader] = {s.path.stem: s for s in processors}

    def _pick_shaders(size: int) -> list[ISFShader]:
        """Pick shaders for a stack, preferring least-used ones."""
        stems = list(usage_count.keys())
        # Sort by usage count (least used first), break ties randomly
        stems.sort(key=lambda s: (usage_count[s], rng.random()))
        picked = stems[:size]
        rng.shuffle(picked)  # randomize order within the stack
        for s in picked:
            usage_count[s] += 1
        return [shader_by_stem[s] for s in picked]

    # Phase 1: coverage pass — ensure every shader appears at least once
    uncovered = list(shader_by_stem.keys())
    rng.shuffle(uncovered)

    while uncovered and len(stacks) < n_stacks:
        # Pick stack size: 2–3 for coverage phase
        size = rng.choice([2, 2, 3, 3])
        size = min(size, len(uncovered) + 1)  # don't exceed available

        # Start with an uncovered shader
        pick = [uncovered.pop(0)]
        usage_count[pick[0]] += 1

        # Fill rest from least-used
        remaining_stems = [s for s in shader_by_stem if s != pick[0]]
        remaining_stems.sort(key=lambda s: (usage_count[s], rng.random()))
        for s in remaining_stems[:size - 1]:
            if s in uncovered:
                uncovered.remove(s)
            pick.append(s)
            usage_count[s] += 1

        rng.shuffle(pick)
        shader_objs = [shader_by_stem[s] for s in pick]
        params = _build_shader_params(shader_objs)

        slug = _random_slug(rng, used_slugs)
        entry: dict[str, Any] = {"shaders": pick}
        if params:
            entry["shader_params"] = params
        stacks[slug] = entry

    # Phase 2: variety pass — fill to n_stacks with diverse combos
    while len(stacks) < n_stacks:
        size = rng.choices([1, 2, 3, 4], weights=[1, 4, 4, 1])[0]
        size = min(size, len(processors))

        shader_objs = _pick_shaders(size)
        pick = [s.path.stem for s in shader_objs]
        params = _build_shader_params(shader_objs)

        slug = _random_slug(rng, used_slugs)
        entry = {"shaders": pick}
        if params:
            entry["shader_params"] = params
        stacks[slug] = entry

    return stacks


# ─── YAML output ──────────────────────────────────────────────────────────────

def _represent_choice(dumper, data):
    """Custom representer for {choice: [...]} dicts — keep on one line."""
    if "choice" in data and len(data) == 1:
        return dumper.represent_mapping("tag:yaml.org,2002:map", data, flow_style=True)
    return dumper.represent_mapping("tag:yaml.org,2002:map", data)


def _represent_range(dumper, data):
    """Custom representer for [min, max] lists — keep on one line."""
    if len(data) == 2 and all(isinstance(v, (int, float)) for v in data):
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data)


def write_stacks_yaml(
    stacks: dict[str, dict],
    output_path: Path,
    *,
    processors: list[ISFShader],
    generators: list[ISFShader],
    failures: list[tuple[str, str]],
) -> None:
    """Write stacks.yaml with header comments."""
    # Custom YAML dumper for compact lists and dicts
    class CompactDumper(yaml.SafeDumper):
        pass

    CompactDumper.add_representer(dict, _represent_choice)
    CompactDumper.add_representer(list, _represent_range)

    header_lines = [
        "# Auto-generated shader stacks",
        f"# {len(stacks)} stacks from {len(processors)} processor shaders",
    ]
    if generators:
        header_lines.append(
            f"# {len(generators)} generator shaders available (not included in stacks)"
        )
    if failures:
        header_lines.append(f"# {len(failures)} shaders failed validation (see below)")
    header_lines.append("#")
    header_lines.append("# Parameter resolution:")
    header_lines.append("#   scalar       → used as-is")
    header_lines.append("#   [min, max]   → rng.uniform(min, max)")
    header_lines.append("#   {choice: []} → rng.choice(list)")
    header_lines.append("")

    yaml_str = yaml.dump(
        {"stacks": stacks},
        Dumper=CompactDumper,
        default_flow_style=False,
        sort_keys=False,
        width=120,
    )

    with open(output_path, "w") as f:
        f.write("\n".join(header_lines))
        f.write(yaml_str)

    # Append failure comments at end
    if failures:
        with open(output_path, "a") as f:
            f.write("\n# ─── Shaders that failed validation ───\n")
            for name, reason in failures:
                f.write(f"#   {name}: {reason}\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate stacks.yaml from a folder of ISF shaders"
    )
    parser.add_argument(
        "shader_dir",
        type=Path,
        help="Directory containing .fs shader files",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output path (default: <shader_dir>/../stacks.yaml)",
    )
    parser.add_argument(
        "-n", "--n-stacks",
        type=int,
        default=None,
        help="Number of stacks to generate (default: ~1.3x shader count)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate shaders only, don't write stacks",
    )

    args = parser.parse_args()

    shader_dir = args.shader_dir.resolve()
    if not shader_dir.is_dir():
        print(f"Error: {shader_dir} is not a directory")
        sys.exit(1)

    # Default output: sibling of shaders/ dir
    output_path = args.output or (shader_dir.parent / "stacks.yaml")

    print(f"Scanning {shader_dir} ...")
    processors, generators, failures = validate_shaders(shader_dir)

    # Report
    print(f"\n  Processors (have inputImage): {len(processors)}")
    for s in processors:
        cats = ", ".join(s.categories) if s.categories else "uncategorized"
        n_params = len(s.param_inputs)
        print(f"    {s.path.stem:40s}  [{cats}]  ({n_params} params)")

    print(f"\n  Generators (no inputImage):   {len(generators)}")
    for s in generators:
        cats = ", ".join(s.categories) if s.categories else "uncategorized"
        print(f"    {s.path.stem:40s}  [{cats}]")

    if failures:
        print(f"\n  Failed ({len(failures)}):")
        for name, reason in failures:
            print(f"    {name}: {reason}")

    if args.dry_run:
        print("\nDry run — no stacks generated.")
        return

    if not processors:
        print("\nNo processor shaders to stack. Exiting.")
        sys.exit(1)

    # Generate
    stacks = generate_stacks(
        processors,
        generators,
        n_stacks=args.n_stacks,
        seed=args.seed,
    )

    # Check for existing file
    if output_path.exists():
        print(f"\nWarning: {output_path} already exists — overwriting.")

    write_stacks_yaml(
        stacks, output_path,
        processors=processors,
        generators=generators,
        failures=failures,
    )

    print(f"\nWrote {len(stacks)} stacks to {output_path}")

    # Summary stats
    all_stems = [stem for s in stacks.values() for stem in s["shaders"]]
    unique_stems = set(all_stems)
    print(f"  Coverage: {len(unique_stems)}/{len(processors)} processors used")
    print(f"  Avg stack size: {len(all_stems) / len(stacks):.1f} shaders")


if __name__ == "__main__":
    main()
