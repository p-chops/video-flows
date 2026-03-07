"""
vf — Video Flows CLI.

Thin dispatcher that calls existing @flow functions from pipeline.flows.

    vf reel [run] -n 8 --seed 42
    vf reel plan -n 8 --seed 42
    vf reel render manifest.json
    vf reel batch 10 --reel-dur 120

    vf show input.mp4 --archetype deep_time --seed 42
    vf show --archetype cascade --seed 42

    vf stack crt_mosaic input.mp4 --seed 42
    vf stack terrain_scan --seed 42

    vf pack create my_effects ~/shaders/
    vf pack stacks packs/my_effects/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# ── Shared helpers ───────────────────────────────────────────────────────────

def _resolve_n_shows(args) -> int:
    """Resolve n_shows from args (explicit > reel_dur > default 20)."""
    from pipeline.flows.show_reel import _resolve_n_shows
    return _resolve_n_shows(args.n_shows, getattr(args, "reel_dur", None),
                            args.min_dur, args.max_dur)


def _add_reel_args(parser):
    """Add the shared show-reel plan/run/batch arguments."""
    parser.add_argument("-n", "--n-shows", type=int, default=None,
                        help="Number of shows")
    parser.add_argument("--reel-dur", type=float, default=None,
                        help="Target reel duration in seconds")
    parser.add_argument("--min-dur", type=float, default=5.0,
                        help="Min show duration (seconds)")
    parser.add_argument("--max-dur", type=float, default=10.0,
                        help="Max show duration (seconds)")
    parser.add_argument("--min-complexity", type=float, default=0.1)
    parser.add_argument("--max-complexity", type=float, default=0.6)
    parser.add_argument("--transition-dur", type=float, default=0.5)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--src", type=str, default=None,
                        help="Source footage file or directory")
    parser.add_argument("--footage-ratio", type=float, default=0.4,
                        help="Fraction of shows using footage (0.0–1.0)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--pack", action="append", dest="packs",
                        help="Restrict to shader packs (repeatable)")
    parser.add_argument("--archetype", type=str, default=None,
                        help="Force archetype for all shows")
    parser.add_argument("-o", "--output", type=str, default=None)
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Keep intermediate files")
    parser.add_argument("--motion-floor", type=float, default=0.005)
    parser.add_argument("--max-reroll", type=int, default=2)


# ── vf reel ──────────────────────────────────────────────────────────────────

def _add_reel_parsers(sub):
    p_reel = sub.add_parser("reel", help="Show reel generation")
    reel_sub = p_reel.add_subparsers(dest="reel_command")

    p_run = reel_sub.add_parser("run", help="Plan and render (default)")
    _add_reel_args(p_run)

    p_plan = reel_sub.add_parser("plan", help="Preview recipes, save manifest")
    _add_reel_args(p_plan)

    p_render = reel_sub.add_parser("render", help="Render from manifest JSON")
    p_render.add_argument("manifest", type=str, help="Path to manifest JSON")
    p_render.add_argument("-o", "--output", type=str, default=None)
    p_render.add_argument("--no-cleanup", action="store_true")
    p_render.add_argument("--motion-floor", type=float, default=0.005)
    p_render.add_argument("--max-reroll", type=int, default=2)
    p_render.add_argument("--pack", action="append", dest="packs")

    p_batch = reel_sub.add_parser("batch", help="Generate multiple reels")
    p_batch.add_argument("n_reels", type=int, help="Number of reels")
    _add_reel_args(p_batch)

    p_reel.set_defaults(func=_handle_reel)


def _handle_reel(args):
    from pipeline.config import Config
    from pipeline.flows.show_reel import (
        show_reel, show_reel_render, show_reel_batch,
        _plan_shows, _resolve_n_shows,
    )

    # Default to "run" if no subcommand
    cmd = getattr(args, "reel_command", None) or "run"
    packs = getattr(args, "packs", None)
    cfg = Config(packs=packs)

    if cmd == "plan":
        cfg.ensure_dirs()
        n_shows = _resolve_n_shows(args.n_shows, args.reel_dur,
                                   args.min_dur, args.max_dur)
        manifest = _plan_shows(
            n_shows=n_shows,
            min_dur=args.min_dur, max_dur=args.max_dur,
            min_complexity=args.min_complexity,
            max_complexity=args.max_complexity,
            transition_dur=args.transition_dur,
            width=args.width, height=args.height,
            src=Path(args.src) if args.src else None,
            footage_ratio=args.footage_ratio,
            seed=args.seed, archetype=args.archetype,
            output=Path(args.output) if args.output else None,
            cfg=cfg,
        )
        manifest_path = cfg.work_dir / f"reel_{manifest['seed']}_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"\n  manifest saved: {manifest_path}")
        print(f"  render with: vf reel render {manifest_path}")

    elif cmd == "render":
        manifest = json.loads(Path(args.manifest).read_text())
        out = Path(args.output) if args.output else None
        show_reel_render(
            manifest, output=out, cleanup=not args.no_cleanup,
            motion_floor=args.motion_floor, max_reroll=args.max_reroll,
            cfg=cfg,
        )

    elif cmd == "run":
        n_shows = _resolve_n_shows(args.n_shows, args.reel_dur,
                                   args.min_dur, args.max_dur)
        show_reel(
            n_shows=n_shows,
            min_dur=args.min_dur, max_dur=args.max_dur,
            min_complexity=args.min_complexity,
            max_complexity=args.max_complexity,
            transition_dur=args.transition_dur,
            width=args.width, height=args.height,
            src=Path(args.src) if args.src else None,
            footage_ratio=args.footage_ratio,
            seed=args.seed, archetype=args.archetype,
            output=Path(args.output) if args.output else None,
            cleanup=not args.no_cleanup,
            motion_floor=args.motion_floor,
            max_reroll=args.max_reroll,
            cfg=cfg,
        )

    elif cmd == "batch":
        n_shows = _resolve_n_shows(args.n_shows, args.reel_dur,
                                   args.min_dur, args.max_dur)
        show_reel_batch(
            n_reels=args.n_reels,
            n_shows=n_shows,
            min_dur=args.min_dur, max_dur=args.max_dur,
            min_complexity=args.min_complexity,
            max_complexity=args.max_complexity,
            transition_dur=args.transition_dur,
            width=args.width, height=args.height,
            src=Path(args.src) if args.src else None,
            footage_ratio=args.footage_ratio,
            seed=args.seed, archetype=args.archetype,
            cleanup=not args.no_cleanup,
            motion_floor=args.motion_floor,
            max_reroll=args.max_reroll,
            cfg=cfg,
        )


# ── vf show ──────────────────────────────────────────────────────────────────

def _add_show_parser(sub):
    p = sub.add_parser("show", help="Render a single clip")
    p.add_argument("src", nargs="?", type=str, default=None,
                   help="Source footage (omit for generator-only)")
    p.add_argument("--archetype", type=str, default=None,
                   help="Force archetype (deep_space, deep_time, cascade, codec_crush)")
    p.add_argument("--preset", type=str, default=None,
                   help="Named recipe preset (stooges, crush-sandwich, etc.)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--complexity", type=float, default=0.5)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--duration", type=float, default=10.0,
                   help="Target duration (seconds)")
    p.add_argument("--pack", action="append", dest="packs")
    p.add_argument("-o", "--output", type=Path, default=None)
    p.set_defaults(func=_handle_show)


def _handle_show(args):
    from pipeline.config import Config
    from pipeline.recipe import random_recipe
    from pipeline.flows.brain_wipe import brain_wipe

    cfg = Config(packs=getattr(args, "packs", None))
    cfg.ensure_dirs()

    src = Path(args.src) if args.src else None

    if args.preset == "stooges":
        from pipeline.flows.stooges import stooges_channels
        if not src:
            print("Error: stooges preset requires source footage")
            sys.exit(1)
        stooges_channels(src=src, seed=args.seed, cfg=cfg)
        return

    if args.preset:
        _handle_preset(args, src, cfg)
        return

    recipe = random_recipe(
        src=src,
        complexity=args.complexity,
        target_dur=args.duration,
        use_generators=src is None,
        seed=args.seed,
        archetype=args.archetype,
        width=args.width,
        height=args.height,
        packs=cfg.packs,
    )

    brain_wipe(recipe, output=args.output, cfg=cfg)


def _handle_preset(args, src, cfg):
    """Handle named recipe presets."""
    from pipeline.recipe import (
        crush_sandwich_recipe, stooges_recipe,
        generator_render_recipe, deep_time_recipe,
    )
    from pipeline.flows.brain_wipe import brain_wipe

    preset = args.preset
    seed = args.seed

    builders = {
        "crush-sandwich": lambda: crush_sandwich_recipe(src, seed=seed),
        "deep-time": lambda: deep_time_recipe(src, seed=seed),
        "generator-render": lambda: generator_render_recipe(seed=seed),
    }

    builder = builders.get(preset)
    if not builder:
        print(f"Error: unknown preset '{preset}'")
        print(f"Available: {', '.join(sorted(builders.keys()))}, stooges")
        sys.exit(1)

    recipe = builder()
    brain_wipe(recipe, output=args.output, cfg=cfg)


# ── vf stack ─────────────────────────────────────────────────────────────────

def _add_stack_parser(sub):
    p = sub.add_parser("stack", help="Run a named shader stack")
    p.add_argument("name", help="Stack name (e.g., crt_mosaic, terrain_scan)")
    p.add_argument("src", nargs="?", type=str, default=None,
                   help="Source footage (omit for generator)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--duration", type=float, default=10.0,
                   help="Clip duration (seconds)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--pack", action="append", dest="packs",
                   help="Restrict pack search (repeatable)")
    p.add_argument("-o", "--output", type=Path, default=None)
    p.set_defaults(func=_handle_stack)


def _handle_stack(args):
    import random as _random
    from pipeline.config import Config
    from pipeline.recipe import (
        load_boutique_stacks, _resolve_shader_params,
        ShaderStep, NormalizeStep, Lane, BrainWipeRecipe,
        FootageSource, GeneratorSource,
    )
    from pipeline.flows.brain_wipe import brain_wipe

    cfg = Config(packs=getattr(args, "packs", None))
    cfg.ensure_dirs()

    # Load all stacks, optionally filtered by pack
    all_stacks = load_boutique_stacks()
    if cfg.packs:
        all_stacks = [s for s in all_stacks if s[3].parent.name in cfg.packs]

    # Find the named stack
    matches = [s for s in all_stacks if s[0] == args.name]
    if not matches:
        available = sorted(set(s[0] for s in all_stacks))
        print(f"Error: stack '{args.name}' not found")
        print(f"Available ({len(available)}): {', '.join(available)}")
        sys.exit(1)

    name, shader_names, params_spec, shader_base = matches[0]
    rng = _random.Random(args.seed)

    # Build ShaderStep
    shader_paths = [shader_base / f"{s}.fs" for s in shader_names]
    param_overrides = _resolve_shader_params(rng, params_spec) or None
    steps = [
        ShaderStep(shader_paths=shader_paths, param_overrides=param_overrides),
        NormalizeStep(),
    ]

    # Source
    dur = args.duration
    if args.src:
        source = FootageSource(
            path=Path(args.src), method="random",
            min_dur=dur, max_dur=dur,
        )
    else:
        source = GeneratorSource(min_dur=dur, max_dur=dur, n_warps=1)

    lane = Lane(source=source, n_segments=1, recipe=steps, sequencing="concat")
    recipe = BrainWipeRecipe(
        lanes=[lane],
        width=args.width, height=args.height,
        seed=args.seed,
    )

    chain = " → ".join(shader_names)
    pack_name = shader_base.parent.name
    print(f"Stack: {name} ({pack_name})")
    print(f"  {chain}")
    if param_overrides:
        for shader_stem, params in param_overrides.items():
            vals = ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                            for k, v in params.items())
            print(f"  {shader_stem}: {vals}")

    brain_wipe(recipe, output=args.output, cfg=cfg)


# ── vf pack ──────────────────────────────────────────────────────────────────

def _add_pack_parsers(sub):
    p_pack = sub.add_parser("pack", help="Shader pack management")
    pack_sub = p_pack.add_subparsers(dest="pack_command")

    p_list = pack_sub.add_parser("list", help="List installed packs")
    p_list.add_argument("-v", "--verbose", action="store_true",
                        help="Show stack names")

    p_info = pack_sub.add_parser("info", help="Inspect a pack")
    p_info.add_argument("name", help="Pack name")
    p_info.add_argument("-v", "--verbose", action="store_true",
                        help="Show stack shader chains and params")

    p_create = pack_sub.add_parser("create", help="Create pack from shader folder")
    p_create.add_argument("name", help="Pack name")
    p_create.add_argument("shader_source", type=Path,
                          help="Folder containing .fs shader files")
    p_create.add_argument("-n", "--n-stacks", type=int, default=None)
    p_create.add_argument("--seed", type=int, default=42)

    p_stacks = pack_sub.add_parser("stacks", help="Regenerate stacks.yaml")
    p_stacks.add_argument("pack_dir", type=Path,
                          help="Pack directory (e.g., packs/my_effects/)")
    p_stacks.add_argument("-n", "--n-stacks", type=int, default=None)
    p_stacks.add_argument("--seed", type=int, default=42)
    p_stacks.add_argument("-o", "--output", type=Path, default=None)

    p_evolve = pack_sub.add_parser("evolve",
                                    help="Evolve diverse stacks via random search")
    p_evolve.add_argument("pack_dir", type=Path,
                           help="Pack directory (e.g., packs/my_effects/)")
    p_evolve.add_argument("--candidates", type=int, default=2000,
                           help="Number of random candidates to evaluate")
    p_evolve.add_argument("-n", "--n-stacks", type=int, default=None,
                           help="Number of stacks to output (default: auto)")
    p_evolve.add_argument("--diversity", type=float, default=1.0,
                           help="Diversity weight λ (higher = more spread)")
    p_evolve.add_argument("--min-fitness", type=float, default=0.5,
                           help="Minimum fitness threshold")
    p_evolve.add_argument("--seed", type=int, default=42)
    p_evolve.add_argument("-o", "--output", type=Path, default=None)

    p_pack.set_defaults(func=_handle_pack)


def _discover_packs() -> list[Path]:
    """Return sorted list of pack directories under packs/."""
    packs_dir = Path(__file__).resolve().parent.parent / "packs"
    if not packs_dir.is_dir():
        return []
    return sorted(d for d in packs_dir.iterdir()
                  if d.is_dir() and (d / "shaders").is_dir())


def _pack_summary(pack_dir: Path) -> dict:
    """Load shader and stack info for a single pack."""
    from pipeline.isf import load_shader_dir

    shaders_dir = pack_dir / "shaders"
    shaders = load_shader_dir(shaders_dir) if shaders_dir.is_dir() else {}

    processors = {k: v for k, v in shaders.items() if v.image_inputs}
    generators = {k: v for k, v in shaders.items() if not v.image_inputs}

    stacks = []
    stacks_file = pack_dir / "stacks.yaml"
    if stacks_file.is_file():
        import yaml
        with open(stacks_file) as f:
            data = yaml.safe_load(f) or {}
        stacks_section = data.get("stacks", data)
        if isinstance(stacks_section, dict):
            for name, spec in stacks_section.items():
                shader_names = spec.get("shaders", []) if isinstance(spec, dict) else []
                params = spec.get("shader_params", {}) if isinstance(spec, dict) else {}
                stacks.append((name, shader_names, params))

    return {
        "name": pack_dir.name,
        "path": pack_dir,
        "shaders": shaders,
        "processors": processors,
        "generators": generators,
        "stacks": stacks,
    }


def _handle_pack_list(verbose: bool):
    packs = _discover_packs()
    if not packs:
        print("No packs installed.")
        return

    # Compute column width for alignment
    max_name = max(len(d.name) for d in packs)

    for pack_dir in packs:
        info = _pack_summary(pack_dir)
        n_proc = len(info["processors"])
        n_gen = len(info["generators"])
        n_stacks = len(info["stacks"])
        name = info["name"].ljust(max_name)
        print(f"  {name}  {n_proc + n_gen:3d} shaders ({n_proc} proc, {n_gen} gen)"
              f"   {n_stacks:2d} stacks")

        if verbose and info["stacks"]:
            names = [s[0] for s in info["stacks"]]
            print(f"  {' ' * max_name}  {', '.join(names)}")


def _handle_pack_info(name: str, verbose: bool):
    packs_dir = Path(__file__).resolve().parent.parent / "packs"
    pack_dir = packs_dir / name
    if not pack_dir.is_dir():
        print(f"Error: pack '{name}' not found in {packs_dir}")
        sys.exit(1)

    info = _pack_summary(pack_dir)
    n_proc = len(info["processors"])
    n_gen = len(info["generators"])
    n_stacks = len(info["stacks"])

    print(f"{name} — {n_proc + n_gen} shaders, {n_stacks} stacks\n")

    # Group processors by first category
    if info["processors"]:
        print(f"Processors ({n_proc}):")
        by_cat: dict[str, list[str]] = {}
        for stem, shader in sorted(info["processors"].items()):
            cat = shader.categories[0] if shader.categories else "Uncategorized"
            by_cat.setdefault(cat, []).append(stem)
        for cat in sorted(by_cat):
            stems = sorted(by_cat[cat])
            print(f"  {cat} ({len(stems)}): {', '.join(stems)}")

    # Generators
    if info["generators"]:
        print(f"\nGenerators ({n_gen}):")
        print(f"  {', '.join(sorted(info['generators']))}")

    # Stacks
    if info["stacks"]:
        if n_stacks > 0:
            avg = sum(len(s[1]) for s in info["stacks"]) / n_stacks
            print(f"\nStacks ({n_stacks}, avg {avg:.1f} shaders):")
        for sname, shader_names, params in info["stacks"]:
            chain = " → ".join(shader_names)
            if verbose:
                print(f"  {sname}:")
                print(f"    {chain}")
                if params:
                    for shader_stem, pspec in params.items():
                        parts = []
                        for pk, pv in pspec.items():
                            if isinstance(pv, list) and len(pv) == 2:
                                parts.append(f"{pk}=[{pv[0]}, {pv[1]}]")
                            elif isinstance(pv, dict) and "choice" in pv:
                                parts.append(f"{pk}={{choice: {pv['choice']}}}")
                            else:
                                parts.append(f"{pk}={pv}")
                        print(f"    {shader_stem}: {', '.join(parts)}")
            else:
                print(f"  {sname} ({len(shader_names)}): {chain}")


def _handle_pack(args):
    cmd = getattr(args, "pack_command", None)
    if not cmd:
        print("Usage: vf pack {list,info,create,stacks,evolve}")
        sys.exit(1)

    if cmd == "list":
        _handle_pack_list(verbose=args.verbose)
        return

    if cmd == "info":
        _handle_pack_info(args.name, verbose=args.verbose)
        return

    if cmd == "create":
        from scripts.create_pack import create_pack
        if not args.shader_source.is_dir():
            print(f"Error: {args.shader_source} is not a directory")
            sys.exit(1)
        create_pack(args.name, args.shader_source,
                    n_stacks=args.n_stacks, seed=args.seed)

    elif cmd == "stacks":
        from scripts.generate_stacks import (
            validate_shaders, generate_stacks, write_stacks_yaml,
        )
        pack_dir = args.pack_dir.resolve()
        shaders_dir = pack_dir / "shaders"
        if not shaders_dir.is_dir():
            print(f"Error: {shaders_dir} not found")
            sys.exit(1)

        processors, generators, failures = validate_shaders(shaders_dir)
        print(f"{len(processors)} processors, {len(generators)} generators")
        if failures:
            print(f"{len(failures)} failures")

        if not processors:
            print("No processor shaders — nothing to generate.")
            sys.exit(1)

        stacks = generate_stacks(
            processors, generators,
            n_stacks=args.n_stacks, seed=args.seed,
        )

        out_path = args.output or (pack_dir / "stacks.yaml")
        write_stacks_yaml(
            stacks, out_path,
            processors=processors,
            generators=generators,
            failures=failures,
        )
        print(f"Wrote {len(stacks)} stacks to {out_path}")

    elif cmd == "evolve":
        from pipeline.flows.evolve_stacks import evolve_stacks
        pack_dir = args.pack_dir.resolve()
        if not (pack_dir / "shaders").is_dir():
            print(f"Error: {pack_dir / 'shaders'} not found")
            sys.exit(1)
        evolve_stacks(
            pack_dir=pack_dir,
            n_candidates=args.candidates,
            n_output=args.n_stacks,
            diversity_weight=args.diversity,
            min_fitness=args.min_fitness,
            seed=args.seed,
            output=args.output,
        )


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="vf",
        description="Video Flows — automated video processing pipeline",
    )
    sub = parser.add_subparsers(dest="command")

    _add_reel_parsers(sub)
    _add_show_parser(sub)
    _add_stack_parser(sub)
    _add_pack_parsers(sub)

    # Handle bare "vf" with no args
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    # Handle "vf reel ..." with no subcommand → default to "run"
    if len(sys.argv) >= 2 and sys.argv[1] == "reel":
        reel_commands = {"run", "plan", "render", "batch"}
        if len(sys.argv) < 3 or sys.argv[2] not in reel_commands:
            sys.argv.insert(2, "run")

    args = parser.parse_args()

    func = getattr(args, "func", None)
    if not func:
        parser.print_help()
        sys.exit(0)

    func(args)


if __name__ == "__main__":
    main()
