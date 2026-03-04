#!/usr/bin/env python3
"""
Create a shader pack from a folder of ISF shaders.

Takes a folder of .fs files, validates them, generates stacks.yaml,
and installs everything into the packs/ directory structure.

Usage:
    python scripts/create_pack.py my_effects ~/Downloads/cool_shaders/
    python scripts/create_pack.py my_effects ~/Downloads/cool_shaders/ --seed 99
    python scripts/create_pack.py my_effects ~/Downloads/cool_shaders/ --n-stacks 15
    python scripts/create_pack.py my_effects packs/my_effects/shaders/  # already in place
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Add project root to path
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from scripts.generate_stacks import (
    generate_stacks,
    validate_shaders,
    write_stacks_yaml,
)


def create_pack(
    pack_name: str,
    shader_source: Path,
    *,
    n_stacks: int | None = None,
    seed: int = 42,
) -> Path:
    """Create a shader pack from a folder of .fs files.

    Returns the pack directory path.
    """
    packs_dir = _root / "packs"
    pack_dir = packs_dir / pack_name
    shaders_dir = pack_dir / "shaders"
    stacks_path = pack_dir / "stacks.yaml"

    shader_source = shader_source.resolve()

    # ── Copy shaders if source is outside the pack ──────────────────────
    if shaders_dir.resolve() != shader_source.resolve():
        fs_files = sorted(shader_source.glob("*.fs"))
        if not fs_files:
            print(f"Error: no .fs files found in {shader_source}")
            sys.exit(1)

        shaders_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        skipped = 0
        for fs in fs_files:
            dst = shaders_dir / fs.name
            if dst.exists():
                # Don't overwrite — user may have edited
                skipped += 1
                continue
            shutil.copy2(fs, dst)
            copied += 1

        print(f"Copied {copied} shaders to {shaders_dir}")
        if skipped:
            print(f"  ({skipped} already existed, skipped)")
    else:
        print(f"Shaders already in place at {shaders_dir}")

    # ── Validate ────────────────────────────────────────────────────────
    print(f"\nValidating shaders in {shaders_dir} ...")
    processors, generators, failures = validate_shaders(shaders_dir)

    print(f"  {len(processors)} processors (have inputImage)")
    print(f"  {len(generators)} generators (no inputImage)")

    if failures:
        print(f"\n  {len(failures)} shaders failed validation:")
        for name, reason in failures:
            print(f"    {name}: {reason}")

        # Remove failed shaders from pack
        for name, _ in failures:
            bad = shaders_dir / name
            if bad.exists():
                bad.unlink()
                print(f"    removed {name}")
        print()

    if not processors and not generators:
        print("Error: no valid shaders. Pack not created.")
        # Clean up empty dir
        if shaders_dir.exists() and not any(shaders_dir.iterdir()):
            shaders_dir.rmdir()
            if pack_dir.exists() and not any(pack_dir.iterdir()):
                pack_dir.rmdir()
        sys.exit(1)

    # ── Generate stacks ─────────────────────────────────────────────────
    if not processors:
        print("No processor shaders — skipping stacks.yaml (generator-only pack).")
        print(f"\nPack created at {pack_dir}")
        return pack_dir

    stacks = generate_stacks(
        processors, generators,
        n_stacks=n_stacks,
        seed=seed,
    )

    write_stacks_yaml(
        stacks, stacks_path,
        processors=processors,
        generators=generators,
        failures=failures,
    )

    # ── Summary ─────────────────────────────────────────────────────────
    all_stems = [s for stack in stacks.values() for s in stack["shaders"]]
    unique = set(all_stems)

    print(f"\nPack created: packs/{pack_name}/")
    print(f"  shaders/    {len(processors) + len(generators)} shaders"
          f" ({len(processors)} processors, {len(generators)} generators)")
    print(f"  stacks.yaml {len(stacks)} stacks"
          f" (avg {len(all_stems)/len(stacks):.1f} shaders each)")
    print(f"  coverage    {len(unique)}/{len(processors)} processors used")

    if failures:
        print(f"  removed     {len(failures)} shaders that failed validation")

    print(f"\nReady to use:")
    print(f"  python -m pipeline.flows.show_reel run -n 8 --pack {pack_name} --seed 42")

    return pack_dir


def main():
    parser = argparse.ArgumentParser(
        description="Create a shader pack from a folder of ISF shaders",
    )
    parser.add_argument(
        "pack_name",
        help="Name for the pack (e.g., 'my_effects')",
    )
    parser.add_argument(
        "shader_source",
        type=Path,
        help="Folder containing .fs shader files",
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

    args = parser.parse_args()

    if not args.shader_source.is_dir():
        print(f"Error: {args.shader_source} is not a directory")
        sys.exit(1)

    # Validate pack name
    if "/" in args.pack_name or "\\" in args.pack_name:
        print("Error: pack name cannot contain path separators")
        sys.exit(1)

    create_pack(
        args.pack_name,
        args.shader_source,
        n_stacks=args.n_stacks,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
