"""
Prefect flow for shader stack evolution.

Discovers diverse, high-quality stacks via random search + greedy
diversity-weighted selection.

Usage:
    vf pack evolve packs/starter/ --candidates 2000 -n 15 --seed 42
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

from prefect import flow, get_run_logger

from pipeline.config import Config
from pipeline.evolve import (
    EvolveConfig,
    evolve,
    genomes_to_stacks_yaml,
)


def _log_progress(gen: int, best_fitness: float, population: list, *, msg: str = "") -> None:
    """Called during evolution — logs to Prefect."""
    try:
        logger = get_run_logger()
    except Exception:
        import logging
        logger = logging.getLogger(__name__)

    if msg:
        logger.info(msg)
        return

    avg = sum(g.fitness for g in population) / len(population)
    best = max(population, key=lambda g: g.fitness)
    chain = " → ".join(g.shader_stem for g in best.genes)
    logger.info(
        f"gen {gen:3d} | best={best_fitness:.4f}  avg={avg:.4f}  | {chain}"
    )


@flow(name="evolve-stacks")
def evolve_stacks(
    pack_dir: Path,
    n_candidates: int = 2000,
    n_output: Optional[int] = None,
    diversity_weight: float = 1.0,
    min_fitness: float = 0.5,
    seed: int = 42,
    output: Optional[Path] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """Discover diverse shader stacks via random search."""
    from scripts.generate_stacks import (
        validate_shaders,
        write_stacks_yaml,
    )

    logger = get_run_logger()
    pack_dir = pack_dir.resolve()
    shaders_dir = pack_dir / "shaders"

    logger.info(f"validating shaders in {shaders_dir}")
    processors, generators, failures = validate_shaders(shaders_dir)
    logger.info(
        f"{len(processors)} processors, {len(generators)} generators, "
        f"{len(failures)} failures"
    )

    if not processors:
        logger.error("no processor shaders — cannot evolve")
        return pack_dir

    if n_output is None:
        n_output = max(4, round(2.5 * math.sqrt(len(processors))))

    config = EvolveConfig(
        n_candidates=n_candidates,
        n_output=n_output,
        diversity_weight=diversity_weight,
        min_fitness=min_fitness,
        seed=seed,
    )

    logger.info(
        f"evolving: {config.n_candidates} candidates → "
        f"{config.n_output} stacks (λ={config.diversity_weight})"
    )
    selected = evolve(
        processors, generators, config,
        progress_callback=_log_progress,
    )

    stacks = genomes_to_stacks_yaml(selected, n_stacks=n_output)

    out_path = output or (pack_dir / "stacks.yaml")
    write_stacks_yaml(
        stacks, out_path,
        processors=processors,
        generators=generators,
        failures=failures,
    )

    if selected:
        for i, g in enumerate(selected):
            chain = " → ".join(gene.shader_stem for gene in g.genes)
            logger.info(f"  {i+1:2d}. [{g.fitness:.3f}] {chain}")

    logger.info(f"wrote {len(stacks)} stacks to {out_path}")

    return out_path
