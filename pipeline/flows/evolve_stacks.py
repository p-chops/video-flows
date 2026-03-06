"""
Prefect flow for evolving shader stacks via genetic algorithm.

Usage:
    vf pack evolve packs/starter/ --generations 20 --population 50 --seed 42
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

from prefect import flow, get_run_logger

from pipeline.config import Config
from pipeline.evolve import (
    EvolutionConfig,
    evolve,
    genomes_to_stacks_yaml,
)


def _log_progress(gen: int, best_fitness: float, population: list) -> None:
    """Called after each generation — logs to Prefect."""
    try:
        logger = get_run_logger()
    except Exception:
        import logging
        logger = logging.getLogger(__name__)

    avg = sum(g.fitness for g in population) / len(population)
    best = max(population, key=lambda g: g.fitness)
    chain = " → ".join(g.shader_stem for g in best.genes)
    logger.info(
        f"gen {gen:3d} | best={best_fitness:.4f}  avg={avg:.4f}  | {chain}"
    )


@flow(name="evolve-stacks")
def evolve_stacks(
    pack_dir: Path,
    generations: int = 20,
    population_size: int = 50,
    n_output: Optional[int] = None,
    seed: int = 42,
    output: Optional[Path] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """Evolve shader stacks using a genetic algorithm."""
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

    config = EvolutionConfig(
        population_size=population_size,
        generations=generations,
        seed=seed,
    )

    logger.info(
        f"evolving: {config.population_size} population × "
        f"{config.generations} generations"
    )
    final_population = evolve(
        processors, generators, config,
        progress_callback=_log_progress,
    )

    # Determine output count
    if n_output is None:
        n_output = max(4, round(2.5 * math.sqrt(len(processors))))

    stacks = genomes_to_stacks_yaml(final_population, n_stacks=n_output)

    out_path = output or (pack_dir / "stacks.yaml")
    write_stacks_yaml(
        stacks, out_path,
        processors=processors,
        generators=generators,
        failures=failures,
    )

    best = final_population[0] if final_population else None
    if best:
        logger.info(
            f"best fitness: {best.fitness:.4f} — "
            + " → ".join(g.shader_stem for g in best.genes)
        )
    logger.info(f"wrote {len(stacks)} stacks to {out_path}")

    return out_path
