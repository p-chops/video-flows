"""Time effect registry — single source of truth for dispatch, serialization, and random generation."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence


@dataclass
class TimeEffectEntry:
    """Metadata for one registered time effect."""
    name: str
    step_class: Optional[type] = None
    process_fn: Optional[Callable] = None
    task_fn: Optional[Callable] = None
    seedless: bool = False
    in_random_pool: bool = True
    random_fn: Optional[Callable] = None


_REGISTRY: dict[str, TimeEffectEntry] = {}
# Cache for isinstance lookups — invalidated on registration
_CLASS_CACHE: dict[type, TimeEffectEntry] = {}


def register_step(
    name: str,
    cls: type,
    *,
    seedless: bool = False,
    in_pool: bool = True,
    random_fn: Optional[Callable] = None,
) -> None:
    """Register a step dataclass (called by recipe.py at import time)."""
    if name in _REGISTRY:
        entry = _REGISTRY[name]
        entry.step_class = cls
        entry.seedless = seedless
        entry.in_random_pool = in_pool
        entry.random_fn = random_fn
    else:
        _REGISTRY[name] = TimeEffectEntry(
            name=name, step_class=cls, seedless=seedless,
            in_random_pool=in_pool, random_fn=random_fn,
        )
    _CLASS_CACHE.clear()


def register_process(
    name: str,
    process_fn: Callable,
    task_fn: Optional[Callable] = None,
) -> None:
    """Register a process function and optional task wrapper (called by time.py at import time)."""
    if name in _REGISTRY:
        entry = _REGISTRY[name]
        entry.process_fn = process_fn
        entry.task_fn = task_fn
    else:
        _REGISTRY[name] = TimeEffectEntry(
            name=name, process_fn=process_fn, task_fn=task_fn,
        )
    _CLASS_CACHE.clear()


def _ensure_class_cache() -> None:
    if not _CLASS_CACHE:
        for entry in _REGISTRY.values():
            if entry.step_class is not None:
                _CLASS_CACHE[entry.step_class] = entry


def get_entry(name: str) -> Optional[TimeEffectEntry]:
    """Look up entry by serialization name."""
    return _REGISTRY.get(name)


def get_entry_for_step(step: Any) -> Optional[TimeEffectEntry]:
    """Look up entry by step instance type."""
    _ensure_class_cache()
    return _CLASS_CACHE.get(type(step))


def all_entries() -> list[TimeEffectEntry]:
    """All registered time effects."""
    return list(_REGISTRY.values())


def time_step_types() -> tuple[type, ...]:
    """Tuple of all registered step classes (replaces _TIME_STEP_TYPES)."""
    return tuple(e.step_class for e in _REGISTRY.values() if e.step_class is not None)


def seedless_time_steps() -> tuple[type, ...]:
    """Tuple of seedless step classes (replaces _SEEDLESS_TIME_STEPS)."""
    return tuple(
        e.step_class for e in _REGISTRY.values()
        if e.step_class is not None and e.seedless
    )


def pool_entries() -> list[TimeEffectEntry]:
    """Entries eligible for random recipe generation."""
    return [e for e in _REGISTRY.values() if e.in_random_pool and e.random_fn is not None]
