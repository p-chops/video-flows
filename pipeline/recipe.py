"""
Recipe data model for the brain wipe meta-flow.

A BrainWipeRecipe declaratively describes a full render:
  - Lanes: parallel processing streams, each with a source, segment count,
    per-segment recipe (list of Steps), and sequencing mode.
  - Compositing: optional spec for combining lane outputs.
  - Post-processing: steps applied after compositing.

Recipes are Python dataclasses — construct them directly or use the
builder functions at the bottom of this module for common patterns.

Extending with new step types (from labs):
  1. Add a dataclass here.
  2. Add it to the Step union type alias.
  3. Add a case branch in the executor (_execute_step in brain_wipe.py).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


# ─── Source specs ─────────────────────────────────────────────────────────────

@dataclass
class FootageSource:
    """Pull segments from existing video footage."""
    path: Path
    method: str = "random"        # "random" | "scene"
    min_dur: float = 5.0          # random mode: min segment duration
    max_dur: float = 30.0         # random mode: max segment duration

@dataclass
class GeneratorSource:
    """Synthesize segments via generator shaders (no source footage)."""
    duration: float = 20.0
    n_warps: int = 0              # warp shaders chained after generator
    warp_categories: list[str] = field(
        default_factory=lambda: ["Warp", "Brain Wipe"],
    )

@dataclass
class StaticSource:
    """Generate random noise video."""
    duration: float = 10.0

@dataclass
class SolidSource:
    """Generate a solid-colour clip."""
    duration: float = 10.0
    color: tuple[int, int, int] = (0, 0, 0)

SourceSpec = FootageSource | GeneratorSource | StaticSource | SolidSource


# ─── Processing steps ────────────────────────────────────────────────────────

@dataclass
class CrushStep:
    """Bitrate-crush via fixed-QP encoding."""
    crush: float = 0.95
    codec: str = "libx264"
    downscale: float = 1.0

@dataclass
class ShaderStep:
    """Apply a shader stack (explicit or randomly selected)."""
    shader_paths: Optional[list[Path]] = None   # explicit list (skip random)
    n_shaders: int = 3                           # random pick count
    categories: Optional[list[str]] = None       # filter by ISF category
    shader_dir: Optional[Path] = None            # override default shader dir

@dataclass
class NormalizeStep:
    """Percentile-based level stretch."""
    black_point: float = 0.01
    white_point: float = 0.99

# Future step types from labs — add dataclasses here, add to Step union below.

Step = CrushStep | ShaderStep | NormalizeStep


# ─── Lanes ────────────────────────────────────────────────────────────────────

@dataclass
class Lane:
    """One parallel processing stream: source → per-segment recipe → sequence."""
    source: SourceSpec
    n_segments: int = 8
    recipe: list[Step] = field(default_factory=list)
    sequencing: str = "shuffle"    # "shuffle" | "concat"
    static_gap: float = 0.0       # >0 = interleave static between segments (sec)


# ─── Compositing specs ────────────────────────────────────────────────────────

@dataclass
class BlendComposite:
    """Blend two lane outputs with a blend mode."""
    mode: str = "screen"
    opacity: float = 0.5

@dataclass
class MaskedComposite:
    """Composite lane outputs via an auto-generated mask."""
    mask_type: str = "luma"       # luma | edge | motion | gradient
    mask_params: dict[str, Any] = field(default_factory=dict)

@dataclass
class RandomComposite:
    """Random compositing operations (compositing-lab style)."""
    n_ops: int = 3

CompositeSpec = BlendComposite | MaskedComposite | RandomComposite


# ─── Top-level recipe ─────────────────────────────────────────────────────────

@dataclass
class BrainWipeRecipe:
    """Full render specification for the brain wipe meta-flow."""
    lanes: list[Lane]
    composite: Optional[CompositeSpec] = None   # None = separate output per lane
    post: list[Step] = field(default_factory=list)
    width: int = 1920
    height: int = 1080
    fps: float = 30.0
    seed: Optional[int] = None
    shader_dir: Optional[Path] = None           # default shader library
    brain_wipe_dir: Path = field(
        default_factory=lambda: Path("brain-wipe-shaders"),
    )


# ─── Recipe utilities ─────────────────────────────────────────────────────────

def _step_label(step: Step) -> str:
    """Short human-readable label for a step."""
    match step:
        case CrushStep(crush=c, codec=codec, downscale=ds):
            parts = [f"crush {c:.2f} ({codec})"]
            if ds > 1.0:
                parts.append(f"↓{ds:.0f}x")
            return " ".join(parts)
        case ShaderStep(shader_paths=paths, n_shaders=n, categories=cats):
            if paths:
                names = [p.stem for p in paths]
                return f"shaders [{', '.join(names)}]"
            cat_tag = f" cats={cats}" if cats else ""
            return f"shaders ×{n}{cat_tag}"
        case NormalizeStep(black_point=bp, white_point=wp):
            return f"normalize [{bp:.2f}–{wp:.2f}]"
        case _:
            return type(step).__name__


def _source_label(source: SourceSpec) -> str:
    """Short human-readable label for a source spec."""
    match source:
        case FootageSource(path=p, method=m):
            return f"footage({p.name}, {m})"
        case GeneratorSource(duration=d, n_warps=nw):
            return f"generator({d:.0f}s, {nw} warps)"
        case StaticSource(duration=d):
            return f"static({d:.0f}s)"
        case SolidSource(duration=d, color=c):
            return f"solid({d:.0f}s, rgb{c})"
        case _:
            return type(source).__name__


def print_recipe(recipe: BrainWipeRecipe) -> None:
    """Pretty-print a recipe to stdout."""
    print(f"═══ Brain Wipe Recipe  (seed={recipe.seed}) ═══")
    print(f"    {recipe.width}×{recipe.height} @ {recipe.fps:.0f}fps")
    if recipe.shader_dir:
        print(f"    shader_dir: {recipe.shader_dir}")
    print()

    for i, lane in enumerate(recipe.lanes):
        print(f"  ─── Lane {i} ───")
        print(f"    source:     {_source_label(lane.source)}")
        print(f"    segments:   {lane.n_segments}")
        print(f"    sequencing: {lane.sequencing}", end="")
        if lane.static_gap > 0:
            print(f" (static gap {lane.static_gap:.1f}s)", end="")
        print()
        if lane.recipe:
            print(f"    recipe:")
            for j, step in enumerate(lane.recipe):
                print(f"      {j+1}. {_step_label(step)}")
        else:
            print(f"    recipe:     (none)")
        print()

    if recipe.composite is not None:
        match recipe.composite:
            case BlendComposite(mode=m, opacity=o):
                print(f"  composite: blend({m}, opacity={o:.2f})")
            case MaskedComposite(mask_type=mt):
                print(f"  composite: masked({mt})")
            case RandomComposite(n_ops=n):
                print(f"  composite: random({n} ops)")
        print()

    if recipe.post:
        print(f"  post-processing:")
        for j, step in enumerate(recipe.post):
            print(f"    {j+1}. {_step_label(step)}")
        print()


def _recipe_to_hashable(recipe: BrainWipeRecipe) -> str:
    """Serialise the recipe structure to a stable string for hashing."""
    def _step_dict(s: Step) -> dict:
        match s:
            case CrushStep():
                return {"type": "crush", "crush": s.crush,
                        "codec": s.codec, "downscale": s.downscale}
            case ShaderStep():
                return {"type": "shader",
                        "paths": [str(p) for p in s.shader_paths] if s.shader_paths else None,
                        "n": s.n_shaders, "cats": s.categories}
            case NormalizeStep():
                return {"type": "normalize", "bp": s.black_point, "wp": s.white_point}
            case _:
                return {"type": type(s).__name__}

    def _source_dict(src: SourceSpec) -> dict:
        match src:
            case FootageSource():
                return {"type": "footage", "path": str(src.path), "method": src.method}
            case GeneratorSource():
                return {"type": "generator", "dur": src.duration, "warps": src.n_warps}
            case StaticSource():
                return {"type": "static", "dur": src.duration}
            case SolidSource():
                return {"type": "solid", "dur": src.duration, "color": list(src.color)}
            case _:
                return {"type": type(src).__name__}

    data = {
        "lanes": [
            {
                "source": _source_dict(l.source),
                "n_segments": l.n_segments,
                "recipe": [_step_dict(s) for s in l.recipe],
                "sequencing": l.sequencing,
                "static_gap": l.static_gap,
            }
            for l in recipe.lanes
        ],
        "composite": type(recipe.composite).__name__ if recipe.composite else None,
        "post": [_step_dict(s) for s in recipe.post],
        "seed": recipe.seed,
    }
    return json.dumps(data, sort_keys=True)


def hash_recipe(recipe: BrainWipeRecipe) -> str:
    """Return an 8-char hex hash of the recipe for output naming."""
    raw = _recipe_to_hashable(recipe)
    return hashlib.sha1(raw.encode()).hexdigest()[:8]


# ─── Recipe builders ──────────────────────────────────────────────────────────

def crush_sandwich_recipe(
    src: Path,
    *,
    n_segments: int = 8,
    crush: float = 0.95,
    n_shaders: int = 3,
    sequencing: str = "shuffle",
    normalize: bool = True,
    seed: Optional[int] = None,
) -> BrainWipeRecipe:
    """Crush → shaders → crush → shaders → normalize."""
    steps: list[Step] = [
        CrushStep(crush=crush),
        ShaderStep(n_shaders=n_shaders),
        CrushStep(crush=crush),
        ShaderStep(n_shaders=n_shaders),
    ]
    if normalize:
        steps.append(NormalizeStep())

    return BrainWipeRecipe(
        lanes=[Lane(
            source=FootageSource(src),
            n_segments=n_segments,
            recipe=steps,
            sequencing=sequencing,
        )],
        seed=seed,
    )


def stooges_recipe(
    src: Path,
    *,
    segment_counts: list[int] | int = 8,
    crush: float = 0.95,
    n_shaders: int = 3,
    static_gap: float = 0.3,
    min_dur: float = 1.0,
    max_dur: float = 30.0,
    seed: Optional[int] = None,
) -> BrainWipeRecipe:
    """Multi-channel CRT content: crush sandwich per segment, interleaved with static."""
    if isinstance(segment_counts, int):
        segment_counts = [segment_counts]

    steps: list[Step] = [
        CrushStep(crush=crush),
        ShaderStep(n_shaders=n_shaders),
        CrushStep(crush=crush),
        ShaderStep(n_shaders=n_shaders),
        NormalizeStep(),
    ]

    return BrainWipeRecipe(
        lanes=[
            Lane(
                source=FootageSource(src, min_dur=min_dur, max_dur=max_dur),
                n_segments=count,
                recipe=steps,
                sequencing="shuffle",
                static_gap=static_gap,
            )
            for count in segment_counts
        ],
        seed=seed,
    )


def generator_render_recipe(
    *,
    n_segments: int = 12,
    segment_dur: float = 20.0,
    min_warps: int = 1,
    max_warps: int = 4,
    normalize: bool = True,
    sequencing: str = "shuffle",
    seed: Optional[int] = None,
    brain_wipe_dir: Optional[Path] = None,
) -> BrainWipeRecipe:
    """Generator shaders + warp chain → shuffle → concat. No source footage."""
    # n_warps will be randomised per-segment between min_warps and max_warps
    # by the executor; we store max here as the lane-level default.
    steps: list[Step] = []
    if normalize:
        steps.append(NormalizeStep())

    return BrainWipeRecipe(
        lanes=[Lane(
            source=GeneratorSource(
                duration=segment_dur,
                n_warps=max_warps,
                warp_categories=["Warp", "Brain Wipe"],
            ),
            n_segments=n_segments,
            recipe=steps,
            sequencing=sequencing,
        )],
        seed=seed,
        brain_wipe_dir=brain_wipe_dir or Path("brain-wipe-shaders"),
    )


def composite_recipe(
    src: Path,
    *,
    n_segments: int = 8,
    recipe_a: Optional[list[Step]] = None,
    recipe_b: Optional[list[Step]] = None,
    composite: Optional[CompositeSpec] = None,
    sequencing_a: str = "concat",
    sequencing_b: str = "shuffle",
    post: Optional[list[Step]] = None,
    seed: Optional[int] = None,
) -> BrainWipeRecipe:
    """Two lanes from same source, composited together."""
    if recipe_a is None:
        recipe_a = [CrushStep(), ShaderStep(), CrushStep(), ShaderStep(), NormalizeStep()]
    if recipe_b is None:
        recipe_b = [CrushStep(), ShaderStep(), CrushStep(), ShaderStep(), NormalizeStep()]
    if composite is None:
        composite = MaskedComposite(mask_type="motion")

    return BrainWipeRecipe(
        lanes=[
            Lane(
                source=FootageSource(src, method="scene"),
                n_segments=n_segments,
                recipe=recipe_a,
                sequencing=sequencing_a,
            ),
            Lane(
                source=FootageSource(src, method="scene"),
                n_segments=n_segments,
                recipe=recipe_b,
                sequencing=sequencing_b,
            ),
        ],
        composite=composite,
        post=post or [NormalizeStep()],
        seed=seed,
    )
