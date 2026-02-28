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
import random as _random_mod
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
    min_dur: float = 15.0
    max_dur: float = 25.0
    n_warps: int = 0              # warp shaders chained after generator
    warp_categories: list[str] = field(
        default_factory=lambda: ["Warp", "Brain Wipe"],
    )

@dataclass
class StaticSource:
    """Generate random noise video."""
    min_dur: float = 8.0
    max_dur: float = 12.0

@dataclass
class SolidSource:
    """Generate a solid-colour clip."""
    min_dur: float = 8.0
    max_dur: float = 12.0
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

@dataclass
class ScrubStep:
    """Temporal scrub — smooth random playhead wandering."""
    smoothness: float = 2.0
    intensity: float = 0.5

@dataclass
class DriftStep:
    """Drift loop — short looping window that drifts through the source."""
    loop_dur: float = 0.5
    drift: Optional[float] = None   # None = auto (+-10% of loop length)

@dataclass
class PingPongStep:
    """Ping-pong — forward-backward breathing repetition."""
    window: float = 0.5

@dataclass
class EchoStep:
    """Temporal echo / trails — ghostly motion trails or distinct echoes."""
    delay: float = 0.0    # 0 = motion blur, >0 = distinct echoes
    trail: float = 0.8    # echo strength / feedback

@dataclass
class PatchStep:
    """Temporal patchwork — random rectangular patches from different moments."""
    patch_min: float = 0.05
    patch_max: float = 0.4

Step = CrushStep | ShaderStep | NormalizeStep | ScrubStep | DriftStep | PingPongStep | EchoStep | PatchStep


# ─── Transitions ──────────────────────────────────────────────────────────────

@dataclass
class TransitionSpec:
    """Transition applied between segments during sequencing."""
    type: str = "crossfade"       # crossfade | luma_wipe | whip_pan | static_burst | flash | random
    duration: float = 1.0
    # luma_wipe
    pattern: str = "horizontal"
    softness: float = 0.1
    angle: float = 0.0
    # whip_pan
    direction: str = "left"
    blur_strength: float = 0.5
    # flash
    decay: float = 3.0


# ─── Lanes ────────────────────────────────────────────────────────────────────

@dataclass
class Lane:
    """One parallel processing stream: source → per-segment recipe → sequence."""
    source: SourceSpec
    n_segments: int = 8
    recipe: list[Step] = field(default_factory=list)
    sequencing: str = "shuffle"    # "shuffle" | "concat"
    static_gap: float = 0.0       # >0 = interleave static between segments (sec)
    transition: Optional[TransitionSpec] = None  # transition between segments


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
        case ScrubStep(smoothness=s, intensity=i):
            return f"scrub (smooth={s:.1f}, intensity={i:.2f})"
        case DriftStep(loop_dur=ld, drift=d):
            drift_str = f"{d:.2f}s" if d is not None else "auto"
            return f"drift (loop={ld:.2f}s, drift={drift_str})"
        case PingPongStep(window=w):
            return f"pingpong (window={w:.2f}s)"
        case EchoStep(delay=d, trail=t):
            mode = "blur" if d <= 0 else f"delay={d:.2f}s"
            return f"echo ({mode}, trail={t:.2f})"
        case PatchStep(patch_min=mn, patch_max=mx):
            return f"patch ({mn:.0%}–{mx:.0%})"
        case _:
            return type(step).__name__


def _source_label(source: SourceSpec) -> str:
    """Short human-readable label for a source spec."""
    match source:
        case FootageSource(path=p, method=m):
            return f"footage({p.name}, {m})"
        case GeneratorSource(min_dur=lo, max_dur=hi, n_warps=nw):
            dur = f"{lo:.0f}s" if lo == hi else f"{lo:.0f}–{hi:.0f}s"
            return f"generator({dur}, {nw} warps)"
        case StaticSource(min_dur=lo, max_dur=hi):
            dur = f"{lo:.0f}s" if lo == hi else f"{lo:.0f}–{hi:.0f}s"
            return f"static({dur})"
        case SolidSource(min_dur=lo, max_dur=hi, color=c):
            dur = f"{lo:.0f}s" if lo == hi else f"{lo:.0f}–{hi:.0f}s"
            return f"solid({dur}, rgb{c})"
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
        if lane.transition:
            t = lane.transition
            detail = f"{t.type} ({t.duration:.1f}s)"
            if t.type == "luma_wipe":
                detail += f" pattern={t.pattern} softness={t.softness:.1f}"
            elif t.type == "whip_pan":
                detail += f" dir={t.direction}"
            elif t.type == "flash":
                detail += f" decay={t.decay:.1f}"
            print(f"    transition: {detail}")
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
            case ScrubStep():
                return {"type": "scrub", "smoothness": s.smoothness,
                        "intensity": s.intensity}
            case DriftStep():
                return {"type": "drift", "loop_dur": s.loop_dur, "drift": s.drift}
            case PingPongStep():
                return {"type": "pingpong", "window": s.window}
            case EchoStep():
                return {"type": "echo", "delay": s.delay, "trail": s.trail}
            case PatchStep():
                return {"type": "patch", "patch_min": s.patch_min,
                        "patch_max": s.patch_max}
            case _:
                return {"type": type(s).__name__}

    def _source_dict(src: SourceSpec) -> dict:
        match src:
            case FootageSource():
                return {"type": "footage", "path": str(src.path), "method": src.method}
            case GeneratorSource():
                return {"type": "generator", "min_dur": src.min_dur, "max_dur": src.max_dur, "warps": src.n_warps}
            case StaticSource():
                return {"type": "static", "min_dur": src.min_dur, "max_dur": src.max_dur}
            case SolidSource():
                return {"type": "solid", "min_dur": src.min_dur, "max_dur": src.max_dur, "color": list(src.color)}
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
                "transition": {
                    "type": l.transition.type, "dur": l.transition.duration,
                    "pattern": l.transition.pattern,
                    "direction": l.transition.direction,
                } if l.transition else None,
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


# ─── Serialization ───────────────────────────────────────────────────────────

def _step_to_dict(s: Step) -> dict:
    match s:
        case CrushStep():
            return {"type": "crush", "crush": s.crush,
                    "codec": s.codec, "downscale": s.downscale}
        case ShaderStep():
            return {"type": "shader",
                    "shader_paths": [str(p) for p in s.shader_paths] if s.shader_paths else None,
                    "n_shaders": s.n_shaders,
                    "categories": s.categories,
                    "shader_dir": str(s.shader_dir) if s.shader_dir else None}
        case NormalizeStep():
            return {"type": "normalize", "black_point": s.black_point,
                    "white_point": s.white_point}
        case ScrubStep():
            return {"type": "scrub", "smoothness": s.smoothness,
                    "intensity": s.intensity}
        case DriftStep():
            return {"type": "drift", "loop_dur": s.loop_dur, "drift": s.drift}
        case PingPongStep():
            return {"type": "pingpong", "window": s.window}
        case EchoStep():
            return {"type": "echo", "delay": s.delay, "trail": s.trail}
        case PatchStep():
            return {"type": "patch", "patch_min": s.patch_min,
                    "patch_max": s.patch_max}
        case _:
            raise ValueError(f"Unknown step type: {type(s).__name__}")


def _step_from_dict(d: dict) -> Step:
    t = d["type"]
    if t == "crush":
        return CrushStep(crush=d["crush"], codec=d["codec"],
                         downscale=d.get("downscale", 1.0))
    elif t == "shader":
        return ShaderStep(
            shader_paths=[Path(p) for p in d["shader_paths"]] if d.get("shader_paths") else None,
            n_shaders=d.get("n_shaders", 3),
            categories=d.get("categories"),
            shader_dir=Path(d["shader_dir"]) if d.get("shader_dir") else None,
        )
    elif t == "normalize":
        return NormalizeStep(black_point=d.get("black_point", 0.01),
                             white_point=d.get("white_point", 0.99))
    elif t == "scrub":
        return ScrubStep(smoothness=d["smoothness"], intensity=d["intensity"])
    elif t == "drift":
        return DriftStep(loop_dur=d["loop_dur"], drift=d.get("drift"))
    elif t == "pingpong":
        return PingPongStep(window=d["window"])
    elif t == "echo":
        return EchoStep(delay=d["delay"], trail=d["trail"])
    elif t == "patch":
        return PatchStep(patch_min=d["patch_min"], patch_max=d["patch_max"])
    else:
        raise ValueError(f"Unknown step type: {t}")


def _source_to_dict(src: SourceSpec) -> dict:
    match src:
        case FootageSource():
            return {"type": "footage", "path": str(src.path),
                    "method": src.method,
                    "min_dur": src.min_dur, "max_dur": src.max_dur}
        case GeneratorSource():
            return {"type": "generator",
                    "min_dur": src.min_dur, "max_dur": src.max_dur,
                    "n_warps": src.n_warps,
                    "warp_categories": src.warp_categories}
        case StaticSource():
            return {"type": "static",
                    "min_dur": src.min_dur, "max_dur": src.max_dur}
        case SolidSource():
            return {"type": "solid",
                    "min_dur": src.min_dur, "max_dur": src.max_dur,
                    "color": list(src.color)}
        case _:
            raise ValueError(f"Unknown source type: {type(src).__name__}")


def _source_from_dict(d: dict) -> SourceSpec:
    t = d["type"]
    if t == "footage":
        return FootageSource(
            path=Path(d["path"]), method=d.get("method", "random"),
            min_dur=d.get("min_dur", 5.0), max_dur=d.get("max_dur", 30.0),
        )
    elif t == "generator":
        return GeneratorSource(
            min_dur=d.get("min_dur", 15.0), max_dur=d.get("max_dur", 25.0),
            n_warps=d.get("n_warps", 0),
            warp_categories=d.get("warp_categories", ["Warp", "Brain Wipe"]),
        )
    elif t == "static":
        return StaticSource(
            min_dur=d.get("min_dur", 8.0), max_dur=d.get("max_dur", 12.0),
        )
    elif t == "solid":
        return SolidSource(
            min_dur=d.get("min_dur", 8.0), max_dur=d.get("max_dur", 12.0),
            color=tuple(d.get("color", [0, 0, 0])),
        )
    else:
        raise ValueError(f"Unknown source type: {t}")


def _transition_to_dict(t: TransitionSpec) -> dict:
    return {
        "type": t.type, "duration": t.duration,
        "pattern": t.pattern, "softness": t.softness, "angle": t.angle,
        "direction": t.direction, "blur_strength": t.blur_strength,
        "decay": t.decay,
    }


def _transition_from_dict(d: dict) -> TransitionSpec:
    return TransitionSpec(
        type=d.get("type", "crossfade"),
        duration=d.get("duration", 1.0),
        pattern=d.get("pattern", "horizontal"),
        softness=d.get("softness", 0.1),
        angle=d.get("angle", 0.0),
        direction=d.get("direction", "left"),
        blur_strength=d.get("blur_strength", 0.5),
        decay=d.get("decay", 3.0),
    )


def _composite_to_dict(c: CompositeSpec) -> dict:
    match c:
        case BlendComposite():
            return {"type": "blend", "mode": c.mode, "opacity": c.opacity}
        case MaskedComposite():
            return {"type": "masked", "mask_type": c.mask_type,
                    "mask_params": c.mask_params}
        case RandomComposite():
            return {"type": "random", "n_ops": c.n_ops}
        case _:
            raise ValueError(f"Unknown composite type: {type(c).__name__}")


def _composite_from_dict(d: dict) -> CompositeSpec:
    t = d["type"]
    if t == "blend":
        return BlendComposite(mode=d.get("mode", "screen"),
                               opacity=d.get("opacity", 0.5))
    elif t == "masked":
        return MaskedComposite(mask_type=d.get("mask_type", "luma"),
                                mask_params=d.get("mask_params", {}))
    elif t == "random":
        return RandomComposite(n_ops=d.get("n_ops", 3))
    else:
        raise ValueError(f"Unknown composite type: {t}")


def recipe_to_dict(recipe: BrainWipeRecipe) -> dict:
    """Serialize a recipe to a plain dict (JSON-safe)."""
    return {
        "lanes": [
            {
                "source": _source_to_dict(l.source),
                "n_segments": l.n_segments,
                "recipe": [_step_to_dict(s) for s in l.recipe],
                "sequencing": l.sequencing,
                "static_gap": l.static_gap,
                "transition": _transition_to_dict(l.transition) if l.transition else None,
            }
            for l in recipe.lanes
        ],
        "composite": _composite_to_dict(recipe.composite) if recipe.composite else None,
        "post": [_step_to_dict(s) for s in recipe.post],
        "width": recipe.width,
        "height": recipe.height,
        "fps": recipe.fps,
        "seed": recipe.seed,
        "shader_dir": str(recipe.shader_dir) if recipe.shader_dir else None,
        "brain_wipe_dir": str(recipe.brain_wipe_dir),
    }


def recipe_from_dict(d: dict) -> BrainWipeRecipe:
    """Deserialize a recipe from a plain dict."""
    lanes = []
    for ld in d["lanes"]:
        lanes.append(Lane(
            source=_source_from_dict(ld["source"]),
            n_segments=ld["n_segments"],
            recipe=[_step_from_dict(s) for s in ld["recipe"]],
            sequencing=ld.get("sequencing", "shuffle"),
            static_gap=ld.get("static_gap", 0.0),
            transition=_transition_from_dict(ld["transition"]) if ld.get("transition") else None,
        ))

    composite = _composite_from_dict(d["composite"]) if d.get("composite") else None

    return BrainWipeRecipe(
        lanes=lanes,
        composite=composite,
        post=[_step_from_dict(s) for s in d.get("post", [])],
        width=d.get("width", 1920),
        height=d.get("height", 1080),
        fps=d.get("fps", 30.0),
        seed=d.get("seed"),
        shader_dir=Path(d["shader_dir"]) if d.get("shader_dir") else None,
        brain_wipe_dir=Path(d.get("brain_wipe_dir", "brain-wipe-shaders")),
    )


def save_recipe(recipe: BrainWipeRecipe, path: Path) -> Path:
    """Save a recipe to a JSON file. Returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = recipe_to_dict(recipe)
    path.write_text(json.dumps(data, indent=2) + "\n")
    return path


def load_recipe(path: Path) -> BrainWipeRecipe:
    """Load a recipe from a JSON file."""
    data = json.loads(Path(path).read_text())
    return recipe_from_dict(data)


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
                min_dur=segment_dur, max_dur=segment_dur,
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


def temporal_sandwich_recipe(
    src: Path,
    *,
    n_segments: int = 8,
    n_shaders: int = 2,
    sequencing: str = "shuffle",
    seed: Optional[int] = None,
) -> BrainWipeRecipe:
    """Time as the crusher: scrub → shaders → echo → shaders → patch → normalize.

    Temporal destruction replaces bitrate crush — the footage is already
    dreaming before color processing hits.
    """
    return BrainWipeRecipe(
        lanes=[Lane(
            source=FootageSource(src),
            n_segments=n_segments,
            recipe=[
                ScrubStep(intensity=0.7, smoothness=1.5),
                ShaderStep(n_shaders=n_shaders),
                EchoStep(delay=0.05, trail=0.85),
                ShaderStep(n_shaders=n_shaders),
                PatchStep(patch_min=0.03, patch_max=0.25),
                NormalizeStep(),
            ],
            sequencing=sequencing,
        )],
        seed=seed,
    )


def deep_time_recipe(
    src: Path,
    *,
    n_segments: int = 8,
    seed: Optional[int] = None,
) -> BrainWipeRecipe:
    """Recursive temporal folding: drift → pingpong → echo → scrub → shader.

    Stacks time effects in sequence — footage is folded through time until
    it becomes abstract texture. A single color shader at the end is all
    the visual processing needed.
    """
    return BrainWipeRecipe(
        lanes=[Lane(
            source=FootageSource(src, method="scene"),
            n_segments=n_segments,
            recipe=[
                DriftStep(loop_dur=2.0),
                PingPongStep(window=0.8),
                EchoStep(delay=0.0, trail=0.9),
                ScrubStep(intensity=0.9, smoothness=0.5),
                ShaderStep(n_shaders=1),
                NormalizeStep(),
            ],
            sequencing="shuffle",
        )],
        seed=seed,
    )


def hybrid_composite_recipe(
    src: Path,
    *,
    n_segments: int = 10,
    crush: float = 0.85,
    n_shaders: int = 3,
    segment_dur: float = 20.0,
    n_warps: int = 3,
    width: Optional[int] = None,
    height: Optional[int] = None,
    composite: Optional[CompositeSpec] = None,
    seed: Optional[int] = None,
    brain_wipe_dir: Optional[Path] = None,
) -> BrainWipeRecipe:
    """Footage meets generator: crushed footage + generator wash, composited via motion mask.

    Movement in the footage opens windows into the generator layer.
    Still moments show pure generator; action reveals glitched footage.

    width/height: resolution for generator lane. Must match source footage
    for masked/blend compositing. If None, defaults to 1920x1080.
    """
    if composite is None:
        composite = MaskedComposite(mask_type="motion")

    w = width or 1920
    h = height or 1080

    return BrainWipeRecipe(
        lanes=[
            Lane(
                source=FootageSource(src),
                n_segments=n_segments,
                recipe=[
                    CrushStep(crush=crush),
                    ShaderStep(n_shaders=n_shaders),
                    CrushStep(crush=crush + 0.05),
                    NormalizeStep(),
                ],
                sequencing="shuffle",
            ),
            Lane(
                source=GeneratorSource(
                    min_dur=segment_dur, max_dur=segment_dur,
                    n_warps=n_warps,
                ),
                n_segments=n_segments,
                recipe=[NormalizeStep()],
                sequencing="shuffle",
            ),
        ],
        composite=composite,
        post=[NormalizeStep()],
        width=w,
        height=h,
        seed=seed,
        brain_wipe_dir=brain_wipe_dir or Path("brain-wipe-shaders"),
    )


def codec_spectrum_recipe(
    src: Path,
    *,
    n_segments: int = 8,
    n_shaders: int = 2,
    sequencing: str = "shuffle",
    seed: Optional[int] = None,
) -> BrainWipeRecipe:
    """Multi-codec crush cascade: mpeg2 → shaders → mpeg4 → shaders → x264 → normalize.

    Three codecs with different artifact characters layered like geological strata.
    MPEG-2 = chunky VHS, MPEG-4 = organic DivX warping, x264 = sharp macroblocking.
    """
    return BrainWipeRecipe(
        lanes=[Lane(
            source=FootageSource(src),
            n_segments=n_segments,
            recipe=[
                CrushStep(crush=0.7, codec="mpeg2video"),
                ShaderStep(n_shaders=n_shaders),
                CrushStep(crush=0.85, codec="mpeg4"),
                ShaderStep(n_shaders=1),
                CrushStep(crush=0.95, codec="libx264"),
                NormalizeStep(),
            ],
            sequencing=sequencing,
        )],
        seed=seed,
    )


def breathing_wall_recipe(
    src: Path,
    *,
    n_shaders: int = 2,
    composite: Optional[CompositeSpec] = None,
    seed: Optional[int] = None,
) -> BrainWipeRecipe:
    """Visual polyrhythm: three lanes with different ping-pong rates, screen-blended.

    Lanes breathe at 0.3s, 0.7s, and 1.2s with echo trails. Screen blend
    preserves brightness from all layers. Creates a living, pulsing wall
    of texture — perfect for drone sets.
    """
    if composite is None:
        composite = BlendComposite(mode="screen", opacity=0.5)

    rates = [(0.3, 6, 0.92), (0.7, 8, 0.85), (1.2, 10, 0.8)]
    lanes = []
    for window, n_seg, trail in rates:
        lanes.append(Lane(
            source=FootageSource(src),
            n_segments=n_seg,
            recipe=[
                PingPongStep(window=window),
                EchoStep(delay=0.0, trail=trail),
                ShaderStep(n_shaders=n_shaders if window < 1.0 else 1),
            ],
            sequencing="shuffle",
        ))

    return BrainWipeRecipe(
        lanes=lanes,
        composite=composite,
        post=[NormalizeStep()],
        seed=seed,
    )


def erosion_recipe(
    src: Path,
    *,
    n_segments: int = 8,
    sequencing: str = "shuffle",
    seed: Optional[int] = None,
) -> BrainWipeRecipe:
    """Progressive downscale crush: 2x → 4x → 8x with shaders between passes.

    Footage erodes into increasingly massive blocks. Shaders process
    the blocky texture at each scale. Structural/brutalist aesthetic.
    """
    return BrainWipeRecipe(
        lanes=[Lane(
            source=FootageSource(src),
            n_segments=n_segments,
            recipe=[
                CrushStep(crush=0.6, downscale=2.0),
                ShaderStep(n_shaders=1),
                CrushStep(crush=0.7, downscale=4.0),
                ShaderStep(n_shaders=1),
                CrushStep(crush=0.8, downscale=8.0),
                NormalizeStep(),
            ],
            sequencing=sequencing,
        )],
        seed=seed,
    )


def palimpsest_recipe(
    src: Path,
    *,
    n_segments: int = 8,
    composite: Optional[CompositeSpec] = None,
    seed: Optional[int] = None,
) -> BrainWipeRecipe:
    """Overwritten memory: same footage, two recipes, composited via edge mask.

    Lane A (dark/heavy) lives in the contours; Lane B (light/temporal)
    lives in the flat areas. Two memories of the same moment on one surface.
    """
    if composite is None:
        composite = MaskedComposite(mask_type="edge")

    return BrainWipeRecipe(
        lanes=[
            Lane(
                source=FootageSource(src, method="scene"),
                n_segments=n_segments,
                recipe=[
                    CrushStep(crush=0.95),
                    ShaderStep(n_shaders=3),
                    EchoStep(delay=0.1, trail=0.7),
                ],
                sequencing="concat",
            ),
            Lane(
                source=FootageSource(src, method="scene"),
                n_segments=n_segments,
                recipe=[
                    ScrubStep(intensity=0.8),
                    ShaderStep(n_shaders=2),
                    PingPongStep(window=0.5),
                    NormalizeStep(),
                ],
                sequencing="concat",
            ),
        ],
        composite=composite,
        post=[NormalizeStep()],
        seed=seed,
    )


def generator_stooges_recipe(
    *,
    segment_counts: list[int] | int = 8,
    segment_dur: float = 15.0,
    n_warps: int = 2,
    crush: float = 0.85,
    n_shaders: int = 2,
    static_gap: float = 0.3,
    seed: Optional[int] = None,
    brain_wipe_dir: Optional[Path] = None,
) -> BrainWipeRecipe:
    """Multi-channel generator CRT: stooges recipe but all-generator, no source footage.

    Each CRT channel shows a different generator shader chain, crush-sandwiched
    and interleaved with static. Alien TV station aesthetic.
    """
    if isinstance(segment_counts, int):
        segment_counts = [segment_counts]

    steps: list[Step] = [
        CrushStep(crush=crush),
        ShaderStep(n_shaders=n_shaders),
        CrushStep(crush=crush + 0.05),
        ShaderStep(n_shaders=n_shaders),
        NormalizeStep(),
    ]

    return BrainWipeRecipe(
        lanes=[
            Lane(
                source=GeneratorSource(min_dur=segment_dur, max_dur=segment_dur, n_warps=n_warps),
                n_segments=count,
                recipe=steps,
                sequencing="shuffle",
                static_gap=static_gap,
            )
            for count in segment_counts
        ],
        seed=seed,
        brain_wipe_dir=brain_wipe_dir or Path("brain-wipe-shaders"),
    )


def gradient_dissolve_recipe(
    src: Path,
    *,
    n_segments: int = 8,
    crush: float = 0.9,
    n_shaders: int = 3,
    segment_dur: float = 20.0,
    n_warps: int = 2,
    direction: str = "radial",
    width: Optional[int] = None,
    height: Optional[int] = None,
    seed: Optional[int] = None,
    brain_wipe_dir: Optional[Path] = None,
) -> BrainWipeRecipe:
    """Spatial wipe composite: footage + generator via gradient mask.

    A radial/horizontal/vertical gradient mask creates spatial zones —
    one treatment in the center, another at the edges. Portal effect.

    width/height: resolution for generator lane. Must match source footage
    for masked compositing. If None, defaults to 1920x1080.
    """
    w = width or 1920
    h = height or 1080

    return BrainWipeRecipe(
        lanes=[
            Lane(
                source=FootageSource(src),
                n_segments=n_segments,
                recipe=[
                    CrushStep(crush=crush),
                    ShaderStep(n_shaders=n_shaders),
                ],
                sequencing="shuffle",
            ),
            Lane(
                source=GeneratorSource(min_dur=segment_dur, max_dur=segment_dur, n_warps=n_warps),
                n_segments=n_segments,
                recipe=[NormalizeStep()],
                sequencing="shuffle",
            ),
        ],
        composite=MaskedComposite(
            mask_type="gradient",
            mask_params={"direction": direction},
        ),
        post=[NormalizeStep()],
        width=w,
        height=h,
        seed=seed,
        brain_wipe_dir=brain_wipe_dir or Path("brain-wipe-shaders"),
    )


def accretion_recipe(
    src: Path,
    *,
    n_segments: int = 12,
    composite: Optional[CompositeSpec] = None,
    seed: Optional[int] = None,
) -> BrainWipeRecipe:
    """Layer-by-layer buildup: four lanes at escalating destruction, screen-blended.

    Lane A = barely touched, Lane D = destroyed. Screen blend accumulates
    all layers — recognizable structure from light processing, texture and
    chaos from heavy. Geological layering.
    """
    if composite is None:
        composite = BlendComposite(mode="screen", opacity=0.4)

    return BrainWipeRecipe(
        lanes=[
            # A: barely touched
            Lane(
                source=FootageSource(src, method="scene"),
                n_segments=n_segments,
                recipe=[ShaderStep(n_shaders=1)],
                sequencing="concat",
            ),
            # B: moderate
            Lane(
                source=FootageSource(src, method="scene"),
                n_segments=n_segments,
                recipe=[
                    CrushStep(crush=0.6),
                    ShaderStep(n_shaders=2),
                    EchoStep(delay=0.0, trail=0.7),
                ],
                sequencing="concat",
            ),
            # C: heavy
            Lane(
                source=FootageSource(src, method="scene"),
                n_segments=n_segments,
                recipe=[
                    CrushStep(crush=0.85),
                    ShaderStep(n_shaders=3),
                    ScrubStep(intensity=0.5),
                ],
                sequencing="concat",
            ),
            # D: destroyed
            Lane(
                source=FootageSource(src, method="scene"),
                n_segments=n_segments,
                recipe=[
                    CrushStep(crush=0.95, downscale=4.0),
                    ShaderStep(n_shaders=4),
                    PatchStep(patch_min=0.05, patch_max=0.3),
                    CrushStep(crush=1.0),
                ],
                sequencing="concat",
            ),
        ],
        composite=composite,
        post=[NormalizeStep()],
        seed=seed,
    )


def transition_reel_recipe(
    src: Path,
    *,
    n_segments: int = 8,
    n_shaders: int = 2,
    transition_dur: float = 1.0,
    sequencing: str = "shuffle",
    seed: Optional[int] = None,
) -> BrainWipeRecipe:
    """Montage: footage segments → shaders → crossfade transitions.

    Clean, flowing reel — segments are individually processed with shaders
    then sequenced with smooth crossfades between each pair.
    """
    return BrainWipeRecipe(
        lanes=[Lane(
            source=FootageSource(src),
            n_segments=n_segments,
            recipe=[ShaderStep(n_shaders=n_shaders)],
            sequencing=sequencing,
            transition=TransitionSpec(type="crossfade", duration=transition_dur),
        )],
        seed=seed,
    )


def channel_surf_recipe(
    src: Path,
    *,
    n_segments: int = 8,
    crush: float = 0.95,
    n_shaders: int = 2,
    seed: Optional[int] = None,
) -> BrainWipeRecipe:
    """Glitchy TV: crush sandwich → static burst transitions.

    Each segment gets a crush sandwich (crush → shaders → crush → normalize)
    then segments are stitched together with short bursts of TV static.
    Channel-surfing aesthetic.
    """
    return BrainWipeRecipe(
        lanes=[Lane(
            source=FootageSource(src),
            n_segments=n_segments,
            recipe=[
                CrushStep(crush=crush),
                ShaderStep(n_shaders=n_shaders),
                CrushStep(crush=crush * 0.95),
                NormalizeStep(),
            ],
            sequencing="shuffle",
            transition=TransitionSpec(type="static_burst", duration=0.3),
        )],
        seed=seed,
    )


def dissolve_dream_recipe(
    src: Path,
    *,
    n_segments: int = 8,
    n_shaders: int = 2,
    transition_dur: float = 2.0,
    seed: Optional[int] = None,
) -> BrainWipeRecipe:
    """Structural/dreamy: time effects + shaders → soft radial luma wipe.

    Segments are drifted, shader-processed, then echo-trailed. Soft radial
    luma wipes dissolve between each pair — slow, hypnotic transitions
    that feel like drifting between states.
    """
    return BrainWipeRecipe(
        lanes=[Lane(
            source=FootageSource(src),
            n_segments=n_segments,
            recipe=[
                DriftStep(loop_dur=0.5),
                ShaderStep(n_shaders=n_shaders),
                EchoStep(delay=0.0, trail=0.7),
            ],
            sequencing="concat",
            transition=TransitionSpec(
                type="luma_wipe", duration=transition_dur,
                pattern="radial", softness=0.3,
            ),
        )],
        seed=seed,
    )


# ─── Procedural recipe generator ─────────────────────────────────────────────

_STEP_POOL: list[tuple[type, int]] = [
    (ShaderStep, 5),
    (CrushStep, 3),
    (ScrubStep, 2),
    (DriftStep, 2),
    (PingPongStep, 2),
    (EchoStep, 2),
    (PatchStep, 2),
]

_TIME_STEPS = (ScrubStep, DriftStep, PingPongStep, EchoStep, PatchStep)

_TRANSITION_POOL: list[tuple[str, int]] = [
    ("crossfade", 4),
    ("luma_wipe", 3),
    ("whip_pan", 2),
    ("static_burst", 2),
    ("flash", 2),
]

_WIPE_PATTERNS = [
    "horizontal", "vertical", "radial", "diagonal",
    "directional", "noise", "star",
]

_BLEND_MODES = ["overlay", "softlight", "difference", "multiply", "normal"]

_MASK_TYPES = ["luma", "edge", "motion", "gradient"]

_CODECS = ["libx264", "mpeg2video", "mpeg4"]

_WHIP_DIRS = ["left", "right", "up", "down"]


def _weighted_choice(rng: _random_mod.Random, pool: list[tuple[Any, int]]) -> Any:
    """Pick from a weighted pool."""
    items, weights = zip(*pool)
    total = sum(weights)
    r = rng.random() * total
    cumulative = 0
    for item, w in zip(items, weights):
        cumulative += w
        if r <= cumulative:
            return item
    return items[-1]


def _random_step(rng: _random_mod.Random, complexity: float = 0.5) -> Step:
    """Generate a single random step with randomized parameters."""
    # At low complexity, suppress time effects (the expensive ones)
    pool = list(_STEP_POOL)
    if complexity < 0.5:
        # Scale down time effect weights: at complexity=0 they're near-zero
        time_scale = max(0.1, complexity * 2)  # 0→0.1, 0.5→1.0
        pool = [
            (cls, max(1, int(w * time_scale)) if cls in _TIME_STEPS else w)
            for cls, w in pool
        ]
    cls = _weighted_choice(rng, pool)
    if cls is ShaderStep:
        max_shaders = max(1, int(1 + complexity * 3))  # 0→1, 0.5→2, 1.0→4
        return ShaderStep(n_shaders=rng.randint(1, max_shaders))
    elif cls is CrushStep:
        return CrushStep(
            crush=rng.uniform(0.5, 1.0),
            codec=rng.choice(_CODECS),
            downscale=rng.choice([1.0, 1.0, 1.0, 2.0, 4.0]),
        )
    elif cls is ScrubStep:
        return ScrubStep(
            smoothness=rng.uniform(1.0, 4.0),
            intensity=rng.uniform(0.2, 0.8),
        )
    elif cls is DriftStep:
        return DriftStep(loop_dur=rng.uniform(0.3, 1.5))
    elif cls is PingPongStep:
        return PingPongStep(window=rng.uniform(0.3, 1.5))
    elif cls is EchoStep:
        delay = 0.0 if rng.random() < 0.4 else rng.uniform(0.02, 0.3)
        return EchoStep(delay=delay, trail=rng.uniform(0.5, 0.9))
    else:  # PatchStep
        mn = rng.uniform(0.03, 0.15)
        return PatchStep(patch_min=mn, patch_max=rng.uniform(mn + 0.1, 0.5))


def _random_steps(rng: _random_mod.Random, n_steps: int, complexity: float = 0.5) -> list[Step]:
    """Generate a random processing recipe with constraints."""
    steps = [_random_step(rng, complexity) for _ in range(n_steps)]

    # Constraint: at most 2 crush steps
    crush_count = 0
    filtered = []
    for s in steps:
        if isinstance(s, CrushStep):
            crush_count += 1
            if crush_count > 2:
                filtered.append(_random_step(rng))  # replace with something else
                continue
        filtered.append(s)
    steps = filtered

    # Constraint: at most 2 time effects
    time_count = 0
    filtered = []
    for s in steps:
        if isinstance(s, _TIME_STEPS):
            time_count += 1
            if time_count > 2:
                max_sh = max(1, int(1 + complexity * 3))
                filtered.append(ShaderStep(n_shaders=rng.randint(1, max_sh)))
                continue
        filtered.append(s)
    steps = filtered

    # Guarantee at least 1 shader step
    if not any(isinstance(s, ShaderStep) for s in steps):
        max_sh = max(1, int(1 + complexity * 3))
        pos = rng.randint(0, max(0, len(steps) - 1))
        steps.insert(pos, ShaderStep(n_shaders=rng.randint(1, max_sh)))

    return steps


def _ensure_motion(rng: _random_mod.Random, steps: list[Step],
                    source: Optional[Source] = None) -> list[Step]:
    """If no time effects in steps, insert one before the last step.

    Skips generator sources — they already have inherent motion from u_time.
    """
    if source is not None and isinstance(source, GeneratorSource):
        return steps
    if any(isinstance(s, _TIME_STEPS) for s in steps):
        return steps

    effect = rng.choice([
        EchoStep(delay=0.0, trail=rng.uniform(0.7, 0.95)),
        DriftStep(loop_dur=rng.uniform(0.5, 1.5), drift=None),
        PingPongStep(window=rng.uniform(0.3, 0.8)),
    ])
    steps = list(steps)
    steps.insert(max(0, len(steps) - 1), effect)
    return steps


def _random_source(
    rng: _random_mod.Random,
    src: Optional[Path],
    use_generators: Optional[bool],
    complexity: float = 0.5,
) -> SourceSpec:
    """Pick a random source type."""
    max_warps = max(1, int(1 + complexity * 3))  # 0→1, 0.5→2, 1.0→4
    if src is not None and use_generators is not True:
        roll = rng.random()
        if roll < 0.70 or use_generators is False:
            return FootageSource(
                src, method=rng.choice(["random", "scene"]),
            )
        elif roll < 0.90:
            lo = rng.uniform(8, 15)
            return GeneratorSource(
                min_dur=lo, max_dur=lo + rng.uniform(5, 20),
                n_warps=rng.randint(0, max_warps),
            )
        else:
            lo = rng.uniform(4, 8)
            return StaticSource(min_dur=lo, max_dur=lo + rng.uniform(3, 10))
    else:
        # No source footage — generators only
        if rng.random() < 0.80:
            lo = rng.uniform(8, 15)
            return GeneratorSource(
                min_dur=lo, max_dur=lo + rng.uniform(5, 20),
                n_warps=rng.randint(1, max_warps),
            )
        else:
            lo = rng.uniform(4, 8)
            return StaticSource(min_dur=lo, max_dur=lo + rng.uniform(3, 10))


def _random_transition(rng: _random_mod.Random) -> TransitionSpec:
    """Generate a random transition spec."""
    t_type = _weighted_choice(rng, _TRANSITION_POOL)

    if t_type == "crossfade":
        return TransitionSpec(type="crossfade", duration=rng.uniform(0.5, 2.0))
    elif t_type == "luma_wipe":
        return TransitionSpec(
            type="luma_wipe",
            duration=rng.uniform(0.5, 2.0),
            pattern=rng.choice(_WIPE_PATTERNS),
            softness=rng.uniform(0.05, 0.4),
            angle=rng.uniform(0, 360) if rng.random() < 0.3 else 0.0,
        )
    elif t_type == "whip_pan":
        return TransitionSpec(
            type="whip_pan",
            duration=rng.uniform(0.3, 0.8),
            direction=rng.choice(_WHIP_DIRS),
            blur_strength=rng.uniform(0.3, 0.8),
        )
    elif t_type == "static_burst":
        return TransitionSpec(
            type="static_burst",
            duration=rng.uniform(0.2, 0.5),
        )
    else:  # flash
        return TransitionSpec(
            type="flash",
            duration=rng.uniform(0.3, 0.8),
            decay=rng.uniform(2.0, 5.0),
        )


def _random_composite(rng: _random_mod.Random) -> CompositeSpec:
    """Pick a random compositing method for multi-lane recipes."""
    return MaskedComposite(mask_type=rng.choice(_MASK_TYPES))


def _random_post(rng: _random_mod.Random) -> list[Step]:
    """Generate random post-processing steps (light touch).

    No shaders here — a global shader pass drowns out per-segment variety.
    Post is for subtle temporal effects and level correction only.
    """
    steps: list[Step] = []
    # Maybe a light echo (motion blur, not distinct echoes)
    if rng.random() < 0.2:
        steps.append(EchoStep(delay=0.0, trail=rng.uniform(0.5, 0.8)))
    return steps


# ─── Shared helpers for archetype builders ──────────────────────────────────

def _resolve_segments(
    rng: _random_mod.Random,
    complexity: float,
    n_segments: Optional[int],
) -> int:
    """Derive segment count from complexity, unless overridden."""
    if n_segments is not None:
        return n_segments
    seg_lo = 3 + int(complexity * 5)
    seg_hi = 5 + int(complexity * 11)
    return rng.randint(seg_lo, seg_hi)


def _seg_dur_target(
    target_dur: Optional[float],
    actual_segments: int,
    wants_transition: bool,
) -> Optional[float]:
    """Compute per-segment duration target from total target_dur."""
    if target_dur is None:
        return None
    n_transitions = actual_segments - 1 if wants_transition else 0
    avg_trans_dur = 1.0
    overlap = n_transitions * avg_trans_dur
    effective_dur = target_dur + overlap
    return max(3.0, effective_dur / max(actual_segments, 1))


def _override_source_dur(source: SourceSpec, seg_target: float) -> SourceSpec:
    """Override source durations to hit a target per-segment duration."""
    lo = max(2.0, seg_target * 0.7)
    hi = seg_target * 1.3
    if isinstance(source, FootageSource):
        return FootageSource(
            path=source.path, method=source.method,
            min_dur=lo, max_dur=hi,
        )
    elif isinstance(source, GeneratorSource):
        return GeneratorSource(
            min_dur=lo, max_dur=hi,
            n_warps=source.n_warps,
            warp_categories=source.warp_categories,
        )
    elif isinstance(source, StaticSource):
        return StaticSource(min_dur=lo, max_dur=hi)
    elif isinstance(source, SolidSource):
        return SolidSource(min_dur=lo, max_dur=hi, color=source.color)
    return source


def _assemble_recipe(
    lanes: list[Lane],
    *,
    rng: _random_mod.Random,
    complexity: float,
    src: Optional[Path],
    seed: Optional[int],
    wants_post: bool,
) -> BrainWipeRecipe:
    """Final assembly: composite, post-processing, resolution."""
    composite: Optional[CompositeSpec] = None
    if len(lanes) > 1:
        composite = _random_composite(rng)

    post: list[Step] = _random_post(rng) if wants_post else []

    has_generator = any(isinstance(l.source, GeneratorSource) for l in lanes)
    has_footage = any(isinstance(l.source, FootageSource) for l in lanes)

    width, height = 1920, 1080
    if has_footage and has_generator and src is not None:
        width, height = 1280, 720

    return BrainWipeRecipe(
        lanes=lanes,
        composite=composite,
        post=post,
        width=width,
        height=height,
        seed=seed,
    )


def _make_lane(
    rng: _random_mod.Random,
    *,
    source: SourceSpec,
    steps: list[Step],
    n_segments: int,
    wants_transition: bool,
    seg_target: Optional[float] = None,
) -> Lane:
    """Build a Lane, optionally overriding source durations."""
    if seg_target is not None:
        source = _override_source_dur(source, seg_target)
    sequencing = "shuffle" if rng.random() < 0.6 else "concat"
    transition = _random_transition(rng) if wants_transition else None
    return Lane(
        source=source,
        n_segments=n_segments,
        recipe=steps,
        sequencing=sequencing,
        transition=transition,
    )


def _random_time_step(rng: _random_mod.Random, complexity: float = 0.5) -> Step:
    """Generate a random time-effect step with randomized parameters."""
    cls = rng.choice([ScrubStep, DriftStep, PingPongStep, EchoStep, PatchStep])
    if cls is ScrubStep:
        return ScrubStep(
            smoothness=rng.uniform(1.0, 4.0),
            intensity=rng.uniform(0.2 + complexity * 0.3, 0.4 + complexity * 0.5),
        )
    elif cls is DriftStep:
        return DriftStep(loop_dur=rng.uniform(0.3, 0.5 + complexity * 1.5))
    elif cls is PingPongStep:
        return PingPongStep(window=rng.uniform(0.3, 0.5 + complexity * 1.0))
    elif cls is EchoStep:
        delay = 0.0 if rng.random() < 0.4 else rng.uniform(0.02, 0.3)
        return EchoStep(delay=delay, trail=rng.uniform(0.5 + complexity * 0.2, 0.9))
    else:  # PatchStep
        mn = rng.uniform(0.03, 0.15)
        return PatchStep(patch_min=mn, patch_max=rng.uniform(mn + 0.1, 0.5))


def _shader_step(rng: _random_mod.Random, complexity: float, n: Optional[int] = None) -> ShaderStep:
    """ShaderStep with complexity-scaled n_shaders."""
    if n is not None:
        return ShaderStep(n_shaders=n)
    max_sh = max(1, int(1 + complexity * 3))
    return ShaderStep(n_shaders=rng.randint(1, max_sh))


# ─── Archetype builders ─────────────────────────────────────────────────────

def _build_crush_sandwich(
    rng: _random_mod.Random,
    complexity: float,
    src: Optional[Path],
    *,
    n_lanes: Optional[int],
    n_steps: Optional[int],
    n_segments: Optional[int],
    use_transitions: Optional[bool],
    use_generators: Optional[bool],
    target_dur: Optional[float],
    seed: Optional[int],
) -> BrainWipeRecipe:
    """Alternating crush/shader pairs (C-S-C-S), optional codec cascade."""
    actual_segments = _resolve_segments(rng, complexity, n_segments)
    wants_transition = use_transitions if use_transitions is not None else (
        rng.random() < 0.2 + 0.6 * complexity
    )
    seg_target = _seg_dur_target(target_dur, actual_segments, wants_transition)

    n_pairs = n_steps or max(1, int(1 + complexity * 2))
    use_codec_cascade = complexity > 0.6 and rng.random() < 0.4
    codecs = ["mpeg2video", "mpeg4", "libx264"] if use_codec_cascade else None

    steps: list[Step] = []
    for i in range(n_pairs):
        crush_val = rng.uniform(0.5 + complexity * 0.2, 0.8 + complexity * 0.2)
        codec = codecs[i % len(codecs)] if codecs else rng.choice(_CODECS)
        downscale = rng.choice([1.0, 1.0, 2.0]) if complexity > 0.5 else 1.0
        steps.append(CrushStep(crush=crush_val, codec=codec, downscale=downscale))
        steps.append(_shader_step(rng, complexity))

    source = _random_source(rng, src, use_generators, complexity)
    steps = _ensure_motion(rng, steps, source)
    lane = _make_lane(rng, source=source, steps=steps, n_segments=actual_segments,
                      wants_transition=wants_transition, seg_target=seg_target)

    wants_post = rng.random() < 0.1 + 0.6 * complexity
    return _assemble_recipe([lane], rng=rng, complexity=complexity, src=src,
                            seed=seed, wants_post=wants_post)


def _build_deep_time(
    rng: _random_mod.Random,
    complexity: float,
    src: Optional[Path],
    *,
    n_lanes: Optional[int],
    n_steps: Optional[int],
    n_segments: Optional[int],
    use_transitions: Optional[bool],
    use_generators: Optional[bool],
    target_dur: Optional[float],
    seed: Optional[int],
) -> BrainWipeRecipe:
    """3–5 stacked time effects, 1 shader, normalize. Temporal destruction."""
    actual_segments = _resolve_segments(rng, complexity, n_segments)
    wants_transition = use_transitions if use_transitions is not None else (
        rng.random() < 0.2 + 0.6 * complexity
    )
    seg_target = _seg_dur_target(target_dur, actual_segments, wants_transition)

    n_time = n_steps or max(3, int(3 + complexity * 2))
    n_time = min(n_time, 5)
    all_time = [ScrubStep, DriftStep, PingPongStep, EchoStep, PatchStep]
    rng.shuffle(all_time)
    selected = all_time[:n_time]

    steps: list[Step] = []
    for cls in selected:
        if cls is ScrubStep:
            steps.append(ScrubStep(
                smoothness=rng.uniform(0.5, 2.0 + complexity * 2.0),
                intensity=rng.uniform(0.4 + complexity * 0.3, 0.7 + complexity * 0.3),
            ))
        elif cls is DriftStep:
            steps.append(DriftStep(loop_dur=rng.uniform(0.3 + complexity * 0.5, 1.0 + complexity * 1.5)))
        elif cls is PingPongStep:
            steps.append(PingPongStep(window=rng.uniform(0.3, 0.5 + complexity * 0.8)))
        elif cls is EchoStep:
            delay = 0.0 if rng.random() < 0.5 else rng.uniform(0.02, 0.15)
            steps.append(EchoStep(delay=delay, trail=rng.uniform(0.7 + complexity * 0.1, 0.95)))
        else:  # PatchStep
            mn = rng.uniform(0.03, 0.1)
            steps.append(PatchStep(patch_min=mn, patch_max=rng.uniform(mn + 0.1, 0.4)))
    steps.append(_shader_step(rng, complexity, n=1))

    source = _random_source(rng, src, use_generators, complexity)
    lane = _make_lane(rng, source=source, steps=steps, n_segments=actual_segments,
                      wants_transition=wants_transition, seg_target=seg_target)

    return _assemble_recipe([lane], rng=rng, complexity=complexity, src=src,
                            seed=seed, wants_post=False)


def _build_temporal_sandwich(
    rng: _random_mod.Random,
    complexity: float,
    src: Optional[Path],
    *,
    n_lanes: Optional[int],
    n_steps: Optional[int],
    n_segments: Optional[int],
    use_transitions: Optional[bool],
    use_generators: Optional[bool],
    target_dur: Optional[float],
    seed: Optional[int],
) -> BrainWipeRecipe:
    """Alternating time/shader pairs (T-S-T-S-T), normalize."""
    actual_segments = _resolve_segments(rng, complexity, n_segments)
    wants_transition = use_transitions if use_transitions is not None else (
        rng.random() < 0.2 + 0.6 * complexity
    )
    seg_target = _seg_dur_target(target_dur, actual_segments, wants_transition)

    n_pairs = n_steps or max(2, int(2 + complexity * 2))

    steps: list[Step] = []
    for _ in range(n_pairs):
        steps.append(_random_time_step(rng, complexity))
        steps.append(_shader_step(rng, complexity))

    source = _random_source(rng, src, use_generators, complexity)
    lane = _make_lane(rng, source=source, steps=steps, n_segments=actual_segments,
                      wants_transition=wants_transition, seg_target=seg_target)

    wants_post = rng.random() < 0.1 + 0.4 * complexity
    return _assemble_recipe([lane], rng=rng, complexity=complexity, src=src,
                            seed=seed, wants_post=wants_post)


def _build_escalation(
    rng: _random_mod.Random,
    complexity: float,
    src: Optional[Path],
    *,
    n_lanes: Optional[int],
    n_steps: Optional[int],
    n_segments: Optional[int],
    use_transitions: Optional[bool],
    use_generators: Optional[bool],
    target_dur: Optional[float],
    seed: Optional[int],
) -> BrainWipeRecipe:
    """Progressive parameter increase across steps or lanes."""
    actual_segments = _resolve_segments(rng, complexity, n_segments)
    wants_transition = use_transitions if use_transitions is not None else (
        rng.random() < 0.2 + 0.6 * complexity
    )
    seg_target = _seg_dur_target(target_dur, actual_segments, wants_transition)
    wants_post = rng.random() < 0.1 + 0.4 * complexity

    actual_lanes = n_lanes or (rng.randint(2, 4) if complexity > 0.7 and rng.random() < 0.5 else 1)

    if actual_lanes == 1:
        # Within-lane escalation: progressive crush or downscale
        n_stages = n_steps or max(2, int(2 + complexity * 2))
        variant = rng.choice(["crush", "downscale"])

        steps: list[Step] = []
        for i in range(n_stages):
            progress = i / max(1, n_stages - 1)  # 0.0 → 1.0
            if variant == "crush":
                steps.append(CrushStep(
                    crush=0.5 + 0.45 * progress,
                    codec=rng.choice(_CODECS),
                    downscale=1.0,
                ))
            else:  # downscale erosion
                steps.append(CrushStep(
                    crush=0.6 + 0.2 * progress,
                    codec=rng.choice(_CODECS),
                    downscale=2.0 ** (1 + progress * 2),
                ))
            steps.append(_shader_step(rng, complexity, n=max(1, int(1 + progress * 2))))

        source = _random_source(rng, src, use_generators, complexity)
        steps = _ensure_motion(rng, steps, source)
        lane = _make_lane(rng, source=source, steps=steps, n_segments=actual_segments,
                          wants_transition=wants_transition, seg_target=seg_target)
        return _assemble_recipe([lane], rng=rng, complexity=complexity, src=src,
                                seed=seed, wants_post=wants_post)
    else:
        # Cross-lane escalation (accretion): each lane progressively more intense
        lanes: list[Lane] = []
        for i in range(actual_lanes):
            intensity = i / max(1, actual_lanes - 1)  # 0.0 → 1.0
            lane_steps: list[Step] = []

            # Mild lanes: just shader(s)
            # Intense lanes: crush + more shaders + time effect + maybe double-crush
            if intensity < 0.3:
                lane_steps.append(_shader_step(rng, complexity, n=max(1, int(complexity * 2))))
            else:
                crush_val = 0.5 + 0.4 * intensity
                downscale = 1.0 if intensity < 0.6 else rng.choice([1.0, 2.0, 4.0])
                lane_steps.append(CrushStep(crush=crush_val, codec=rng.choice(_CODECS),
                                            downscale=downscale))
                lane_steps.append(_shader_step(rng, complexity,
                                               n=max(1, int(1 + intensity * 3))))
                if intensity > 0.5:
                    lane_steps.append(_random_time_step(rng, complexity))
                if intensity > 0.8:
                    lane_steps.append(CrushStep(crush=rng.uniform(0.9, 1.0),
                                                codec=rng.choice(_CODECS)))

            source = _random_source(rng, src, use_generators, complexity)
            lane_steps = _ensure_motion(rng, lane_steps, source)
            lane = _make_lane(rng, source=source, steps=lane_steps,
                              n_segments=actual_segments,
                              wants_transition=wants_transition, seg_target=seg_target)
            lanes.append(lane)

        recipe = _assemble_recipe(lanes, rng=rng, complexity=complexity, src=src,
                                  seed=seed, wants_post=wants_post)
        recipe.composite = MaskedComposite(
            mask_type=rng.choice(_MASK_TYPES),
        )
        return recipe


def _build_polyrhythm(
    rng: _random_mod.Random,
    complexity: float,
    src: Optional[Path],
    *,
    n_lanes: Optional[int],
    n_steps: Optional[int],
    n_segments: Optional[int],
    use_transitions: Optional[bool],
    use_generators: Optional[bool],
    target_dur: Optional[float],
    seed: Optional[int],
) -> BrainWipeRecipe:
    """2–4 lanes with harmonically-related temporal rates, brightness-neutral blend."""
    actual_segments = _resolve_segments(rng, complexity, n_segments)
    wants_transition = use_transitions if use_transitions is not None else (
        rng.random() < 0.2 + 0.6 * complexity
    )
    seg_target = _seg_dur_target(target_dur, actual_segments, wants_transition)

    actual_lanes = n_lanes or max(2, int(2 + complexity * 2))
    base_rate = rng.uniform(0.2, 0.5)
    multipliers = [1.0, 2.0, 3.5, 5.0][:actual_lanes]

    lanes: list[Lane] = []
    for i, mult in enumerate(multipliers):
        rate = base_rate * mult
        trail = 0.95 - i * 0.05
        n_sh = max(1, int(1 + complexity * (1 + i * 0.3)))

        lane_steps: list[Step] = [
            PingPongStep(window=rate) if rng.random() < 0.7 else DriftStep(loop_dur=rate),
            EchoStep(delay=0.0, trail=trail),
            _shader_step(rng, complexity, n=n_sh),
        ]
        source = _random_source(rng, src, use_generators, complexity)
        lane = _make_lane(rng, source=source, steps=lane_steps,
                          n_segments=actual_segments,
                          wants_transition=wants_transition, seg_target=seg_target)
        lanes.append(lane)

    recipe = _assemble_recipe(lanes, rng=rng, complexity=complexity, src=src,
                              seed=seed, wants_post=rng.random() < 0.3)
    recipe.composite = MaskedComposite(
        mask_type=rng.choice(_MASK_TYPES),
    )
    return recipe


def _build_palimpsest(
    rng: _random_mod.Random,
    complexity: float,
    src: Optional[Path],
    *,
    n_lanes: Optional[int],
    n_steps: Optional[int],
    n_segments: Optional[int],
    use_transitions: Optional[bool],
    use_generators: Optional[bool],
    target_dur: Optional[float],
    seed: Optional[int],
) -> BrainWipeRecipe:
    """Two lanes, same source, contrasting treatments, masked composite."""
    actual_segments = _resolve_segments(rng, complexity, n_segments)
    wants_transition = use_transitions if use_transitions is not None else (
        rng.random() < 0.2 + 0.6 * complexity
    )
    seg_target = _seg_dur_target(target_dur, actual_segments, wants_transition)

    # Lane A: crush-dominant (dark/heavy)
    a_steps: list[Step] = [
        CrushStep(crush=rng.uniform(0.8, 1.0), codec=rng.choice(_CODECS),
                  downscale=rng.choice([1.0, 1.0, 2.0])),
        _shader_step(rng, complexity, n=max(1, int(1 + complexity * 2))),
    ]
    if complexity > 0.4:
        a_steps.append(EchoStep(delay=rng.uniform(0.02, 0.15),
                                trail=rng.uniform(0.6, 0.8)))

    # Lane B: time-dominant (light/temporal)
    b_steps: list[Step] = [
        _random_time_step(rng, complexity),
        _shader_step(rng, complexity, n=max(1, int(complexity * 2))),
    ]
    if complexity > 0.5:
        b_steps.append(_random_time_step(rng, complexity))

    # Same source for both lanes
    source = _random_source(rng, src, use_generators, complexity)
    a_steps = _ensure_motion(rng, a_steps, source)

    lane_a = _make_lane(rng, source=source, steps=a_steps,
                        n_segments=actual_segments,
                        wants_transition=wants_transition, seg_target=seg_target)
    lane_b = _make_lane(rng, source=source, steps=b_steps,
                        n_segments=actual_segments,
                        wants_transition=wants_transition, seg_target=seg_target)

    recipe = _assemble_recipe([lane_a, lane_b], rng=rng, complexity=complexity,
                              src=src, seed=seed, wants_post=rng.random() < 0.3)
    # Always masked composite for palimpsest
    recipe.composite = MaskedComposite(
        mask_type=rng.choice(["edge", "motion", "luma"]),
    )
    return recipe


def _build_hybrid(
    rng: _random_mod.Random,
    complexity: float,
    src: Optional[Path],
    *,
    n_lanes: Optional[int],
    n_steps: Optional[int],
    n_segments: Optional[int],
    use_transitions: Optional[bool],
    use_generators: Optional[bool],
    target_dur: Optional[float],
    seed: Optional[int],
) -> BrainWipeRecipe:
    """Footage + generator lane, masked composite."""
    actual_segments = _resolve_segments(rng, complexity, n_segments)
    wants_transition = use_transitions if use_transitions is not None else (
        rng.random() < 0.2 + 0.6 * complexity
    )
    seg_target = _seg_dur_target(target_dur, actual_segments, wants_transition)

    # Footage lane: crush/shader treatment
    footage_steps: list[Step] = []
    if rng.random() < 0.6:
        footage_steps.append(CrushStep(crush=rng.uniform(0.6, 0.95),
                                       codec=rng.choice(_CODECS)))
    footage_steps.append(_shader_step(rng, complexity))
    if complexity > 0.5 and rng.random() < 0.5:
        footage_steps.append(_random_time_step(rng, complexity))

    footage_source = FootageSource(
        src, method=rng.choice(["random", "scene"]),
    )

    # Generator lane: warps + minimal processing
    max_warps = max(1, int(1 + complexity * 3))
    gen_source = GeneratorSource(
        n_warps=rng.randint(1, max_warps),
    )
    gen_steps: list[Step] = [_shader_step(rng, complexity, n=1)]

    footage_steps = _ensure_motion(rng, footage_steps, footage_source)
    gen_steps = _ensure_motion(rng, gen_steps, gen_source)

    lane_a = _make_lane(rng, source=footage_source, steps=footage_steps,
                        n_segments=actual_segments,
                        wants_transition=wants_transition, seg_target=seg_target)
    lane_b = _make_lane(rng, source=gen_source, steps=gen_steps,
                        n_segments=actual_segments,
                        wants_transition=wants_transition, seg_target=seg_target)

    recipe = _assemble_recipe([lane_a, lane_b], rng=rng, complexity=complexity,
                              src=src, seed=seed, wants_post=rng.random() < 0.3)
    recipe.composite = MaskedComposite(
        mask_type=rng.choice(["motion", "gradient"]),
    )
    recipe.width, recipe.height = 1280, 720
    return recipe


def _build_grab_bag(
    rng: _random_mod.Random,
    complexity: float,
    src: Optional[Path],
    *,
    n_lanes: Optional[int],
    n_steps: Optional[int],
    n_segments: Optional[int],
    use_transitions: Optional[bool],
    use_generators: Optional[bool],
    target_dur: Optional[float],
    seed: Optional[int],
) -> BrainWipeRecipe:
    """Original random recipe behavior — independent step draws from pool."""
    actual_lanes = n_lanes or max(1, int(1 + complexity * 3 * rng.random()))
    actual_segments = _resolve_segments(rng, complexity, n_segments)

    wants_transition = use_transitions if use_transitions is not None else (
        rng.random() < 0.2 + 0.6 * complexity
    )
    wants_post = rng.random() < 0.1 + 0.6 * complexity
    seg_target = _seg_dur_target(target_dur, actual_segments, wants_transition)

    lanes: list[Lane] = []
    for _ in range(actual_lanes):
        lane_steps = n_steps or max(1, int(1 + complexity * 5 * rng.random()))
        source = _random_source(rng, src, use_generators, complexity)
        if seg_target is not None:
            source = _override_source_dur(source, seg_target)
        steps = _ensure_motion(rng, _random_steps(rng, lane_steps, complexity), source)
        sequencing = "shuffle" if rng.random() < 0.6 else "concat"
        transition = _random_transition(rng) if wants_transition else None
        lanes.append(Lane(
            source=source, n_segments=actual_segments,
            recipe=steps, sequencing=sequencing, transition=transition,
        ))

    return _assemble_recipe(lanes, rng=rng, complexity=complexity, src=src,
                            seed=seed, wants_post=wants_post)


def _build_stutter(
    rng: _random_mod.Random,
    complexity: float,
    src: Optional[Path],
    *,
    n_lanes: Optional[int],
    n_steps: Optional[int],
    n_segments: Optional[int],
    use_transitions: Optional[bool],
    use_generators: Optional[bool],
    target_dur: Optional[float],
    seed: Optional[int],
) -> BrainWipeRecipe:
    """Rapid-fire short segments with hard cuts. Channel-surfing within a show."""
    # Many short segments — the structure IS the effect
    actual_segments = n_segments or max(4, int(4 + complexity * 8))

    # Short per-segment durations (0.5–2s each)
    seg_dur = rng.uniform(0.5, 1.0 + complexity * 1.0)

    # Light processing per segment — let the cuts do the work
    n_st = n_steps or max(1, int(1 + complexity * 2))
    steps = _random_steps(rng, n_st, complexity)
    source = _random_source(rng, src, use_generators, complexity)
    steps = _ensure_motion(rng, steps, source)
    source = _override_source_dur(source, seg_dur)

    # Always shuffle for maximum discontinuity
    transition = None  # hard cuts — no transitions
    lane = Lane(
        source=source,
        n_segments=actual_segments,
        recipe=steps,
        sequencing="shuffle",
        transition=transition,
    )

    return _assemble_recipe([lane], rng=rng, complexity=complexity, src=src,
                            seed=seed, wants_post=rng.random() < 0.2)


def _build_echo_chamber(
    rng: _random_mod.Random,
    complexity: float,
    src: Optional[Path],
    *,
    n_lanes: Optional[int],
    n_steps: Optional[int],
    n_segments: Optional[int],
    use_transitions: Optional[bool],
    use_generators: Optional[bool],
    target_dur: Optional[float],
    seed: Optional[int],
) -> BrainWipeRecipe:
    """Stacked echo passes with increasing delay — temporal feedback ghosts."""
    actual_segments = _resolve_segments(rng, complexity, n_segments)
    wants_transition = use_transitions if use_transitions is not None else (
        rng.random() < 0.2 + 0.6 * complexity
    )
    seg_target = _seg_dur_target(target_dur, actual_segments, wants_transition)

    # 3–5 echo passes with increasing delay
    n_echoes = n_steps or max(3, int(3 + complexity * 2))
    n_echoes = min(n_echoes, 5)

    steps: list[Step] = []
    for i in range(n_echoes):
        progress = i / max(1, n_echoes - 1)  # 0.0 → 1.0
        # Delay ramps from motion blur (0) to distinct echoes (0.3s)
        delay = progress * rng.uniform(0.15, 0.35)
        # Trail stays high — each pass accumulates
        trail = rng.uniform(0.75, 0.95)
        steps.append(EchoStep(delay=delay, trail=trail))
        # Shader between echoes to shift color per ghost layer
        if i < n_echoes - 1 and rng.random() < 0.5 + complexity * 0.3:
            steps.append(_shader_step(rng, complexity, n=1))

    # Final shader pass
    steps.append(_shader_step(rng, complexity))

    source = _random_source(rng, src, use_generators, complexity)
    lane = _make_lane(rng, source=source, steps=steps, n_segments=actual_segments,
                      wants_transition=wants_transition, seg_target=seg_target)

    return _assemble_recipe([lane], rng=rng, complexity=complexity, src=src,
                            seed=seed, wants_post=False)


def _build_warp_focus(
    rng: _random_mod.Random,
    complexity: float,
    src: Optional[Path],
    *,
    n_lanes: Optional[int],
    n_steps: Optional[int],
    n_segments: Optional[int],
    use_transitions: Optional[bool],
    use_generators: Optional[bool],
    target_dur: Optional[float],
    seed: Optional[int],
) -> BrainWipeRecipe:
    """Generator with heavy warp chain, minimal post-processing. Warps are the star."""
    actual_segments = _resolve_segments(rng, complexity, n_segments)
    wants_transition = use_transitions if use_transitions is not None else (
        rng.random() < 0.2 + 0.6 * complexity
    )
    seg_target = _seg_dur_target(target_dur, actual_segments, wants_transition)

    # Heavy warps — 3–4 stacked
    n_warps = max(3, int(3 + complexity))

    source = GeneratorSource(
        n_warps=n_warps,
    )
    if seg_target is not None:
        source = _override_source_dur(source, seg_target)

    # Minimal steps — one time effect + one shader at most
    steps: list[Step] = []
    steps.append(_random_time_step(rng, complexity))
    if rng.random() < 0.3 + complexity * 0.3:
        steps.append(_shader_step(rng, complexity, n=1))

    lane = Lane(
        source=source,
        n_segments=actual_segments,
        recipe=steps,
        sequencing="shuffle" if rng.random() < 0.6 else "concat",
        transition=_random_transition(rng) if wants_transition else None,
    )

    return _assemble_recipe([lane], rng=rng, complexity=complexity, src=src,
                            seed=seed, wants_post=False)


# ─── Archetype registry ─────────────────────────────────────────────────────

def _eligible_crush_sandwich(src, n_lanes, use_generators):
    return n_lanes is None or n_lanes == 1

def _eligible_deep_time(src, n_lanes, use_generators):
    return n_lanes is None or n_lanes == 1

def _eligible_temporal_sandwich(src, n_lanes, use_generators):
    return n_lanes is None or n_lanes == 1

def _eligible_escalation(src, n_lanes, use_generators):
    return True  # adapts to single or multi-lane

def _eligible_polyrhythm(src, n_lanes, use_generators):
    return n_lanes is None or n_lanes >= 2

def _eligible_palimpsest(src, n_lanes, use_generators):
    if n_lanes is not None and n_lanes < 2:
        return False
    return src is not None or use_generators is True

def _eligible_hybrid(src, n_lanes, use_generators):
    if n_lanes is not None and n_lanes < 2:
        return False
    return src is not None and use_generators is not False

def _eligible_grab_bag(src, n_lanes, use_generators):
    return True

def _eligible_stutter(src, n_lanes, use_generators):
    return n_lanes is None or n_lanes == 1

def _eligible_echo_chamber(src, n_lanes, use_generators):
    return n_lanes is None or n_lanes == 1

def _eligible_warp_focus(src, n_lanes, use_generators):
    # Generator-only — always eligible (ignores src)
    return n_lanes is None or n_lanes == 1

_ARCHETYPES: dict[str, tuple] = {
    "crush_sandwich":    (_build_crush_sandwich, _eligible_crush_sandwich),
    "deep_time":         (_build_deep_time, _eligible_deep_time),
    "temporal_sandwich": (_build_temporal_sandwich, _eligible_temporal_sandwich),
    "escalation":        (_build_escalation, _eligible_escalation),
    "polyrhythm":        (_build_polyrhythm, _eligible_polyrhythm),
    "palimpsest":        (_build_palimpsest, _eligible_palimpsest),
    "hybrid":            (_build_hybrid, _eligible_hybrid),
    "grab_bag":          (_build_grab_bag, _eligible_grab_bag),
    "stutter":           (_build_stutter, _eligible_stutter),
    "echo_chamber":      (_build_echo_chamber, _eligible_echo_chamber),
    "warp_focus":        (_build_warp_focus, _eligible_warp_focus),
}


def random_recipe(
    src: Optional[Path] = None,
    *,
    complexity: float = 0.5,
    target_dur: Optional[float] = None,
    n_lanes: Optional[int] = None,
    n_steps: Optional[int] = None,
    n_segments: Optional[int] = None,
    use_transitions: Optional[bool] = None,
    use_generators: Optional[bool] = None,
    seed: Optional[int] = None,
    archetype: Optional[str] = None,
) -> BrainWipeRecipe:
    """Procedurally generate a recipe from all available components.

    Picks a random structural archetype, then fills it in with randomized
    parameters scaled by complexity.

    complexity: 0.0 (simple) to 1.0 (wild). Scales number of lanes, steps,
                segments, and probability of transitions/compositing.
    target_dur: approximate output duration in seconds.
    archetype:  force a specific archetype (crush_sandwich, deep_time,
                temporal_sandwich, escalation, polyrhythm, palimpsest,
                hybrid, grab_bag). None = auto-select from eligible set.
    Granular overrides (n_lanes, n_steps, etc.) pin specific choices;
    everything else is still derived from complexity.
    src: source footage path. None = pure generator/synthetic mode.
    """
    complexity = max(0.0, min(1.0, complexity))
    rng = _random_mod.Random(seed)

    if archetype is not None:
        if archetype not in _ARCHETYPES:
            raise ValueError(
                f"Unknown archetype: {archetype!r}. "
                f"Valid: {', '.join(_ARCHETYPES)}"
            )
        builder, eligible = _ARCHETYPES[archetype]
        if not eligible(src, n_lanes, use_generators):
            raise ValueError(
                f"Archetype {archetype!r} not eligible with "
                f"src={'set' if src else 'None'}, n_lanes={n_lanes}, "
                f"use_generators={use_generators}"
            )
    else:
        eligible_names = [
            name for name, (_, elig) in _ARCHETYPES.items()
            if elig(src, n_lanes, use_generators)
        ]
        archetype = rng.choice(eligible_names)
        builder = _ARCHETYPES[archetype][0]

    return builder(
        rng, complexity, src,
        n_lanes=n_lanes, n_steps=n_steps, n_segments=n_segments,
        use_transitions=use_transitions, use_generators=use_generators,
        target_dur=target_dur, seed=seed,
    )
