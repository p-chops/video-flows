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
                    duration=segment_dur,
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
                source=GeneratorSource(duration=segment_dur, n_warps=n_warps),
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
                source=GeneratorSource(duration=segment_dur, n_warps=n_warps),
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
