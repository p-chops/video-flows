"""
Transition Lab — transition experiments between video clips.

Effects:
  crossfade     Standard fade between two clips (ffmpeg xfade)
  luma-wipe     Grayscale pattern wipe revealing clip B
  whip-pan      Simulated fast camera motion connecting two shots
  static-burst  Short burst of TV noise between clips
  flash         Rapid white flash transition
  sequence      Chain multiple clips with transitions

Usage (Python):
    from pipeline.flows.transition_lab import (
        transition_lab_crossfade, transition_lab_luma_wipe,
        transition_lab_whip_pan, transition_lab_static_burst,
        transition_lab_flash, transition_lab_sequence,
    )
    transition_lab_crossfade(Path("a.mp4"), Path("b.mp4"), duration=1.0)
    transition_lab_luma_wipe(Path("a.mp4"), Path("b.mp4"), pattern="radial")
    transition_lab_whip_pan(Path("a.mp4"), Path("b.mp4"), direction="left")
    transition_lab_static_burst(Path("a.mp4"), Path("b.mp4"), duration=0.3)
    transition_lab_flash(Path("a.mp4"), Path("b.mp4"), duration=0.5)
    transition_lab_sequence([Path("a.mp4"), Path("b.mp4"), Path("c.mp4")])

CLI:
    python -m pipeline.flows.transition_lab crossfade a.mp4 b.mp4 --duration 1.0
    python -m pipeline.flows.transition_lab luma-wipe a.mp4 b.mp4 --pattern radial
    python -m pipeline.flows.transition_lab whip-pan a.mp4 b.mp4 --direction left
    python -m pipeline.flows.transition_lab static-burst a.mp4 b.mp4 --duration 0.3
    python -m pipeline.flows.transition_lab flash a.mp4 b.mp4 --duration 0.5 --decay 3.0
    python -m pipeline.flows.transition_lab sequence a.mp4 b.mp4 c.mp4 --duration 0.5
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from prefect import flow

from ..config import Config
from ..tasks.transition import (
    crossfade, luma_wipe, whip_pan, static_burst, flash,
    transition_sequence, WIPE_PATTERNS,
)


# ── Flows ────────────────────────────────────────────────────────────────────

@flow(name="transition-lab-crossfade", log_prints=True)
def transition_lab_crossfade(
    clip_a: Path,
    clip_b: Path,
    output: Optional[Path] = None,
    duration: float = 1.0,
    cfg: Optional[Config] = None,
) -> Path:
    """Standard crossfade between two clips."""
    c = cfg or Config()
    c.ensure_dirs()
    out = output or c.output_dir / "crossfade.mp4"
    return crossfade(clip_a, clip_b, out, duration=duration, cfg=c)


@flow(name="transition-lab-luma-wipe", log_prints=True)
def transition_lab_luma_wipe(
    clip_a: Path,
    clip_b: Path,
    output: Optional[Path] = None,
    duration: float = 1.0,
    pattern: str = "horizontal",
    softness: float = 0.1,
    angle: float = 0.0,
    seed: Optional[int] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """Luma wipe — grayscale pattern sweeps between clips."""
    c = cfg or Config()
    c.ensure_dirs()
    out = output or c.output_dir / "luma_wipe.mp4"
    return luma_wipe(clip_a, clip_b, out, duration=duration,
                     pattern=pattern, softness=softness, angle=angle,
                     seed=seed, cfg=c)


@flow(name="transition-lab-whip-pan", log_prints=True)
def transition_lab_whip_pan(
    clip_a: Path,
    clip_b: Path,
    output: Optional[Path] = None,
    duration: float = 0.5,
    direction: str = "left",
    blur_strength: float = 0.5,
    cfg: Optional[Config] = None,
) -> Path:
    """Whip pan — fast directional slide with motion blur."""
    c = cfg or Config()
    c.ensure_dirs()
    out = output or c.output_dir / "whip_pan.mp4"
    return whip_pan(clip_a, clip_b, out, duration=duration,
                    direction=direction, blur_strength=blur_strength, cfg=c)


@flow(name="transition-lab-static-burst", log_prints=True)
def transition_lab_static_burst(
    clip_a: Path,
    clip_b: Path,
    output: Optional[Path] = None,
    duration: float = 0.3,
    seed: Optional[int] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """Static burst — short burst of TV noise between clips."""
    c = cfg or Config()
    c.ensure_dirs()
    out = output or c.output_dir / "static_burst.mp4"
    return static_burst(clip_a, clip_b, out, duration=duration, seed=seed, cfg=c)


@flow(name="transition-lab-flash", log_prints=True)
def transition_lab_flash(
    clip_a: Path,
    clip_b: Path,
    output: Optional[Path] = None,
    duration: float = 0.5,
    decay: float = 3.0,
    cfg: Optional[Config] = None,
) -> Path:
    """Flash — rapid white flash fading into clip B."""
    c = cfg or Config()
    c.ensure_dirs()
    out = output or c.output_dir / "flash.mp4"
    return flash(clip_a, clip_b, out, duration=duration, decay=decay, cfg=c)


@flow(name="transition-lab-sequence", log_prints=True)
def transition_lab_sequence(
    clips: list[Path],
    output: Optional[Path] = None,
    transition_type: str = "crossfade",
    duration: float = 1.0,
    seed: Optional[int] = None,
    cfg: Optional[Config] = None,
    **kwargs,
) -> Path:
    """Chain multiple clips with transitions between each pair."""
    c = cfg or Config()
    c.ensure_dirs()
    out = output or c.output_dir / "transition_sequence.mp4"
    return transition_sequence(clips, out, transition_type=transition_type,
                               duration=duration, seed=seed, cfg=c, **kwargs)


# Backward-compatible alias
transition_lab = transition_lab_crossfade


# ── CLI ──────────────────────────────────────────────────────────────────────

def _cli():
    import argparse

    parser = argparse.ArgumentParser(
        description="Transition Lab — transition experiments between clips.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- crossfade ---
    p_xf = sub.add_parser("crossfade", help="Standard crossfade")
    p_xf.add_argument("clip_a", type=Path, help="First clip")
    p_xf.add_argument("clip_b", type=Path, help="Second clip")
    p_xf.add_argument("-o", "--output", type=Path, default=None)
    p_xf.add_argument("--duration", type=float, default=1.0,
                       help="Transition duration in seconds (default: 1.0)")

    # --- luma-wipe ---
    p_lw = sub.add_parser("luma-wipe", help="Luma wipe pattern transition")
    p_lw.add_argument("clip_a", type=Path, help="First clip")
    p_lw.add_argument("clip_b", type=Path, help="Second clip")
    p_lw.add_argument("-o", "--output", type=Path, default=None)
    p_lw.add_argument("--duration", type=float, default=1.0)
    p_lw.add_argument("--pattern", type=str, default="horizontal",
                       choices=WIPE_PATTERNS,
                       help="Wipe pattern (default: horizontal)")
    p_lw.add_argument("--softness", type=float, default=0.1,
                       help="Edge gradient width 0–1 (default: 0.1)")
    p_lw.add_argument("--angle", type=float, default=0.0,
                       help="Wipe angle in degrees for 'directional' pattern. "
                            "0=left→right, 90=top→bottom (default: 0)")
    p_lw.add_argument("--seed", type=int, default=None,
                       help="Random seed for noise pattern")

    # --- whip-pan ---
    p_wp = sub.add_parser("whip-pan", help="Whip pan with motion blur")
    p_wp.add_argument("clip_a", type=Path, help="First clip")
    p_wp.add_argument("clip_b", type=Path, help="Second clip")
    p_wp.add_argument("-o", "--output", type=Path, default=None)
    p_wp.add_argument("--duration", type=float, default=0.5,
                       help="Transition duration in seconds (default: 0.5)")
    p_wp.add_argument("--direction", type=str, default="left",
                       choices=["left", "right", "up", "down"])
    p_wp.add_argument("--blur-strength", type=float, default=0.5,
                       help="Motion blur intensity 0–1 (default: 0.5)")

    # --- static-burst ---
    p_sb = sub.add_parser("static-burst", help="TV noise burst transition")
    p_sb.add_argument("clip_a", type=Path, help="First clip")
    p_sb.add_argument("clip_b", type=Path, help="Second clip")
    p_sb.add_argument("-o", "--output", type=Path, default=None)
    p_sb.add_argument("--duration", type=float, default=0.3,
                       help="Transition duration in seconds (default: 0.3)")
    p_sb.add_argument("--seed", type=int, default=None,
                       help="Random seed for noise generation")

    # --- flash ---
    p_fl = sub.add_parser("flash", help="White flash transition")
    p_fl.add_argument("clip_a", type=Path, help="First clip")
    p_fl.add_argument("clip_b", type=Path, help="Second clip")
    p_fl.add_argument("-o", "--output", type=Path, default=None)
    p_fl.add_argument("--duration", type=float, default=0.5,
                       help="Transition duration in seconds (default: 0.5)")
    p_fl.add_argument("--decay", type=float, default=3.0,
                       help="Flash decay speed (default: 3.0)")

    # --- sequence ---
    p_seq = sub.add_parser("sequence",
                            help="Chain multiple clips with transitions")
    p_seq.add_argument("clips", type=Path, nargs="+",
                        help="Two or more clips to join")
    p_seq.add_argument("-o", "--output", type=Path, default=None)
    p_seq.add_argument("--type", dest="transition_type", type=str,
                        default="crossfade",
                        choices=["crossfade", "luma_wipe", "whip_pan",
                                 "static_burst", "flash"])
    p_seq.add_argument("--duration", type=float, default=1.0)
    p_seq.add_argument("--pattern", type=str, default="horizontal")
    p_seq.add_argument("--softness", type=float, default=0.1)
    p_seq.add_argument("--angle", type=float, default=0.0)
    p_seq.add_argument("--direction", type=str, default="left")
    p_seq.add_argument("--blur-strength", type=float, default=0.5)
    p_seq.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()
    cfg = Config()
    cfg.ensure_dirs()

    if args.command == "crossfade":
        transition_lab_crossfade(
            clip_a=args.clip_a, clip_b=args.clip_b,
            output=args.output, duration=args.duration, cfg=cfg,
        )
    elif args.command == "luma-wipe":
        transition_lab_luma_wipe(
            clip_a=args.clip_a, clip_b=args.clip_b,
            output=args.output, duration=args.duration,
            pattern=args.pattern, softness=args.softness,
            angle=args.angle, seed=args.seed, cfg=cfg,
        )
    elif args.command == "whip-pan":
        transition_lab_whip_pan(
            clip_a=args.clip_a, clip_b=args.clip_b,
            output=args.output, duration=args.duration,
            direction=args.direction,
            blur_strength=args.blur_strength, cfg=cfg,
        )
    elif args.command == "static-burst":
        transition_lab_static_burst(
            clip_a=args.clip_a, clip_b=args.clip_b,
            output=args.output, duration=args.duration,
            seed=args.seed, cfg=cfg,
        )
    elif args.command == "flash":
        transition_lab_flash(
            clip_a=args.clip_a, clip_b=args.clip_b,
            output=args.output, duration=args.duration,
            decay=args.decay, cfg=cfg,
        )
    elif args.command == "sequence":
        transition_lab_sequence(
            clips=args.clips, output=args.output,
            transition_type=args.transition_type,
            duration=args.duration, seed=args.seed, cfg=cfg,
            pattern=args.pattern, softness=args.softness,
            angle=args.angle, direction=args.direction,
            blur_strength=args.blur_strength,
        )


if __name__ == "__main__":
    _cli()
