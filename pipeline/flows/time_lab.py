"""
Time Lab — temporal manipulation experiments.

Effects:
  scrub    — virtual playhead wanders through source at varying speeds
  drift    — fixed-length loop window slides slowly through source
  pingpong — short subsegment plays forward-backward continuously
  echo     — ghostly motion trails via frame accumulation
  patch    — temporal patchwork, random crops reveal 'now' over frozen past

Usage (Python):
    from pipeline.flows.time_lab import time_lab_scrub, time_lab_drift, time_lab_pingpong, time_lab_echo, time_lab_patch
    time_lab_scrub(Path("src.mp4"), smoothness=2.0, intensity=0.5, seed=42)
    time_lab_drift(Path("src.mp4"), loop_dur=0.5, seed=1)
    time_lab_pingpong(Path("src.mp4"), window=0.5, seed=1)
    time_lab_echo(Path("src.mp4"), delay=0.5, trail=0.8)
    time_lab_patch(Path("src.mp4"), patch_min=0.05, patch_max=0.4, seed=42)

CLI:
    python -m pipeline.flows.time_lab scrub src.mp4 --intensity 0.5 --seed 42
    python -m pipeline.flows.time_lab drift src.mp4 --loop-dur 0.5 --seed 1
    python -m pipeline.flows.time_lab pingpong src.mp4 --window 0.5 --seed 1
    python -m pipeline.flows.time_lab echo src.mp4 --delay 0.5 --trail 0.8
    python -m pipeline.flows.time_lab patch src.mp4 --patch-min 0.05 --patch-max 0.4 --seed 42
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from prefect import flow

from ..config import Config
from ..tasks.time import (
    drift_loop, echo_trail, ping_pong, time_patch, time_scrub, temporal_fft,
    temporal_gradient, temporal_median, axis_swap,
)


# ─── Flows ────────────────────────────────────────────────────────────────────

@flow(name="time-lab-scrub", log_prints=True)
def time_lab_scrub(
    src: Path,
    output: Optional[Path] = None,
    smoothness: float = 2.0,
    intensity: float = 0.5,
    seed: Optional[int] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """Random temporal scrubbing — playhead wanders at varying speeds."""
    c = cfg or Config()
    c.ensure_dirs()
    out = output or c.output_dir / "time_scrub.mp4"
    return time_scrub(src, out, smoothness=smoothness, intensity=intensity,
                      seed=seed, cfg=c)


@flow(name="time-lab-drift", log_prints=True)
def time_lab_drift(
    src: Path,
    output: Optional[Path] = None,
    loop_dur: float = 0.5,
    drift: Optional[float] = None,
    seed: Optional[int] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """Drift loop — looping window slides through the source."""
    c = cfg or Config()
    c.ensure_dirs()
    out = output or c.output_dir / "drift_loop.mp4"
    return drift_loop(src, out, loop_dur=loop_dur, drift=drift,
                      seed=seed, cfg=c)


@flow(name="time-lab-pingpong", log_prints=True)
def time_lab_pingpong(
    src: Path,
    output: Optional[Path] = None,
    window: float = 0.5,
    seed: Optional[int] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """Ping-pong — short subsegment breathes forward-backward."""
    c = cfg or Config()
    c.ensure_dirs()
    out = output or c.output_dir / "ping_pong.mp4"
    return ping_pong(src, out, window=window, seed=seed, cfg=c)


@flow(name="time-lab-echo", log_prints=True)
def time_lab_echo(
    src: Path,
    output: Optional[Path] = None,
    delay: float = 0.0,
    trail: float = 0.8,
    cfg: Optional[Config] = None,
) -> Path:
    """Echo trails — ghostly temporal echoes with configurable delay."""
    c = cfg or Config()
    c.ensure_dirs()
    out = output or c.output_dir / "echo_trail.mp4"
    return echo_trail(src, out, delay=delay, trail=trail, cfg=c)


@flow(name="time-lab-patch", log_prints=True)
def time_lab_patch(
    src: Path,
    output: Optional[Path] = None,
    patch_min: float = 0.05,
    patch_max: float = 0.4,
    seed: Optional[int] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """Temporal patchwork — random crops reveal 'now', rest frozen in past."""
    c = cfg or Config()
    c.ensure_dirs()
    out = output or c.output_dir / "time_patch.mp4"
    return time_patch(src, out, patch_min=patch_min, patch_max=patch_max,
                      seed=seed, cfg=c)


@flow(name="time-lab-temporal-fft", log_prints=True)
def time_lab_temporal_fft(
    src: Path,
    output: Optional[Path] = None,
    filter_type: str = "low_pass",
    cutoff_low: float = 0.1,
    cutoff_high: float = 0.5,
    preserve_dc: bool = True,
    seed: Optional[int] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """Temporal FFT filtering — manipulate frequency content along time axis."""
    c = cfg or Config()
    c.ensure_dirs()
    out = output or c.output_dir / "temporal_fft.mp4"
    return temporal_fft(src, out, filter_type=filter_type, cutoff_low=cutoff_low,
                        cutoff_high=cutoff_high, preserve_dc=preserve_dc,
                        seed=seed, cfg=c)


@flow(name="time-lab-temporal-gradient", log_prints=True)
def time_lab_temporal_gradient(
    src: Path,
    output: Optional[Path] = None,
    order: int = 1,
    seed: Optional[int] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """Temporal gradient — per-pixel temporal derivative."""
    c = cfg or Config()
    c.ensure_dirs()
    out = output or c.output_dir / "temporal_gradient.mp4"
    return temporal_gradient(src, out, order=order, seed=seed, cfg=c)


@flow(name="time-lab-temporal-median", log_prints=True)
def time_lab_temporal_median(
    src: Path,
    output: Optional[Path] = None,
    window: int = 7,
    seed: Optional[int] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """Temporal median — rolling median removes transient motion."""
    c = cfg or Config()
    c.ensure_dirs()
    out = output or c.output_dir / "temporal_median.mp4"
    return temporal_median(src, out, window=window, seed=seed, cfg=c)


@flow(name="time-lab-axis-swap", log_prints=True)
def time_lab_axis_swap(
    src: Path,
    output: Optional[Path] = None,
    axis: str = "horizontal",
    seed: Optional[int] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """Axis swap — view the frame volume from the side."""
    c = cfg or Config()
    c.ensure_dirs()
    out = output or c.output_dir / "axis_swap.mp4"
    return axis_swap(src, out, axis=axis, seed=seed, cfg=c)


# Keep backward-compatible alias
time_lab = time_lab_scrub


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _cli():
    import argparse

    parser = argparse.ArgumentParser(
        description="Time Lab — temporal manipulation experiments.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- scrub ---
    p_scrub = sub.add_parser("scrub",
                             help="Random temporal scrubbing")
    p_scrub.add_argument("src", type=Path,
                         help="Source video file")
    p_scrub.add_argument("-o", "--output", type=Path, default=None,
                         help="Output path (default: output/time_scrub.mp4)")
    p_scrub.add_argument("--smoothness", type=float, default=2.0,
                         help="Temporal scale of speed changes in seconds (default: 2.0)")
    p_scrub.add_argument("--intensity", type=float, default=0.5,
                         help="Speed variation magnitude, 0=normal 1=wild (default: 0.5)")
    p_scrub.add_argument("--seed", type=int, default=None,
                         help="Random seed for reproducibility")

    # --- drift ---
    p_drift = sub.add_parser("drift",
                             help="Drift loop — sliding loop window")
    p_drift.add_argument("src", type=Path,
                         help="Source video file")
    p_drift.add_argument("-o", "--output", type=Path, default=None,
                         help="Output path (default: output/drift_loop.mp4)")
    p_drift.add_argument("--loop-dur", type=float, default=0.5,
                         help="Seconds per loop cycle (default: 0.5)")
    p_drift.add_argument("--drift", type=float, default=None,
                         help="Seconds of drift per cycle; negative=backward (default: auto)")
    p_drift.add_argument("--seed", type=int, default=None,
                         help="Random seed (start position + auto-drift direction)")

    # --- pingpong ---
    p_pp = sub.add_parser("pingpong",
                          help="Ping-pong — breathing forward-backward loop")
    p_pp.add_argument("src", type=Path,
                      help="Source video file")
    p_pp.add_argument("-o", "--output", type=Path, default=None,
                      help="Output path (default: output/ping_pong.mp4)")
    p_pp.add_argument("--window", type=float, default=0.5,
                      help="Seconds of source to ping-pong (default: 0.5)")
    p_pp.add_argument("--seed", type=int, default=None,
                      help="Random seed (controls which subsegment)")

    # --- echo ---
    p_echo = sub.add_parser("echo",
                            help="Echo trails — ghostly motion smear")
    p_echo.add_argument("src", type=Path,
                        help="Source video file")
    p_echo.add_argument("-o", "--output", type=Path, default=None,
                        help="Output path (default: output/echo_trail.mp4)")
    p_echo.add_argument("--delay", type=float, default=0.0,
                        help="Echo delay in seconds; 0=motion blur (default: 0.0)")
    p_echo.add_argument("--trail", type=float, default=0.8,
                        help="Echo strength/feedback 0-1, higher=more ghost (default: 0.8)")

    # --- patch ---
    p_patch = sub.add_parser("patch",
                             help="Temporal patchwork — random crop reveals")
    p_patch.add_argument("src", type=Path,
                         help="Source video file")
    p_patch.add_argument("-o", "--output", type=Path, default=None,
                         help="Output path (default: output/time_patch.mp4)")
    p_patch.add_argument("--patch-min", type=float, default=0.05,
                         help="Min patch size as fraction of frame (default: 0.05)")
    p_patch.add_argument("--patch-max", type=float, default=0.4,
                         help="Max patch size as fraction of frame (default: 0.4)")
    p_patch.add_argument("--seed", type=int, default=None,
                         help="Random seed for reproducibility")

    # --- temporal-fft ---
    p_fft = sub.add_parser("temporal-fft",
                           help="Temporal FFT filtering")
    p_fft.add_argument("src", type=Path,
                       help="Source video file")
    p_fft.add_argument("-o", "--output", type=Path, default=None,
                       help="Output path (default: output/temporal_fft.mp4)")
    p_fft.add_argument("--filter-type", type=str, default="low_pass",
                       choices=["low_pass", "high_pass", "band_pass", "notch"],
                       help="Filter mode (default: low_pass)")
    p_fft.add_argument("--cutoff-low", type=float, default=0.1,
                       help="Low cutoff freq 0-1 (default: 0.1)")
    p_fft.add_argument("--cutoff-high", type=float, default=0.5,
                       help="High cutoff freq 0-1 (default: 0.5)")
    p_fft.add_argument("--no-preserve-dc", action="store_true",
                       help="Don't preserve DC bin (allows brightness shift)")
    p_fft.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")

    # --- temporal-gradient ---
    p_grad = sub.add_parser("temporal-gradient",
                            help="Temporal gradient — motion-only output")
    p_grad.add_argument("src", type=Path,
                        help="Source video file")
    p_grad.add_argument("-o", "--output", type=Path, default=None,
                        help="Output path (default: output/temporal_gradient.mp4)")
    p_grad.add_argument("--order", type=int, default=1, choices=[1, 2],
                        help="Derivative order: 1=velocity, 2=acceleration (default: 1)")
    p_grad.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")

    # --- temporal-median ---
    p_med = sub.add_parser("temporal-median",
                           help="Temporal median — removes transient motion")
    p_med.add_argument("src", type=Path,
                       help="Source video file")
    p_med.add_argument("-o", "--output", type=Path, default=None,
                       help="Output path (default: output/temporal_median.mp4)")
    p_med.add_argument("--window", type=int, default=7,
                       help="Median kernel size in frames, odd (default: 7)")
    p_med.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")

    # --- axis-swap ---
    p_swap = sub.add_parser("axis-swap",
                            help="Axis swap — view volume from the side")
    p_swap.add_argument("src", type=Path,
                        help="Source video file")
    p_swap.add_argument("-o", "--output", type=Path, default=None,
                        help="Output path (default: output/axis_swap.mp4)")
    p_swap.add_argument("--axis", type=str, default="horizontal",
                        choices=["horizontal", "vertical"],
                        help="Which axis to swap with time (default: horizontal)")
    p_swap.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")

    args = parser.parse_args()
    cfg = Config()
    cfg.ensure_dirs()

    if args.command == "scrub":
        time_lab_scrub(
            src=args.src, output=args.output,
            smoothness=args.smoothness, intensity=args.intensity,
            seed=args.seed, cfg=cfg,
        )
    elif args.command == "drift":
        time_lab_drift(
            src=args.src, output=args.output,
            loop_dur=args.loop_dur, drift=args.drift,
            seed=args.seed, cfg=cfg,
        )
    elif args.command == "pingpong":
        time_lab_pingpong(
            src=args.src, output=args.output,
            window=args.window, seed=args.seed,
            cfg=cfg,
        )
    elif args.command == "echo":
        time_lab_echo(
            src=args.src, output=args.output,
            delay=args.delay, trail=args.trail, cfg=cfg,
        )
    elif args.command == "patch":
        time_lab_patch(
            src=args.src, output=args.output,
            patch_min=args.patch_min, patch_max=args.patch_max,
            seed=args.seed, cfg=cfg,
        )
    elif args.command == "temporal-fft":
        time_lab_temporal_fft(
            src=args.src, output=args.output,
            filter_type=args.filter_type,
            cutoff_low=args.cutoff_low, cutoff_high=args.cutoff_high,
            preserve_dc=not args.no_preserve_dc,
            seed=args.seed, cfg=cfg,
        )
    elif args.command == "temporal-gradient":
        time_lab_temporal_gradient(
            src=args.src, output=args.output,
            order=args.order, seed=args.seed, cfg=cfg,
        )
    elif args.command == "temporal-median":
        time_lab_temporal_median(
            src=args.src, output=args.output,
            window=args.window, seed=args.seed, cfg=cfg,
        )
    elif args.command == "axis-swap":
        time_lab_axis_swap(
            src=args.src, output=args.output,
            axis=args.axis, seed=args.seed, cfg=cfg,
        )


if __name__ == "__main__":
    _cli()
