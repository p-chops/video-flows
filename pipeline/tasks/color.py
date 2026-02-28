"""
Color correction tasks — normalization, levels, etc.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
from prefect import task

from ..config import Config
from ..ffmpeg import probe, read_frames


@task(name="normalize-levels")
def normalize_levels(
    src: Path,
    dst: Path,
    *,
    black_point: float = 0.01,
    white_point: float = 0.99,
    cfg: Optional[Config] = None,
) -> Path:
    """
    Normalize video levels — clips the darkest and brightest percentiles
    and stretches the remaining range to fill 0–255.

    Uses ffmpeg's colorlevels filter: a static per-pixel LUT operation
    with negligible overhead beyond decode + encode.

    black_point: fraction of darkest values to clip to black (default 0.01).
    white_point: fraction of brightest values to clip to white (default 0.01).
                 Note: this is the fraction clipped from the TOP, not a
                 percentile — so 0.01 means clip the top 1%.
    """
    c = cfg or Config()

    # colorlevels rimin/rimax are input intensity levels [0.0–1.0].
    # rimin = black point (values below become black)
    # rimax = white point (values above become white)
    # Everything between is stretched to fill 0–255.
    bp = black_point
    wp = white_point
    vf = (
        f"colorlevels="
        f"rimin={bp}:gimin={bp}:bimin={bp}:"
        f"rimax={wp}:gimax={wp}:bimax={wp}"
    )

    subprocess.run([
        c.ffmpeg_bin, "-y", "-loglevel", c.ffmpeg_loglevel,
        "-i", str(src),
        "-vf", vf,
        "-an",
        *c.encode_args(),
        str(dst),
    ], check=True)

    return dst


def _probe_brightness(src: Path, cfg: Config, n_samples: int = 10) -> float:
    """Sample frames and return average brightness in [0, 1]."""
    info = probe(src, cfg)
    total_frames = int(info.duration * info.fps)
    sample_interval = max(1, total_frames // n_samples)

    total = 0.0
    count = 0
    for i, frame in enumerate(read_frames(src, cfg=cfg)):
        if i % sample_interval == 0:
            total += float(np.mean(frame)) / 255.0
            count += 1
    return total / max(count, 1)


@task(name="auto-levels")
def auto_levels(
    src: Path,
    dst: Path,
    *,
    target: float = 0.45,
    max_gamma: float = 3.0,
    threshold: float = 0.05,
    cfg: Optional[Config] = None,
) -> Path:
    """
    Probe average brightness and apply gamma correction toward a target.

    Measures the clip's average luma, computes a gamma that would push it
    toward `target` (0–1 scale), and applies via ffmpeg's eq filter.
    Skips the encode if no correction is needed (avg already near target).

    target:    desired average brightness in [0, 1] (default 0.45)
    max_gamma: clamp gamma to [1/max_gamma, max_gamma] (default 3.0)
    threshold: skip correction if |gamma - 1| < threshold (default 0.05)
    """
    c = cfg or Config()

    avg = _probe_brightness(src, c)

    # Compute gamma: eq filter applies out = in^(1/gamma)
    # We want avg^(1/gamma) = target, so gamma = log(avg) / log(target)
    if avg < 0.005:
        gamma = max_gamma
    elif avg > 0.995:
        gamma = 1.0 / max_gamma
    else:
        gamma = np.log(avg) / np.log(target)
        gamma = float(np.clip(gamma, 1.0 / max_gamma, max_gamma))

    if abs(gamma - 1.0) < threshold:
        shutil.copy2(src, dst)
        return dst

    subprocess.run([
        c.ffmpeg_bin, "-y", "-loglevel", c.ffmpeg_loglevel,
        "-i", str(src),
        "-vf", f"eq=gamma={gamma:.3f}",
        "-an",
        *c.encode_args(),
        str(dst),
    ], check=True)

    return dst
