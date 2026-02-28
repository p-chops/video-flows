"""
Spatial and colour transform tasks — mirror, zoom, invert, hue_shift, saturate.

Pure ffmpeg filter-graph operations — fast, no frame-level Python.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

from prefect import task

from ..config import Config


@task(name="mirror")
def mirror(
    src: Path,
    dst: Path,
    *,
    axis: str = "horizontal",
    cfg: Optional[Config] = None,
) -> Path:
    """
    Mirror video along an axis.

    axis: "horizontal" (left-right flip) or "vertical" (top-bottom flip).
    """
    c = cfg or Config()
    vf = "hflip" if axis == "horizontal" else "vflip"

    subprocess.run([
        c.ffmpeg_bin, "-y", "-loglevel", c.ffmpeg_loglevel,
        "-i", str(src),
        "-vf", vf,
        "-an",
        *c.encode_args(),
        str(dst),
    ], check=True)

    return dst


@task(name="zoom")
def zoom(
    src: Path,
    dst: Path,
    *,
    factor: float = 1.5,
    center_x: float = 0.5,
    center_y: float = 0.5,
    cfg: Optional[Config] = None,
) -> Path:
    """
    Crop-zoom into video — enlarges a region and scales back to original size.

    factor:   zoom level (1.0 = no zoom, 2.0 = 2x zoom into center).
    center_x: horizontal center of zoom region (0.0–1.0).
    center_y: vertical center of zoom region (0.0–1.0).
    """
    c = cfg or Config()
    from ..ffmpeg import probe as ffprobe
    info = ffprobe(src, c)
    w, h = info.width, info.height

    # Compute integer crop region, then scale back to original (even) dims
    cw = max(2, int(w / factor) & ~1)  # even width
    ch = max(2, int(h / factor) & ~1)  # even height
    cx = int((w - cw) * center_x)
    cy = int((h - ch) * center_y)
    vf = f"crop={cw}:{ch}:{cx}:{cy},scale={w}:{h}:flags=lanczos"

    subprocess.run([
        c.ffmpeg_bin, "-y", "-loglevel", c.ffmpeg_loglevel,
        "-i", str(src),
        "-vf", vf,
        "-an",
        *c.encode_args(),
        str(dst),
    ], check=True)

    return dst


@task(name="invert")
def invert(
    src: Path,
    dst: Path,
    *,
    cfg: Optional[Config] = None,
) -> Path:
    """Invert video colours (negative)."""
    c = cfg or Config()

    subprocess.run([
        c.ffmpeg_bin, "-y", "-loglevel", c.ffmpeg_loglevel,
        "-i", str(src),
        "-vf", "negate",
        "-an",
        *c.encode_args(),
        str(dst),
    ], check=True)

    return dst


@task(name="hue-shift")
def hue_shift(
    src: Path,
    dst: Path,
    *,
    degrees: float = 90.0,
    cfg: Optional[Config] = None,
) -> Path:
    """
    Rotate hue by a fixed angle in degrees.

    degrees: hue rotation (0–360). 90 = warm→cool, 180 = complementary.
    """
    c = cfg or Config()

    subprocess.run([
        c.ffmpeg_bin, "-y", "-loglevel", c.ffmpeg_loglevel,
        "-i", str(src),
        "-vf", f"hue=h={degrees}",
        "-an",
        *c.encode_args(),
        str(dst),
    ], check=True)

    return dst


@task(name="saturate")
def saturate(
    src: Path,
    dst: Path,
    *,
    amount: float = 2.0,
    cfg: Optional[Config] = None,
) -> Path:
    """
    Adjust saturation.

    amount: multiplier (0.0 = grayscale, 1.0 = unchanged, 2.0+ = oversaturated).
    """
    c = cfg or Config()

    subprocess.run([
        c.ffmpeg_bin, "-y", "-loglevel", c.ffmpeg_loglevel,
        "-i", str(src),
        "-vf", f"hue=s={amount}",
        "-an",
        *c.encode_args(),
        str(dst),
    ], check=True)

    return dst
