"""
Non-shader glitch tasks — bitrate crush, codec abuse, etc.

These exploit codec behavior to introduce artifacts rather than
processing frames directly. Fast because they're pure ffmpeg.
"""

from __future__ import annotations

import math
import subprocess
from pathlib import Path
from typing import Optional

from prefect import task

from ..config import Config
from ..ffmpeg import probe


@task(name="bitrate-crush")
def bitrate_crush(
    src: Path,
    dst: Path,
    *,
    crush: float = 0.7,
    downscale: float = 1.0,
    codec: str = "libx264",
    cfg: Optional[Config] = None,
) -> Path:
    """
    Encode at aggressively low quality to introduce compression artifacts,
    then re-encode at normal quality to bake them in.

    Uses constant-quality mode (CRF/QP) so every frame is equally crushed —
    no rate-control oscillation, no bright flashes from starved keyframes.
    Large GOP preserves temporal drift as quantization errors accumulate
    across long P-frame chains.

    crush:     0.0 = mild artifacts, 1.0 = maximum destruction
               Maps to QP 30–51 for libx264, q:v 10–31 for mpeg2/mpeg4.
    downscale: resolution reduction factor before crushing.
               1.0 = native, 2.0 = half, 4.0 = quarter, etc.
               Lower resolution → bigger blocks when scaled back up.
    codec:     codec for the dirty pass. Options:
               - "libx264"  — classic macroblocking, DCT ringing
               - "mpeg2video" — chunkier blocks, VHS-era feel
               - "mpeg4"    — DivX-style, wriggly edges
    """
    c = cfg or Config()
    info = probe(src, c)

    total_frames = max(2, math.ceil(info.fps * info.duration) + 1)

    # Map crush 0–1 to codec quality parameter (higher = worse)
    if codec == "libx264":
        qp_val = int(30 + crush * 21)
        quality_args = ["-qp", str(qp_val)]
    else:
        qv_val = int(10 + crush * 21)
        quality_args = ["-q:v", str(qv_val)]

    # Downscale + crush + nearest-neighbor upscale in one filter chain.
    # Doing the round-trip in the dirty pass bakes the blocky pixels into
    # full-res frames so the clean re-encode just preserves them.
    vf_filters = []
    if downscale > 1.0:
        small_w = max(2, int(info.width / downscale)) // 2 * 2   # keep even
        small_h = max(2, int(info.height / downscale)) // 2 * 2
        vf_filters = ["-vf",
            f"scale={small_w}:{small_h}:flags=bilinear,"
            f"scale={info.width}:{info.height}:flags=neighbor"]

    # Dirty encode — fixed QP, huge GOP, scene detection disabled.
    crushed = dst.with_suffix(".crushed.mp4")
    dirty_cmd = [
        c.ffmpeg_bin, "-y", "-loglevel", c.ffmpeg_loglevel,
        "-i", str(src),
        "-an",
        *vf_filters,
        "-c:v", codec,
        *quality_args,
        "-g", str(total_frames),
        "-keyint_min", str(total_frames),
        "-sc_threshold", "0",
        "-bf", "0",
        "-pix_fmt", c.default_pix_fmt,
        str(crushed),
    ]
    subprocess.run(dirty_cmd, check=True)

    # Clean re-encode — bake artifacts into a proper file
    clean_cmd = [
        c.ffmpeg_bin, "-y", "-loglevel", c.ffmpeg_loglevel,
        "-i", str(crushed),
        "-an",
        *c.encode_args(),
        str(dst),
    ]
    subprocess.run(clean_cmd, check=True)

    # Clean up intermediate
    crushed.unlink(missing_ok=True)

    return dst
