"""
Color correction tasks — normalization, levels, etc.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from prefect import task

from ..config import Config
from ..ffmpeg import FrameWriter, probe, read_frames


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
    Normalize video levels — stretches luminance so the darkest pixels
    hit black and the brightest hit white.

    Two-pass: first pass finds the percentile-based black/white points
    across a sample of frames, second pass applies the correction.

    black_point / white_point: percentiles (0–1) for clipping.
    Using 0.01/0.99 avoids outlier pixels blowing out the range.
    """
    c = cfg or Config()
    info = probe(src, c)

    # --- Pass 1: sample frames to find luminance range ---
    luma_mins = []
    luma_maxs = []
    frame_count = 0
    # Sample every Nth frame to keep pass 1 fast
    sample_interval = max(1, int(info.fps))  # ~1 sample per second

    for frame in read_frames(src, c):
        frame_count += 1
        if frame_count % sample_interval != 0:
            continue
        # Compute luminance
        luma = (
            frame[:, :, 0].astype(np.float32) * 0.299
            + frame[:, :, 1].astype(np.float32) * 0.587
            + frame[:, :, 2].astype(np.float32) * 0.114
        )
        luma_mins.append(np.percentile(luma, black_point * 100))
        luma_maxs.append(np.percentile(luma, white_point * 100))

    if not luma_mins:
        # Fallback: no frames sampled, just copy
        import shutil
        shutil.copy2(src, dst)
        return dst

    # Use median of sampled percentiles for stability
    lo = np.median(luma_mins)
    hi = np.median(luma_maxs)

    # Avoid division by zero / no-op if range is already full
    if hi - lo < 5.0:
        import shutil
        shutil.copy2(src, dst)
        return dst

    scale = 255.0 / (hi - lo)
    print(f"  normalize: lo={lo:.1f} hi={hi:.1f} scale={scale:.2f}x")

    # --- Pass 2: apply correction ---
    with FrameWriter(dst, info, cfg=c) as writer:
        for frame in read_frames(src, c):
            f = frame.astype(np.float32)
            f = (f - lo) * scale
            np.clip(f, 0, 255, out=f)
            writer.write(f.astype(np.uint8))

    return dst
