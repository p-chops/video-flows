"""
Prefect tasks for sequencing — concatenation, interleaving, static generation.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np
from prefect import task

from ..config import Config
import subprocess

from ..ffmpeg import probe, concat_files, FrameWriter, VideoInfo


@task(name="concat-clips")
def concat_clips(clips: list[Path], dst: Path,
                 cfg: Optional[Config] = None) -> Path:
    """
    Concatenate clips in order using ffmpeg's concat demuxer.
    Returns the output path.
    """
    concat_files(clips, dst, cfg)
    return dst


@task(name="shuffle-clips")
def shuffle_clips(clips: list[Path], dst: Path,
                  seed: Optional[int] = None,
                  cfg: Optional[Config] = None) -> Path:
    """
    Shuffle clips into a random order and concatenate.
    """
    rng = random.Random(seed)
    order = list(clips)
    rng.shuffle(order)
    concat_files(order, dst, cfg)
    return dst


@task(name="interleave-clips")
def interleave_clips(groups: list[list[Path]], dst: Path,
                     cfg: Optional[Config] = None) -> Path:
    """
    Interleave clips from multiple groups: A1, B1, A2, B2, ...
    Groups can be different lengths; stops when the shortest is exhausted.
    """
    min_len = min(len(g) for g in groups)
    ordered = []
    for i in range(min_len):
        for g in groups:
            ordered.append(g[i])
    concat_files(ordered, dst, cfg)
    return dst


@task(name="generate-static")
def generate_static(dst: Path, duration: float,
                    width: int = 1920, height: int = 1080,
                    fps: float = 30.0,
                    cfg: Optional[Config] = None) -> Path:
    """
    Generate a clip of static (random noise) at the given resolution.
    Useful as filler, texture source, or compositing layer.
    """
    c = cfg or Config()
    total_frames = int(duration * fps)
    info = VideoInfo(width=width, height=height, fps=fps, duration=duration, codec=c.default_codec)

    with FrameWriter(dst, info, cfg=c) as writer:
        for _ in range(total_frames):
            frame = np.random.randint(0, 256, (height, width, 3),
                                      dtype=np.uint8)
            writer.write(frame)
    return dst


@task(name="generate-solid")
def generate_solid(dst: Path, duration: float,
                   color: tuple[int, int, int] = (0, 0, 0),
                   width: int = 1920, height: int = 1080,
                   fps: float = 30.0,
                   cfg: Optional[Config] = None) -> Path:
    """
    Generate a solid-colour clip (default black).
    Useful as a background layer for compositing.
    Uses ffmpeg's color source filter — sub-second for any duration.
    """
    c = cfg or Config()
    hex_color = f"0x{color[0]:02x}{color[1]:02x}{color[2]:02x}"
    subprocess.run([
        c.ffmpeg_bin, "-y", "-loglevel", c.ffmpeg_loglevel,
        "-f", "lavfi", "-i",
        f"color=c={hex_color}:s={width}x{height}:r={fps}:d={duration}",
        "-an",
        "-c:v", "libx264", "-preset", "ultrafast", "-qp", "51",
        "-pix_fmt", c.default_pix_fmt,
        str(dst),
    ], check=True)
    return dst


@task(name="repeat-clip")
def repeat_clip(src: Path, dst: Path, times: int = 2,
                cfg: Optional[Config] = None) -> Path:
    """
    Repeat a clip N times via concat.
    """
    clips = [src] * times
    concat_files(clips, dst, cfg)
    return dst
