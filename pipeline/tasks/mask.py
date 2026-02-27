"""
Prefect tasks for generating masks from video frames.

Masks are single-channel (grayscale) videos where white = include,
black = exclude. They can be passed to composite tasks as alpha layers.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from prefect import task
from ..cache import FILE_VALIDATED_INPUTS
from prefect.logging import get_run_logger

from ..config import Config
from ..ffmpeg import probe, read_frames, FrameWriter


@task(name="luma-mask", cache_policy=FILE_VALIDATED_INPUTS, persist_result=True)
def luma_mask(src: Path, dst: Path,
              threshold: float = 0.5,
              invert: bool = False,
              blur: int = 0,
              cfg: Optional[Config] = None) -> Path:
    """
    Generate a binary mask based on luminance.

    threshold: 0-1, pixels brighter than this → white.
    invert:    flip the mask.
    blur:      Gaussian blur kernel size (0 = none, must be odd).
    """
    c = cfg or Config()
    log = get_run_logger()
    info = probe(src, c)
    total_frames = int(info.duration * info.fps) if info.duration > 0 else 0
    t0 = time.monotonic()
    last_log = t0

    with FrameWriter(dst, info.width, info.height, fps=info.fps, cfg=c) as writer:
        for frame_num, frame in enumerate(read_frames(src, cfg=c)):
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(
                gray, int(threshold * 255), 255, cv2.THRESH_BINARY
            )
            if invert:
                binary = 255 - binary
            if blur > 0:
                k = blur if blur % 2 == 1 else blur + 1
                binary = cv2.GaussianBlur(binary, (k, k), 0)
            mask_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
            writer.write(mask_rgb)

            now = time.monotonic()
            if now - last_log >= 5.0:
                elapsed = now - t0
                pct = (frame_num / total_frames * 100) if total_frames > 0 else 0
                log.info("luma-mask: %.0f%% frame=%d/%d fps=%.1f elapsed=%.1fs",
                         pct, frame_num, total_frames,
                         frame_num / elapsed if elapsed > 0 else 0, elapsed)
                last_log = now

    elapsed = time.monotonic() - t0
    log.info("luma-mask: done in %.1fs", elapsed)
    return dst


@task(name="edge-mask", cache_policy=FILE_VALIDATED_INPUTS, persist_result=True)
def edge_mask(src: Path, dst: Path,
              low: int = 50, high: int = 150,
              dilate: int = 0,
              cfg: Optional[Config] = None) -> Path:
    """
    Generate an edge-detection mask using Canny.

    low/high: Canny thresholds.
    dilate:   dilation iterations to thicken edges (0 = none).
    """
    c = cfg or Config()
    log = get_run_logger()
    info = probe(src, c)
    total_frames = int(info.duration * info.fps) if info.duration > 0 else 0
    t0 = time.monotonic()
    last_log = t0

    with FrameWriter(dst, info.width, info.height, fps=info.fps, cfg=c) as writer:
        for frame_num, frame in enumerate(read_frames(src, cfg=c)):
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, low, high)
            if dilate > 0:
                kernel = np.ones((3, 3), np.uint8)
                edges = cv2.dilate(edges, kernel, iterations=dilate)
            mask_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            writer.write(mask_rgb)

            now = time.monotonic()
            if now - last_log >= 5.0:
                elapsed = now - t0
                pct = (frame_num / total_frames * 100) if total_frames > 0 else 0
                log.info("edge-mask: %.0f%% frame=%d/%d fps=%.1f elapsed=%.1fs",
                         pct, frame_num, total_frames,
                         frame_num / elapsed if elapsed > 0 else 0, elapsed)
                last_log = now

    elapsed = time.monotonic() - t0
    log.info("edge-mask: done in %.1fs", elapsed)
    return dst


@task(name="motion-mask", cache_policy=FILE_VALIDATED_INPUTS, persist_result=True)
def motion_mask(src: Path, dst: Path,
                threshold: int = 25,
                blur: int = 5,
                cfg: Optional[Config] = None) -> Path:
    """
    Generate a mask from frame-to-frame motion (absolute difference).

    threshold: pixel diff threshold (0-255).
    blur:      Gaussian blur kernel size for smoothing.
    """
    c = cfg or Config()
    log = get_run_logger()
    info = probe(src, c)
    total_frames = int(info.duration * info.fps) if info.duration > 0 else 0
    t0 = time.monotonic()
    last_log = t0
    prev_gray = None

    with FrameWriter(dst, info.width, info.height, fps=info.fps, cfg=c) as writer:
        for frame_num, frame in enumerate(read_frames(src, cfg=c)):
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            if prev_gray is None:
                mask = np.zeros_like(gray)
            else:
                diff = cv2.absdiff(gray, prev_gray)
                _, mask = cv2.threshold(diff, threshold, 255,
                                        cv2.THRESH_BINARY)
                if blur > 0:
                    k = blur if blur % 2 == 1 else blur + 1
                    mask = cv2.GaussianBlur(mask, (k, k), 0)
            prev_gray = gray
            mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            writer.write(mask_rgb)

            now = time.monotonic()
            if now - last_log >= 5.0:
                elapsed = now - t0
                pct = (frame_num / total_frames * 100) if total_frames > 0 else 0
                log.info("motion-mask: %.0f%% frame=%d/%d fps=%.1f elapsed=%.1fs",
                         pct, frame_num, total_frames,
                         frame_num / elapsed if elapsed > 0 else 0, elapsed)
                last_log = now

    elapsed = time.monotonic() - t0
    log.info("motion-mask: done in %.1fs", elapsed)
    return dst


@task(name="chroma-mask", cache_policy=FILE_VALIDATED_INPUTS, persist_result=True)
def chroma_mask(src: Path, dst: Path,
                hue_center: int = 60,
                hue_range: int = 20,
                sat_min: int = 50,
                invert: bool = False,
                blur: int = 3,
                cfg: Optional[Config] = None) -> Path:
    """
    Generate a mask based on hue (chroma key).

    hue_center: target hue in OpenCV scale (0-179).
    hue_range:  +/- range around center.
    sat_min:    minimum saturation to qualify.
    invert:     flip the mask (keep everything except the target colour).
    """
    c = cfg or Config()
    log = get_run_logger()
    info = probe(src, c)
    total_frames = int(info.duration * info.fps) if info.duration > 0 else 0
    t0 = time.monotonic()
    last_log = t0

    lower = np.array([max(0, hue_center - hue_range), sat_min, 50])
    upper = np.array([min(179, hue_center + hue_range), 255, 255])

    with FrameWriter(dst, info.width, info.height, fps=info.fps, cfg=c) as writer:
        for frame_num, frame in enumerate(read_frames(src, cfg=c)):
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(hsv, lower, upper)
            if invert:
                mask = 255 - mask
            if blur > 0:
                k = blur if blur % 2 == 1 else blur + 1
                mask = cv2.GaussianBlur(mask, (k, k), 0)
            mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            writer.write(mask_rgb)

            now = time.monotonic()
            if now - last_log >= 5.0:
                elapsed = now - t0
                pct = (frame_num / total_frames * 100) if total_frames > 0 else 0
                log.info("chroma-mask: %.0f%% frame=%d/%d fps=%.1f elapsed=%.1fs",
                         pct, frame_num, total_frames,
                         frame_num / elapsed if elapsed > 0 else 0, elapsed)
                last_log = now

    elapsed = time.monotonic() - t0
    log.info("chroma-mask: done in %.1fs", elapsed)
    return dst


@task(name="gradient-mask")
def gradient_mask(dst: Path, width: int, height: int,
                  duration: float, fps: float = 30.0,
                  direction: str = "horizontal",
                  cfg: Optional[Config] = None) -> Path:
    """
    Generate a static gradient mask (useful for wipes / blends).

    direction: "horizontal", "vertical", "radial"
    """
    c = cfg or Config()
    total_frames = int(duration * fps)

    if direction == "horizontal":
        grad = np.tile(
            np.linspace(0, 255, width, dtype=np.uint8),
            (height, 1)
        )
    elif direction == "vertical":
        grad = np.tile(
            np.linspace(0, 255, height, dtype=np.uint8).reshape(-1, 1),
            (1, width)
        )
    elif direction == "radial":
        y, x = np.mgrid[:height, :width]
        cx, cy = width / 2, height / 2
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        max_dist = np.sqrt(cx ** 2 + cy ** 2)
        grad = (dist / max_dist * 255).clip(0, 255).astype(np.uint8)
    else:
        raise ValueError(f"Unknown direction: {direction}")

    mask_rgb = cv2.cvtColor(grad, cv2.COLOR_GRAY2RGB)

    with FrameWriter(dst, width, height, fps=fps, cfg=c) as writer:
        for _ in range(total_frames):
            writer.write(mask_rgb)

    return dst
