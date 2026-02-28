"""
Prefect tasks for compositing — blending, layering, masked merges.

Uses ffmpeg filter graphs for blend and mask operations (fast, stays
in ffmpeg's C pipeline). Falls back to frame-by-frame Python only
for chromakey which needs OpenCV's HSV logic.
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
from ..ffmpeg import probe, read_frames, FrameWriter, run_ffmpeg_logged


# ── ffmpeg blend mode name mapping ──────────────────────────────────────────

FFMPEG_BLEND_MODES = {
    "normal": "normal",
    "add": "addition",
    "multiply": "multiply",
    "screen": "screen",
    "overlay": "overlay",
    "difference": "difference",
    "softlight": "softlight",
}


# ── Tasks ────────────────────────────────────────────────────────────────────

@task(name="blend-layers", cache_policy=FILE_VALIDATED_INPUTS, persist_result=True)
def blend_layers(base: Path, overlay: Path, dst: Path,
                 mode: str = "normal",
                 opacity: float = 0.5,
                 cfg: Optional[Config] = None) -> Path:
    """
    Blend two videos frame-by-frame using ffmpeg's blend filter.

    mode:    one of normal, add, multiply, screen, overlay, difference, softlight
    opacity: 0-1, strength of the overlay layer.

    If videos differ in length, stops at the shorter one.
    """
    c = cfg or Config()
    log = get_run_logger()
    ff_mode = FFMPEG_BLEND_MODES.get(mode)
    if ff_mode is None:
        raise ValueError(f"Unknown blend mode '{mode}'. "
                         f"Available: {list(FFMPEG_BLEND_MODES.keys())}")

    info = probe(base, c)
    log.info("blend-layers: %s + %s → %s  mode=%s opacity=%.2f  "
             "(%dx%d, %.1fs)",
             base.name, overlay.name, dst.name, mode, opacity,
             info.width, info.height, info.duration)

    run_ffmpeg_logged([
        c.ffmpeg_bin, "-y", "-loglevel", c.ffmpeg_loglevel,
        "-i", str(base), "-i", str(overlay),
        "-filter_complex",
        f"[0:v][1:v]blend=all_mode={ff_mode}:all_opacity={opacity}:shortest=1",
        "-an",
        *c.encode_args(),
        str(dst),
    ], duration=info.duration, logger=log, label="blend-layers")

    return dst


@task(name="masked-composite", cache_policy=FILE_VALIDATED_INPUTS, persist_result=True)
def masked_composite(base: Path, overlay: Path,
                     mask: Path, dst: Path,
                     cfg: Optional[Config] = None) -> Path:
    """
    Composite overlay onto base using a grayscale mask (ffmpeg maskedmerge).

    Where mask is white → overlay visible.
    Where mask is black → base visible.
    Grayscale values give partial transparency.

    All three inputs must have the same resolution.
    Stops at the shortest clip.
    """
    c = cfg or Config()
    log = get_run_logger()
    info = probe(base, c)
    log.info("masked-composite: %s + %s (mask=%s) → %s  (%dx%d, %.1fs)",
             base.name, overlay.name, mask.name, dst.name,
             info.width, info.height, info.duration)

    # maskedmerge requires all inputs in the same pixel format;
    # it has no "shortest" option — use -shortest output flag instead
    filter_graph = (
        "[0:v]format=yuv420p[base];"
        "[1:v]format=yuv420p[over];"
        "[2:v]format=yuv420p[msk];"
        "[base][over][msk]maskedmerge"
    )

    run_ffmpeg_logged([
        c.ffmpeg_bin, "-y", "-loglevel", c.ffmpeg_loglevel,
        "-i", str(base), "-i", str(overlay), "-i", str(mask),
        "-filter_complex", filter_graph,
        "-an", "-shortest",
        *c.encode_args(),
        str(dst),
    ], duration=info.duration, logger=log, label="masked-composite")

    return dst


@task(name="multi-layer-composite", cache_policy=FILE_VALIDATED_INPUTS, persist_result=True)
def multi_layer_composite(
    layers: list[tuple[Path, float, str]],
    dst: Path,
    cfg: Optional[Config] = None,
) -> Path:
    """
    Composite multiple layers bottom-to-top via chained ffmpeg blend filters.

    layers: list of (video_path, opacity, blend_mode) tuples.
            First entry is the base layer.
    """
    c = cfg or Config()
    log = get_run_logger()
    if len(layers) < 2:
        raise ValueError("Need at least 2 layers")

    info = probe(layers[0][0], c)
    layer_desc = ", ".join(f"{p.name}({m}@{o:.2f})" for p, o, m in layers)
    log.info("multi-layer-composite: %d layers → %s  (%dx%d, %.1fs)  [%s]",
             len(layers), dst.name, info.width, info.height, info.duration,
             layer_desc)

    # Build ffmpeg inputs
    inputs = []
    for path, _, _ in layers:
        inputs += ["-i", str(path)]

    # Build filter graph: chain blend filters bottom-to-top
    # [0:v][1:v]blend=...[t0]; [t0][2:v]blend=...[t1]; ...
    filter_parts = []
    for i in range(1, len(layers)):
        _, opacity, mode = layers[i]
        ff_mode = FFMPEG_BLEND_MODES.get(mode, "normal")

        if i == 1:
            src = "[0:v]"
        else:
            src = f"[t{i - 2}]"

        overlay_in = f"[{i}:v]"

        if i == len(layers) - 1:
            # Last blend: no output label needed
            filter_parts.append(
                f"{src}{overlay_in}blend=all_mode={ff_mode}"
                f":all_opacity={opacity}:shortest=1"
            )
        else:
            filter_parts.append(
                f"{src}{overlay_in}blend=all_mode={ff_mode}"
                f":all_opacity={opacity}:shortest=1[t{i - 1}]"
            )

    filter_graph = ";".join(filter_parts)

    run_ffmpeg_logged([
        c.ffmpeg_bin, "-y", "-loglevel", c.ffmpeg_loglevel,
        *inputs,
        "-filter_complex", filter_graph,
        "-an",
        *c.encode_args(),
        str(dst),
    ], duration=info.duration, logger=log, label="multi-layer-composite")

    return dst


@task(name="picture-in-picture", cache_policy=FILE_VALIDATED_INPUTS, persist_result=True)
def picture_in_picture(base: Path, overlay: Path, dst: Path,
                       x: int = 0, y: int = 0,
                       scale: float = 0.25,
                       cfg: Optional[Config] = None) -> Path:
    """
    Scale and position overlay on top of base (picture-in-picture).

    x, y:    pixel position of the overlay's top-left corner on the base.
    scale:   overlay width as a fraction of base width (aspect preserved).

    Stops at the shorter clip.
    """
    c = cfg or Config()
    log = get_run_logger()
    info = probe(base, c)
    ow = max(1, int(info.width * scale))
    # force even dimensions for yuv420p
    ow = ow if ow % 2 == 0 else ow + 1

    log.info("picture-in-picture: %s + %s → %s  pos=(%d,%d) scale=%.2f  "
             "(%dx%d, %.1fs)",
             base.name, overlay.name, dst.name, x, y, scale,
             info.width, info.height, info.duration)

    filter_graph = (
        f"[1:v]scale={ow}:-2[pip];"
        f"[0:v][pip]overlay={x}:{y}:shortest=1"
    )

    run_ffmpeg_logged([
        c.ffmpeg_bin, "-y", "-loglevel", c.ffmpeg_loglevel,
        "-i", str(base), "-i", str(overlay),
        "-filter_complex", filter_graph,
        "-an",
        *c.encode_args(),
        str(dst),
    ], duration=info.duration, logger=log, label="picture-in-picture")

    return dst


@task(name="chromakey-composite", cache_policy=FILE_VALIDATED_INPUTS, persist_result=True)
def chromakey_composite(base: Path, overlay: Path, dst: Path,
                        hue_center: int = 60,
                        hue_range: int = 20,
                        sat_min: int = 50,
                        blur: int = 3,
                        cfg: Optional[Config] = None) -> Path:
    """
    Chroma-key composite: remove a colour from the overlay and
    place it over the base.

    Default targets green (hue_center=60 in OpenCV's 0-179 scale).
    Uses frame-by-frame Python for HSV-based keying.
    """
    c = cfg or Config()
    log = get_run_logger()
    info = probe(base, c)
    log.info("chromakey-composite: %s + %s → %s  hue=%d±%d  (%dx%d, %.1fs)",
             base.name, overlay.name, dst.name, hue_center, hue_range,
             info.width, info.height, info.duration)

    t0 = time.monotonic()
    last_log = t0
    lower = np.array([max(0, hue_center - hue_range), sat_min, 50])
    upper = np.array([min(179, hue_center + hue_range), 255, 255])
    total_frames = int(info.duration * info.fps) if info.duration > 0 else 0

    with FrameWriter(dst, info.width, info.height, fps=info.fps, cfg=c) as writer:
        for frame_num, (frame_a, frame_b) in enumerate(
            zip(read_frames(base, cfg=c), read_frames(overlay, cfg=c))
        ):
            hsv = cv2.cvtColor(frame_b, cv2.COLOR_RGB2HSV)
            key_mask = cv2.inRange(hsv, lower, upper)
            if blur > 0:
                k = blur if blur % 2 == 1 else blur + 1
                key_mask = cv2.GaussianBlur(key_mask, (k, k), 0)
            # Where key colour is detected → show base, else show overlay
            alpha = key_mask.astype(np.float32)[:, :, np.newaxis] / 255.0
            result = frame_a.astype(np.float32) * alpha + \
                     frame_b.astype(np.float32) * (1.0 - alpha)
            writer.write(result.clip(0, 255).astype(np.uint8))

            now = time.monotonic()
            if now - last_log >= 5.0:
                elapsed = now - t0
                cur_time = frame_num / info.fps if info.fps > 0 else 0
                pct = (frame_num / total_frames * 100) if total_frames > 0 else 0
                fps_actual = frame_num / elapsed if elapsed > 0 else 0
                log.info(
                    "chromakey-composite: %.0f%% (%.1f/%.1fs) "
                    "frame=%d fps=%.1f elapsed=%.1fs",
                    pct, cur_time, info.duration,
                    frame_num, fps_actual, elapsed,
                )
                last_log = now

    elapsed = time.monotonic() - t0
    log.info("chromakey-composite: done in %.1fs (%.1fx realtime)",
             elapsed, info.duration / elapsed if elapsed > 0 else 0)

    return dst
