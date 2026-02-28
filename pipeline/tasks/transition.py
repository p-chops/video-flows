"""
Prefect tasks for transitions between video clips.

Transitions operate on TWO clips, producing output shorter than the sum
of inputs (clips overlap during the transition). The overlap region from
clip A's tail and clip B's head are blended according to the transition type.

Output length = duration_a + duration_b - transition_duration.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from prefect import task
from prefect.logging import get_run_logger

from ..cache import FILE_VALIDATED_INPUTS
from ..config import Config
from ..ffmpeg import VideoInfo, probe, read_frames, FrameWriter, run_ffmpeg_logged


# ── Helpers ──────────────────────────────────────────────────────────────────

def _validate_pair(
    clip_a: Path, clip_b: Path, duration: float, cfg: Config,
) -> tuple[VideoInfo, VideoInfo, int]:
    """
    Probe both clips, validate resolution/fps match, compute overlap frames.

    Returns (info_a, info_b, overlap_frames).
    Raises ValueError on mismatch or if duration exceeds either clip.
    """
    info_a = probe(clip_a, cfg)
    info_b = probe(clip_b, cfg)
    if (info_a.width, info_a.height) != (info_b.width, info_b.height):
        raise ValueError(
            f"Resolution mismatch: {clip_a.name} is {info_a.width}x{info_a.height}, "
            f"{clip_b.name} is {info_b.width}x{info_b.height}"
        )
    if abs(info_a.fps - info_b.fps) > 0.1:
        raise ValueError(
            f"FPS mismatch: {clip_a.name} is {info_a.fps:.2f}fps, "
            f"{clip_b.name} is {info_b.fps:.2f}fps"
        )
    overlap_frames = int(round(duration * info_a.fps))
    total_a = int(round(info_a.duration * info_a.fps))
    total_b = int(round(info_b.duration * info_b.fps))
    if overlap_frames > total_a or overlap_frames > total_b:
        raise ValueError(
            f"Transition duration {duration:.2f}s ({overlap_frames} frames) "
            f"exceeds clip length: A={info_a.duration:.2f}s, B={info_b.duration:.2f}s"
        )
    return info_a, info_b, overlap_frames


WIPE_PATTERNS = [
    "horizontal", "vertical", "radial", "diagonal",
    "directional", "noise", "star",
]


def _generate_wipe_mask(
    h: int, w: int, pattern: str, progress: float, softness: float,
    noise_field: Optional[np.ndarray] = None,
    angle: float = 0.0,
) -> np.ndarray:
    """
    Generate a single-frame wipe mask (H, W) float32 in [0, 1].

    progress: 0.0 = fully clip A, 1.0 = fully clip B.
    softness: edge gradient width, 0.0 = hard, 1.0 = very soft.
    Returns mask where 1.0 = show clip B, 0.0 = show clip A.
    """
    if pattern == "horizontal":
        field = np.tile(np.linspace(0, 1, w, dtype=np.float32), (h, 1))
    elif pattern == "vertical":
        field = np.tile(
            np.linspace(0, 1, h, dtype=np.float32).reshape(-1, 1), (1, w),
        )
    elif pattern == "radial":
        y, x = np.mgrid[:h, :w]
        cx, cy = w / 2.0, h / 2.0
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.float32)
        field = dist / dist.max()
    elif pattern == "diagonal":
        y, x = np.mgrid[:h, :w]
        field = ((x / max(w, 1) + y / max(h, 1)) / 2.0).astype(np.float32)
    elif pattern == "directional":
        # Arbitrary-angle linear wipe. angle in degrees: 0=L→R, 90=T→B
        angle_rad = np.deg2rad(angle)
        y_arr, x_arr = np.mgrid[:h, :w]
        nx = x_arr.astype(np.float32) / max(w - 1, 1)
        ny = y_arr.astype(np.float32) / max(h - 1, 1)
        field = nx * np.cos(angle_rad) + ny * np.sin(angle_rad)
        field = (field - field.min()) / max(field.max() - field.min(), 1e-10)
    elif pattern == "star":
        # Five-pointed star with straight edges via ray-line intersection
        n_points = 5
        inner_ratio = 0.38
        n_verts = 2 * n_points  # 10 vertices alternating outer/inner

        # Vertex angles (top-pointing star)
        vert_angles = np.array(
            [-np.pi / 2 + i * np.pi / n_points for i in range(n_verts)]
        )
        vert_radii = np.array(
            [1.0 if i % 2 == 0 else inner_ratio for i in range(n_verts)]
        )
        vx = vert_radii * np.cos(vert_angles)
        vy = vert_radii * np.sin(vert_angles)

        y_arr, x_arr = np.mgrid[:h, :w]
        cx, cy = w / 2.0, h / 2.0
        max_dist = np.sqrt(cx ** 2 + cy ** 2)
        dx = (x_arr - cx).astype(np.float64)
        dy = (y_arr - cy).astype(np.float64)
        theta = np.arctan2(dy, dx)
        norm_dist = np.sqrt(dx ** 2 + dy ** 2) / max_dist

        # Determine which sector each pixel falls in
        phi = np.mod(theta + np.pi / 2, 2 * np.pi)
        sector = np.clip(
            np.floor(phi / (np.pi / n_points)).astype(int), 0, n_verts - 1
        )
        idx_b = (sector + 1) % n_verts

        # Bounding vertex positions for each pixel's sector
        ax_v, ay_v = vx[sector], vy[sector]
        bx_v, by_v = vx[idx_b], vy[idx_b]

        # Ray-line intersection: distance from center to star edge
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        numer = ay_v * bx_v - ax_v * by_v
        denom = sin_t * (bx_v - ax_v) - cos_t * (by_v - ay_v)
        safe_denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
        star_r = np.abs(numer / safe_denom)

        field = np.clip(
            norm_dist / np.maximum(star_r, 1e-10), 0, 1
        ).astype(np.float32)
    elif pattern == "noise":
        if noise_field is not None:
            field = noise_field
        else:
            field = np.random.rand(h, w).astype(np.float32)
    else:
        raise ValueError(
            f"Unknown wipe pattern: {pattern}. "
            f"Available: {WIPE_PATTERNS}"
        )

    edge_width = max(softness * 0.5, 0.001)
    mask = np.clip((progress - field + edge_width) / (2 * edge_width), 0.0, 1.0)
    return mask


# ── Tasks ────────────────────────────────────────────────────────────────────

@task(name="crossfade", cache_policy=FILE_VALIDATED_INPUTS, persist_result=True)
def crossfade(
    clip_a: Path, clip_b: Path, dst: Path,
    duration: float = 1.0,
    cfg: Optional[Config] = None,
) -> Path:
    """
    Crossfade between two clips using ffmpeg's xfade filter.

    duration: transition length in seconds (default 1.0).
    """
    c = cfg or Config()
    log = get_run_logger()
    info_a, info_b, _ = _validate_pair(clip_a, clip_b, duration, c)

    offset = info_a.duration - duration
    total_dur = info_a.duration + info_b.duration - duration

    log.info("crossfade: %s + %s → %s  dur=%.2fs offset=%.2fs  (%dx%d)",
             clip_a.name, clip_b.name, dst.name, duration, offset,
             info_a.width, info_a.height)

    run_ffmpeg_logged([
        c.ffmpeg_bin, "-y", "-loglevel", c.ffmpeg_loglevel,
        "-i", str(clip_a), "-i", str(clip_b),
        "-filter_complex",
        f"[0:v][1:v]xfade=transition=fade:duration={duration}:offset={offset}",
        "-an",
        "-c:v", c.default_codec, "-crf", str(c.default_crf),
        "-pix_fmt", c.default_pix_fmt,
        str(dst),
    ], duration=total_dur, logger=log, label="crossfade")

    return dst


@task(name="luma-wipe", cache_policy=FILE_VALIDATED_INPUTS, persist_result=True)
def luma_wipe(
    clip_a: Path, clip_b: Path, dst: Path,
    duration: float = 1.0,
    pattern: str = "horizontal",
    softness: float = 0.1,
    angle: float = 0.0,
    seed: Optional[int] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """
    Luma wipe transition: a grayscale pattern sweeps across the frame,
    progressively revealing clip B over clip A.

    duration:  transition length in seconds (default 1.0).
    pattern:   "horizontal", "vertical", "radial", "diagonal", "directional",
               "noise", "star"
    softness:  edge gradient width 0.0–1.0 (default 0.1). 0 = hard, 1 = soft.
    angle:     wipe direction in degrees for "directional" pattern.
               0 = left→right, 90 = top→bottom.
    seed:      random seed for noise pattern.
    """
    c = cfg or Config()
    log = get_run_logger()
    info_a, info_b, overlap_frames = _validate_pair(clip_a, clip_b, duration, c)

    h, w = info_a.height, info_a.width

    # Pre-generate noise field if needed
    noise_field = None
    if pattern == "noise":
        rng = np.random.default_rng(seed)
        noise_field = rng.random((h, w), dtype=np.float32)

    log.info("luma-wipe: %s + %s → %s  dur=%.2fs pattern=%s softness=%.2f  "
             "(%dx%d)", clip_a.name, clip_b.name, dst.name, duration,
             pattern, softness, w, h)

    t0 = time.monotonic()
    last_log = t0

    frames_a = list(read_frames(clip_a, cfg=c))
    frames_b = list(read_frames(clip_b, cfg=c))

    non_overlap_a = len(frames_a) - overlap_frames

    with FrameWriter(dst, info_a, cfg=c) as writer:
        frame_num = 0

        # Phase 1: non-overlapping clip A frames
        for i in range(non_overlap_a):
            writer.write(frames_a[i])
            frame_num += 1

        # Phase 2: transition frames (overlap region)
        for i in range(overlap_frames):
            t = (i + 1) / (overlap_frames + 1)
            fa = frames_a[non_overlap_a + i].astype(np.float32)
            fb = frames_b[i].astype(np.float32)
            mask = _generate_wipe_mask(h, w, pattern, t, softness, noise_field,
                                         angle=angle)
            mask_3ch = mask[:, :, np.newaxis]
            blended = fa * (1.0 - mask_3ch) + fb * mask_3ch
            writer.write(np.clip(blended, 0, 255).astype(np.uint8))
            frame_num += 1

            now = time.monotonic()
            if now - last_log >= 5.0:
                elapsed = now - t0
                log.info("luma-wipe: transition frame %d/%d  elapsed=%.1fs",
                         i + 1, overlap_frames, elapsed)
                last_log = now

        # Phase 3: non-overlapping clip B frames
        for i in range(overlap_frames, len(frames_b)):
            writer.write(frames_b[i])
            frame_num += 1

    elapsed = time.monotonic() - t0
    log.info("luma-wipe: done in %.1fs (%d frames)", elapsed, frame_num)
    return dst


@task(name="whip-pan", cache_policy=FILE_VALIDATED_INPUTS, persist_result=True)
def whip_pan(
    clip_a: Path, clip_b: Path, dst: Path,
    duration: float = 0.5,
    direction: str = "left",
    blur_strength: float = 0.5,
    cfg: Optional[Config] = None,
) -> Path:
    """
    Whip pan transition: simulated fast camera motion connecting two shots.

    Clip A slides out with motion blur, clip B slides in from opposite side.

    duration:       transition length in seconds (default 0.5).
    direction:      "left", "right", "up", "down" (default "left").
    blur_strength:  motion blur intensity 0.0–1.0 (default 0.5).
    """
    c = cfg or Config()
    log = get_run_logger()
    info_a, info_b, overlap_frames = _validate_pair(clip_a, clip_b, duration, c)

    h, w = info_a.height, info_a.width
    horizontal = direction in ("left", "right")

    # Max blur kernel size: fraction of frame dimension
    dim = w if horizontal else h
    max_blur = int(dim * blur_strength * 0.3)
    max_blur = max(max_blur, 3) | 1  # ensure odd, minimum 3

    log.info("whip-pan: %s + %s → %s  dur=%.2fs dir=%s blur=%.2f  (%dx%d)",
             clip_a.name, clip_b.name, dst.name, duration,
             direction, blur_strength, w, h)

    t0 = time.monotonic()
    frames_a = list(read_frames(clip_a, cfg=c))
    frames_b = list(read_frames(clip_b, cfg=c))

    non_overlap_a = len(frames_a) - overlap_frames
    half = overlap_frames // 2

    with FrameWriter(dst, info_a, cfg=c) as writer:
        # Phase 1: non-overlapping A frames
        for i in range(non_overlap_a):
            writer.write(frames_a[i])

        # Phase 2: transition
        for i in range(overlap_frames):
            if i < half:
                # Out-slide: clip A accelerating off-screen
                slide_t = i / max(half - 1, 1)
                frame = frames_a[non_overlap_a + i]
            else:
                # In-slide: clip B decelerating into frame
                slide_t = 1.0 - ((i - half) / max(overlap_frames - half - 1, 1))
                frame = frames_b[i]

            # Quadratic ease for slide amount
            ease = slide_t ** 2
            max_offset = w if horizontal else h
            offset = int(ease * max_offset)

            # Directional shift via numpy roll + zero-fill
            if horizontal:
                shift = -offset if direction == "left" else offset
                shifted = np.roll(frame, shift, axis=1)
                if shift < 0:
                    shifted[:, shift:] = 0
                elif shift > 0:
                    shifted[:, :shift] = 0
            else:
                shift = -offset if direction == "up" else offset
                shifted = np.roll(frame, shift, axis=0)
                if shift < 0:
                    shifted[shift:, :] = 0
                elif shift > 0:
                    shifted[:shift, :] = 0

            # Directional motion blur
            blur_k = int(ease * max_blur)
            blur_k = max(blur_k, 1) | 1  # odd, >= 1
            if blur_k >= 3:
                if horizontal:
                    kernel = np.zeros((1, blur_k), dtype=np.float32)
                    kernel[0, :] = 1.0 / blur_k
                else:
                    kernel = np.zeros((blur_k, 1), dtype=np.float32)
                    kernel[:, 0] = 1.0 / blur_k
                shifted = cv2.filter2D(shifted, -1, kernel)

            writer.write(shifted)

        # Phase 3: non-overlapping B frames
        for i in range(overlap_frames, len(frames_b)):
            writer.write(frames_b[i])

    elapsed = time.monotonic() - t0
    log.info("whip-pan: done in %.1fs", elapsed)
    return dst


@task(name="static-burst", cache_policy=FILE_VALIDATED_INPUTS, persist_result=True)
def static_burst(
    clip_a: Path, clip_b: Path, dst: Path,
    duration: float = 0.3,
    seed: Optional[int] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """
    Static burst transition: a short burst of TV noise between two clips.

    First quarter fades clip A into static, middle half is pure static,
    last quarter fades static into clip B.

    duration: transition length in seconds (default 0.3).
    seed:     random seed for static pattern.
    """
    c = cfg or Config()
    log = get_run_logger()
    info_a, info_b, overlap_frames = _validate_pair(clip_a, clip_b, duration, c)

    h, w = info_a.height, info_a.width
    rng = np.random.default_rng(seed)

    log.info("static-burst: %s + %s → %s  dur=%.2fs  (%dx%d)",
             clip_a.name, clip_b.name, dst.name, duration, w, h)

    t0 = time.monotonic()
    frames_a = list(read_frames(clip_a, cfg=c))
    frames_b = list(read_frames(clip_b, cfg=c))

    non_overlap_a = len(frames_a) - overlap_frames
    # Ramp zones: first/last quarter fade in/out, middle is pure static
    ramp_in = overlap_frames // 4
    ramp_out = overlap_frames // 4

    with FrameWriter(dst, info_a, cfg=c) as writer:
        # Phase 1: non-overlapping A frames
        for i in range(non_overlap_a):
            writer.write(frames_a[i])

        # Phase 2: transition
        for i in range(overlap_frames):
            static = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)

            if i < ramp_in:
                # Fade clip A → static
                t = (i + 1) / (ramp_in + 1)
                fa = frames_a[non_overlap_a + i].astype(np.float32)
                blended = fa * (1.0 - t) + static.astype(np.float32) * t
                writer.write(np.clip(blended, 0, 255).astype(np.uint8))
            elif i >= overlap_frames - ramp_out:
                # Fade static → clip B
                t = (i - (overlap_frames - ramp_out) + 1) / (ramp_out + 1)
                fb = frames_b[i].astype(np.float32)
                blended = static.astype(np.float32) * (1.0 - t) + fb * t
                writer.write(np.clip(blended, 0, 255).astype(np.uint8))
            else:
                # Pure static
                writer.write(static)

        # Phase 3: non-overlapping B frames
        for i in range(overlap_frames, len(frames_b)):
            writer.write(frames_b[i])

    elapsed = time.monotonic() - t0
    log.info("static-burst: done in %.1fs", elapsed)
    return dst


@task(name="flash", cache_policy=FILE_VALIDATED_INPUTS, persist_result=True)
def flash(
    clip_a: Path, clip_b: Path, dst: Path,
    duration: float = 0.5,
    decay: float = 3.0,
    cfg: Optional[Config] = None,
) -> Path:
    """
    White flash transition: clip A flashes to white, then rapidly
    fades from white into clip B.

    duration: transition length in seconds (default 0.5).
    decay:    how fast the flash fades (default 3.0). Higher = faster
              fade from white to clip B. 1.0 = linear.
    """
    c = cfg or Config()
    log = get_run_logger()
    info_a, info_b, overlap_frames = _validate_pair(clip_a, clip_b, duration, c)

    h, w = info_a.height, info_a.width

    log.info("flash: %s + %s → %s  dur=%.2fs decay=%.1f  (%dx%d)",
             clip_a.name, clip_b.name, dst.name, duration, decay, w, h)

    t0 = time.monotonic()
    frames_a = list(read_frames(clip_a, cfg=c))
    frames_b = list(read_frames(clip_b, cfg=c))

    non_overlap_a = len(frames_a) - overlap_frames
    white = np.full((h, w, 3), 255, dtype=np.float32)

    # Flash peaks at ~25% through the transition
    peak = 0.25

    with FrameWriter(dst, info_a, cfg=c) as writer:
        # Phase 1: non-overlapping A frames
        for i in range(non_overlap_a):
            writer.write(frames_a[i])

        # Phase 2: transition
        for i in range(overlap_frames):
            t = (i + 1) / (overlap_frames + 1)  # 0→1 exclusive
            fa = frames_a[non_overlap_a + i].astype(np.float32)
            fb = frames_b[i].astype(np.float32)

            if t <= peak:
                # Ramp up to white: quadratic ease from clip A
                ramp = (t / peak) ** 2
                blended = fa * (1.0 - ramp) + white * ramp
            else:
                # Decay from white to clip B: exponential falloff
                fade_t = (t - peak) / (1.0 - peak)  # 0→1 within decay phase
                white_amount = (1.0 - fade_t) ** decay
                blended = fb * (1.0 - white_amount) + white * white_amount

            writer.write(np.clip(blended, 0, 255).astype(np.uint8))

        # Phase 3: non-overlapping B frames
        for i in range(overlap_frames, len(frames_b)):
            writer.write(frames_b[i])

    elapsed = time.monotonic() - t0
    log.info("flash: done in %.1fs", elapsed)
    return dst


# ── Blender factories ─────────────────────────────────────────────────────────
#
# Each factory captures transition parameters and returns a callable:
#     blend(tail_buf, head_buf, n, h, w) -> generator of uint8 frames
# where tail_buf/head_buf are lists of uint8 frames, n = overlap frame count.


def _make_luma_wipe_blender(
    pattern: str, softness: float, angle: float, seed: Optional[int],
):
    """Factory for luma wipe overlap blending."""
    noise_field = None
    if pattern == "noise" and seed is not None:
        # Deferred: noise_field generated on first call (needs h, w)
        pass

    def blend(tail, head, n, h, w):
        nf = None
        if pattern == "noise":
            rng = np.random.default_rng(seed)
            nf = rng.random((h, w), dtype=np.float32)
        for i in range(n):
            t = (i + 1) / (n + 1)
            fa = tail[i].astype(np.float32)
            fb = head[i].astype(np.float32)
            mask = _generate_wipe_mask(h, w, pattern, t, softness, nf,
                                       angle=angle)
            mask_3ch = mask[:, :, np.newaxis]
            blended = fa * (1.0 - mask_3ch) + fb * mask_3ch
            yield np.clip(blended, 0, 255).astype(np.uint8)

    return blend


def _make_whip_pan_blender(direction: str, blur_strength: float):
    """Factory for whip pan overlap blending."""
    horizontal = direction in ("left", "right")

    def blend(tail, head, n, h, w):
        dim = w if horizontal else h
        max_blur = int(dim * blur_strength * 0.3)
        max_blur = max(max_blur, 3) | 1
        half = n // 2

        for i in range(n):
            if i < half:
                slide_t = i / max(half - 1, 1)
                frame = tail[i]
            else:
                slide_t = 1.0 - ((i - half) / max(n - half - 1, 1))
                frame = head[i]

            ease = slide_t ** 2
            max_offset = w if horizontal else h
            offset = int(ease * max_offset)

            if horizontal:
                shift = -offset if direction == "left" else offset
                shifted = np.roll(frame, shift, axis=1)
                if shift < 0:
                    shifted[:, shift:] = 0
                elif shift > 0:
                    shifted[:, :shift] = 0
            else:
                shift = -offset if direction == "up" else offset
                shifted = np.roll(frame, shift, axis=0)
                if shift < 0:
                    shifted[shift:, :] = 0
                elif shift > 0:
                    shifted[:shift, :] = 0

            blur_k = int(ease * max_blur)
            blur_k = max(blur_k, 1) | 1
            if blur_k >= 3:
                if horizontal:
                    kernel = np.zeros((1, blur_k), dtype=np.float32)
                    kernel[0, :] = 1.0 / blur_k
                else:
                    kernel = np.zeros((blur_k, 1), dtype=np.float32)
                    kernel[:, 0] = 1.0 / blur_k
                shifted = cv2.filter2D(shifted, -1, kernel)

            yield shifted

    return blend


def _make_static_burst_blender(seed: Optional[int]):
    """Factory for static burst overlap blending."""
    def blend(tail, head, n, h, w):
        rng = np.random.default_rng(seed)
        ramp_in = n // 4
        ramp_out = n // 4

        for i in range(n):
            static = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)

            if i < ramp_in:
                t = (i + 1) / (ramp_in + 1)
                fa = tail[i].astype(np.float32)
                blended = fa * (1.0 - t) + static.astype(np.float32) * t
                yield np.clip(blended, 0, 255).astype(np.uint8)
            elif i >= n - ramp_out:
                t = (i - (n - ramp_out) + 1) / (ramp_out + 1)
                fb = head[i].astype(np.float32)
                blended = static.astype(np.float32) * (1.0 - t) + fb * t
                yield np.clip(blended, 0, 255).astype(np.uint8)
            else:
                yield static

    return blend


def _make_flash_blender(decay: float):
    """Factory for white flash overlap blending."""
    def blend(tail, head, n, h, w):
        white = np.full((h, w, 3), 255, dtype=np.float32)
        peak = 0.25

        for i in range(n):
            t = (i + 1) / (n + 1)
            fa = tail[i].astype(np.float32)
            fb = head[i].astype(np.float32)

            if t <= peak:
                ramp = (t / peak) ** 2
                blended = fa * (1.0 - ramp) + white * ramp
            else:
                fade_t = (t - peak) / (1.0 - peak)
                white_amount = (1.0 - fade_t) ** decay
                blended = fb * (1.0 - white_amount) + white * white_amount

            yield np.clip(blended, 0, 255).astype(np.uint8)

    return blend


def _make_blender(transition_type: str, seed: Optional[int], **kwargs):
    """Dispatch to the appropriate blender factory."""
    if transition_type == "luma_wipe":
        return _make_luma_wipe_blender(
            pattern=kwargs.get("pattern", "horizontal"),
            softness=kwargs.get("softness", 0.1),
            angle=kwargs.get("angle", 0.0),
            seed=seed,
        )
    elif transition_type == "whip_pan":
        return _make_whip_pan_blender(
            direction=kwargs.get("direction", "left"),
            blur_strength=kwargs.get("blur_strength", 0.5),
        )
    elif transition_type == "static_burst":
        return _make_static_burst_blender(seed=seed)
    elif transition_type == "flash":
        return _make_flash_blender(decay=kwargs.get("decay", 3.0))
    else:
        raise ValueError(f"Unknown transition type for streaming: {transition_type}")


_RANDOM_TRANSITION_TYPES = ["crossfade", "luma_wipe", "whip_pan", "static_burst", "flash"]
_WIPE_PATTERNS = [
    "horizontal", "vertical", "radial", "diagonal",
    "directional", "noise", "star",
]
_WHIP_DIRS = ["left", "right", "up", "down"]


def _make_random_blender(seed: int):
    """Pick a random transition type and params, return a blender."""
    rng = np.random.default_rng(seed)
    t_type = rng.choice(_RANDOM_TRANSITION_TYPES)

    if t_type == "crossfade":
        # Crossfade as a blender — simple linear interpolation
        def blend(tail, head, n, h, w):
            for i in range(n):
                t = (i + 1) / (n + 1)
                blended = np.clip(
                    (1.0 - t) * tail[i].astype(np.float32)
                    + t * head[i].astype(np.float32),
                    0, 255,
                ).astype(np.uint8)
                yield blended
        return blend
    elif t_type == "luma_wipe":
        pattern = str(rng.choice(_WIPE_PATTERNS))
        softness = float(rng.uniform(0.05, 0.4))
        angle = float(rng.uniform(0, 360)) if rng.random() < 0.3 else 0.0
        return _make_luma_wipe_blender(pattern, softness, angle, seed)
    elif t_type == "whip_pan":
        direction = str(rng.choice(_WHIP_DIRS))
        blur_strength = float(rng.uniform(0.3, 0.8))
        return _make_whip_pan_blender(direction, blur_strength)
    elif t_type == "static_burst":
        return _make_static_burst_blender(seed)
    else:  # flash
        decay = float(rng.uniform(2.0, 5.0))
        return _make_flash_blender(decay)


# ── Multi-clip chaining ──────────────────────────────────────────────────────

def _xfade_chain(
    clips: list[Path], dst: Path, duration: float, cfg: Config, log,
) -> Path:
    """Chain N clips with crossfade transitions in one ffmpeg call."""
    inputs: list[str] = []
    for clip in clips:
        inputs += ["-i", str(clip)]

    durations = [probe(clip, cfg).duration for clip in clips]

    filter_parts = []
    cumulative_offset = durations[0] - duration
    for i in range(1, len(clips)):
        src = "[0:v]" if i == 1 else f"[v{i - 1}]"
        overlay = f"[{i}:v]"
        out_label = "" if i == len(clips) - 1 else f"[v{i}]"
        filter_parts.append(
            f"{src}{overlay}xfade=transition=fade"
            f":duration={duration}:offset={cumulative_offset:.6f}{out_label}"
        )
        if i < len(clips) - 1:
            cumulative_offset += durations[i] - duration

    filter_graph = ";".join(filter_parts)
    total_dur = sum(durations) - duration * (len(clips) - 1)

    log.info("xfade-chain: %d clips, %.2fs transitions → %s (%.1fs total)",
             len(clips), duration, dst.name, total_dur)

    run_ffmpeg_logged([
        cfg.ffmpeg_bin, "-y", "-loglevel", cfg.ffmpeg_loglevel,
        *inputs,
        "-filter_complex", filter_graph,
        "-an",
        "-c:v", cfg.default_codec, "-crf", str(cfg.default_crf),
        "-pix_fmt", cfg.default_pix_fmt,
        str(dst),
    ], duration=total_dur, logger=log, label="xfade-chain")

    return dst


def _streaming_chain(
    clips: list[Path], dst: Path, transition_type: str,
    duration: float, seed: Optional[int], cfg: Config, log,
    **kwargs,
) -> Path:
    """
    Chain N clips with frame-level transitions in a single streaming pass.

    Instead of pair-by-pair iteration (O(n²) re-encoding), this reads each
    clip exactly once and writes to a single FrameWriter. Only the overlap
    region (~overlap_frames) is buffered at any given time.

    Frame flow:
        Clip 0:  [stream to writer][tail buffer]
                                     ↓ blend ↓
        Clip 1:                     [head buf][stream][tail buffer]
                                                       ↓ blend ↓
        Clip 2:                                       [head buf][stream]
    """
    from collections import deque

    # Probe first clip for output info and compute overlap
    info_0 = probe(clips[0], cfg)
    h, w = info_0.height, info_0.width
    overlap_frames = int(round(duration * info_0.fps))

    # Derive per-pair seeds deterministically from base seed
    rng_seeds = np.random.default_rng(seed)
    pair_seeds = [int(rng_seeds.integers(0, 2**31)) for _ in range(len(clips) - 1)]

    total_written = 0
    t0 = time.monotonic()
    last_log = t0

    log.info("streaming-chain: %d clips, %s transitions (%.2fs each) → %s  (%dx%d)",
             len(clips), transition_type, duration, dst.name, w, h)

    with FrameWriter(dst, info_0, cfg=cfg) as writer:
        tail_buffer: list[np.ndarray] = []

        for clip_idx, clip_path in enumerate(clips):
            gen = read_frames(clip_path, cfg=cfg)
            is_last_clip = (clip_idx == len(clips) - 1)

            if clip_idx == 0:
                # First clip: stream everything except the tail
                window = deque(maxlen=overlap_frames)
                for frame in gen:
                    if len(window) == overlap_frames:
                        writer.write(window[0])
                        total_written += 1
                    window.append(frame)
                tail_buffer = list(window)

            else:
                # Subsequent clips: blend head with previous tail, then stream

                # Collect head frames for blending
                head_buffer: list[np.ndarray] = []
                for frame in gen:
                    if len(head_buffer) < overlap_frames:
                        head_buffer.append(frame)
                    else:
                        break
                # 'frame' is now the first non-overlap frame (or gen is exhausted)

                # Blend overlap region
                pair_seed = pair_seeds[clip_idx - 1]
                if transition_type == "random":
                    blender = _make_random_blender(pair_seed)
                else:
                    blender = _make_blender(transition_type, pair_seed, **kwargs)
                n_blend = min(len(tail_buffer), len(head_buffer))
                for blended_frame in blender(tail_buffer[:n_blend],
                                             head_buffer[:n_blend],
                                             n_blend, h, w):
                    writer.write(blended_frame)
                    total_written += 1

                # Write any leftover head frames beyond the blend region
                for hf in head_buffer[n_blend:]:
                    writer.write(hf)
                    total_written += 1

                if is_last_clip:
                    # Last clip: stream remaining frames directly
                    # (we already consumed one frame breaking out of the
                    #  head-collection loop — write it if gen wasn't exhausted)
                    try:
                        writer.write(frame)
                        total_written += 1
                    except UnboundLocalError:
                        pass  # gen exhausted during head collection
                    for f in gen:
                        writer.write(f)
                        total_written += 1
                else:
                    # Interior clip: stream body, buffer tail for next transition
                    window = deque(maxlen=overlap_frames)
                    # The 'frame' we broke out with goes into the window
                    try:
                        window.append(frame)
                    except UnboundLocalError:
                        pass
                    for f in gen:
                        if len(window) == overlap_frames:
                            writer.write(window[0])
                            total_written += 1
                        window.append(f)
                    tail_buffer = list(window)

            now = time.monotonic()
            if now - last_log >= 5.0:
                elapsed = now - t0
                log.info("streaming-chain: clip %d/%d done, %d frames written  "
                         "elapsed=%.1fs", clip_idx + 1, len(clips),
                         total_written, elapsed)
                last_log = now

    elapsed = time.monotonic() - t0
    log.info("streaming-chain: done in %.1fs (%d frames)", elapsed, total_written)
    return dst


@task(name="transition-sequence", cache_policy=FILE_VALIDATED_INPUTS, persist_result=True)
def transition_sequence(
    clips: list[Path], dst: Path,
    transition_type: str = "crossfade",
    duration: float = 1.0,
    seed: Optional[int] = None,
    cfg: Optional[Config] = None,
    **kwargs,
) -> Path:
    """
    Join multiple clips with transitions between each adjacent pair.

    For crossfade: chained ffmpeg xfade filters (single invocation).
    For luma_wipe / whip_pan / static_burst / flash: streaming chain
    (single FrameWriter, each clip read exactly once).

    transition_type: "crossfade", "luma_wipe", "whip_pan", "static_burst",
                     "flash", "random" (picks a different type per pair)
    duration:        transition length in seconds
    **kwargs:        passed to underlying transition (pattern, softness,
                     direction, blur_strength, decay)
    """
    c = cfg or Config()
    log = get_run_logger()

    if len(clips) < 2:
        if len(clips) == 1:
            shutil.copy2(clips[0], dst)
            return dst
        raise ValueError("Need at least one clip")

    if transition_type == "crossfade":
        return _xfade_chain(clips, dst, duration, c, log)
    else:
        return _streaming_chain(clips, dst, transition_type, duration,
                                seed, c, log, **kwargs)
