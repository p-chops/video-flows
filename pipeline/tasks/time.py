"""
Temporal manipulation tasks — time scrubbing, drift looping, ping-pong,
echo trails, temporal patchwork.
"""

from __future__ import annotations

import logging
import time as _time
from pathlib import Path
from typing import Optional

import numpy as np
from prefect import task

from ..config import Config
from ..ffmpeg import FrameWriter, probe, read_frames


logger = logging.getLogger(__name__)


def _generate_playhead_curve(
    n_frames: int,
    fps: float,
    smoothness: float,
    intensity: float,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a smooth playhead position curve for time scrubbing.

    Returns an array of length n_frames where each value is a fractional
    source frame index in [0, n_frames-1].

    The curve is built by synthesising coloured noise as a velocity signal
    (via spectral shaping), then integrating to get position.

    smoothness: temporal scale of speed changes in seconds.
                Higher = slower, more gradual transitions.
    intensity:  magnitude of speed variation (0–1).
                0 = constant 1x forward playback.
                0.5 = strong scrubbing with frequent reversals.
                1.0 = extreme — playhead barely progresses forward.
    """
    rng = np.random.default_rng(seed)
    n = n_frames

    if n < 2:
        return np.zeros(max(n, 1))

    # Random phase spectrum
    n_bins = n // 2 + 1
    phases = rng.uniform(0, 2 * np.pi, n_bins)

    # Spectral envelope: gentle rolloff at cutoff frequency
    # cutoff = 1 / (smoothness * fps) in normalised frequency
    freqs = np.arange(n_bins, dtype=np.float64) / n
    f_cutoff = 1.0 / max(smoothness * fps, 1.0)
    envelope = 1.0 / (1.0 + (freqs / max(f_cutoff, 1e-9)) ** 2)

    # Synthesise coloured noise
    spectrum = envelope * np.exp(1j * phases)
    spectrum[0] = 0  # zero DC — we add our own mean later
    noise = np.fft.irfft(spectrum, n=n)

    # Normalise to zero mean, unit std
    std = noise.std()
    if std > 1e-8:
        noise = (noise - noise.mean()) / std
    else:
        noise = np.zeros(n)

    # Velocity: base forward speed + scaled variation
    # Scale factor 3x so intensity=0.5 gives strong scrubbing (~30% reverse)
    # and intensity=1.0 is extreme (playhead barely drifts forward)
    velocity = 1.0 + 3.0 * intensity * noise

    # Integrate to get position
    position = np.cumsum(velocity)
    # Shift so position[0] = 0
    position -= position[0]

    # Clamp to valid frame range
    np.clip(position, 0, n_frames - 1, out=position)

    return position


@task(name="time-scrub")
def time_scrub(
    src: Path,
    dst: Path,
    *,
    smoothness: float = 2.0,
    intensity: float = 0.5,
    seed: Optional[int] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """
    Random temporal scrub — remap a video's timeline via a smooth random
    playhead curve.

    Output is the same duration as input. The virtual playhead wanders
    through the source at varying speeds, including reverse, controlled
    by smoothness (how gradual) and intensity (how wild).

    smoothness: seconds — temporal scale of speed changes (default 2.0).
                Low (~0.5) = jittery.  High (~5.0) = glacial.
    intensity:  speed variation magnitude 0–1 (default 0.5).
                0 = normal playback.  0.5 = strong scrub.  1.0 = extreme.
    seed:       random seed for reproducibility.
    """
    c = cfg or Config()
    info = probe(src, c)
    n_frames = int(round(info.duration * info.fps))

    print(f"time_scrub: {n_frames} frames, {info.duration:.1f}s @ {info.fps}fps")
    print(f"  smoothness={smoothness}, intensity={intensity}, seed={seed}")

    # --- Load all source frames into memory ---
    frames: list[np.ndarray] = []
    for frame in read_frames(src, c):
        frames.append(frame)
    n_loaded = len(frames)

    if n_loaded == 0:
        raise ValueError(f"No frames read from {src}")

    # Use actual loaded count (may differ slightly from duration * fps)
    print(f"  loaded {n_loaded} frames ({n_loaded * info.width * info.height * 3 / 1e6:.0f} MB)")

    # --- Generate playhead curve ---
    position = _generate_playhead_curve(
        n_loaded, info.fps, smoothness, intensity, seed,
    )

    # Log playhead stats
    velocity = np.diff(position)
    pct_reverse = (velocity < 0).sum() / max(len(velocity), 1) * 100
    print(f"  playhead: speed range [{velocity.min():.2f}, {velocity.max():.2f}], "
          f"{pct_reverse:.0f}% reverse")

    # --- Write remapped frames ---
    with FrameWriter(dst, info, cfg=c) as writer:
        t0 = _time.monotonic()
        last_log = t0
        for i in range(n_loaded):
            src_idx = int(round(position[i]))
            src_idx = max(0, min(src_idx, n_loaded - 1))
            writer.write(frames[src_idx])

            now = _time.monotonic()
            if now - last_log >= 5.0:
                pct = (i + 1) / n_loaded * 100
                elapsed = now - t0
                logger.info("time_scrub: %.0f%% (%d/%d) elapsed=%.1fs",
                            pct, i + 1, n_loaded, elapsed)
                last_log = now

    elapsed = _time.monotonic() - t0
    print(f"  wrote {n_loaded} frames in {elapsed:.1f}s → {dst.name}")

    return dst


@task(name="drift-loop")
def drift_loop(
    src: Path,
    dst: Path,
    *,
    loop_dur: float = 0.5,
    drift: Optional[float] = None,
    seed: Optional[int] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """
    Drift loop — loop a fixed-length window starting at a random position
    in the source. The window drifts forward or backward each cycle.
    Hard cuts at loop boundaries.

    Output is the same duration as input.

    loop_dur:   seconds per loop cycle (default 0.5).
    drift:      seconds of drift per cycle (default: auto, +-10% of loop
                length, direction chosen randomly). Positive = forward,
                negative = backward. Magnitude controls evolution speed.
    seed:       random seed (controls start position and auto-drift direction).
    """
    c = cfg or Config()
    info = probe(src, c)
    rng = np.random.default_rng(seed)

    # --- Load all source frames into memory ---
    frames: list[np.ndarray] = []
    for frame in read_frames(src, c):
        frames.append(frame)
    n = len(frames)

    if n == 0:
        raise ValueError(f"No frames read from {src}")

    fps = info.fps
    loop_frames = max(1, round(loop_dur * fps))
    loop_frames = min(loop_frames, n)

    n_cycles = n / loop_frames
    # Auto-drift: +-10% of loop length, random direction
    if drift is None:
        mag = max(1, round(0.1 * loop_frames))
        drift_frames = mag if rng.random() > 0.5 else -mag
    else:
        raw = round(drift * fps)
        drift_frames = raw if raw != 0 else (1 if drift >= 0 else -1)

    # Random start position — leave room for the loop window
    # and for drift to run without immediately clamping
    total_drift = abs(drift_frames) * n_cycles
    if drift_frames >= 0:
        # Drifting forward: start early enough that we don't clamp too soon
        max_start = max(0, n - loop_frames - total_drift)
    else:
        # Drifting backward: start late enough to have room behind us
        max_start = n - loop_frames
        min_start = min(n - loop_frames, total_drift)
        max_start = max(max_start, min_start)

    start_frame = rng.integers(0, max(1, int(max_start) + 1))

    direction = "forward" if drift_frames >= 0 else "backward"
    print(f"drift_loop: {n} frames, {info.duration:.1f}s @ {fps}fps")
    print(f"  loop={loop_frames}f ({loop_frames/fps:.2f}s), "
          f"drift={drift_frames:+d}f ({drift_frames/fps:+.2f}s/cycle, {direction})")
    print(f"  start={start_frame}f ({start_frame/fps:.2f}s), "
          f"{n_cycles:.1f} cycles, seed={seed}")

    mem_mb = n * info.width * info.height * 3 / 1e6
    print(f"  loaded {n} frames ({mem_mb:.0f} MB)")

    # --- Write drift-looped frames ---
    with FrameWriter(dst, info, cfg=c) as writer:
        t0 = _time.monotonic()
        last_log = t0
        for out_idx in range(n):
            cycle = out_idx // loop_frames
            pos_in_loop = out_idx % loop_frames

            window_start = start_frame + cycle * drift_frames
            src_idx = max(0, min(window_start + pos_in_loop, n - 1))
            writer.write(frames[src_idx])

            now = _time.monotonic()
            if now - last_log >= 5.0:
                pct = (out_idx + 1) / n * 100
                elapsed = now - t0
                logger.info("drift_loop: %.0f%% (%d/%d) elapsed=%.1fs",
                            pct, out_idx + 1, n, elapsed)
                last_log = now

    elapsed = _time.monotonic() - t0
    print(f"  wrote {n} frames in {elapsed:.1f}s → {dst.name}")

    return dst


@task(name="ping-pong")
def ping_pong(
    src: Path,
    dst: Path,
    *,
    window: float = 0.5,
    seed: Optional[int] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """
    Ping-pong — extract a short subsegment and play it forward-backward
    continuously for the full duration of the input. Breathing, pulsing
    repetition.

    Output is the same duration as input.

    window: seconds of source to use for the ping-pong (default 0.5).
    seed:   random seed (controls which subsegment is chosen).
    """
    c = cfg or Config()
    info = probe(src, c)
    rng = np.random.default_rng(seed)

    # --- Load all source frames into memory ---
    frames: list[np.ndarray] = []
    for frame in read_frames(src, c):
        frames.append(frame)
    n = len(frames)

    if n == 0:
        raise ValueError(f"No frames read from {src}")

    fps = info.fps
    win_frames = max(2, round(window * fps))
    win_frames = min(win_frames, n)

    # Random start position for the window
    max_start = n - win_frames
    start = rng.integers(0, max(1, max_start + 1))

    # Build one ping-pong cycle: forward then backward (excluding endpoints
    # to avoid a double-frame stutter at the turnaround)
    fwd = list(range(start, start + win_frames))
    bwd = list(range(start + win_frames - 2, start, -1))  # skip first & last
    cycle = fwd + bwd
    cycle_len = len(cycle)

    n_cycles = n / cycle_len
    print(f"ping_pong: {n} frames, {info.duration:.1f}s @ {fps}fps")
    print(f"  window={win_frames}f ({win_frames/fps:.2f}s), "
          f"start={start}f ({start/fps:.2f}s)")
    print(f"  cycle={cycle_len}f ({cycle_len/fps:.2f}s), "
          f"{n_cycles:.1f} breaths, seed={seed}")

    mem_mb = n * info.width * info.height * 3 / 1e6
    print(f"  loaded {n} frames ({mem_mb:.0f} MB)")

    # --- Write ping-pong frames ---
    with FrameWriter(dst, info, cfg=c) as writer:
        t0 = _time.monotonic()
        last_log = t0
        for out_idx in range(n):
            src_idx = cycle[out_idx % cycle_len]
            writer.write(frames[src_idx])

            now = _time.monotonic()
            if now - last_log >= 5.0:
                pct = (out_idx + 1) / n * 100
                elapsed = now - t0
                logger.info("ping_pong: %.0f%% (%d/%d) elapsed=%.1fs",
                            pct, out_idx + 1, n, elapsed)
                last_log = now

    elapsed = _time.monotonic() - t0
    print(f"  wrote {n} frames in {elapsed:.1f}s → {dst.name}")

    return dst


@task(name="echo-trail")
def echo_trail(
    src: Path,
    dst: Path,
    *,
    delay: float = 0.0,
    trail: float = 0.8,
    cfg: Optional[Config] = None,
) -> Path:
    """
    Temporal echo / trails — blend each frame with a delayed version of the
    output, producing ghostly motion trails or distinct temporal echoes.

    With feedback: echoes echo themselves, creating multiple decaying ghosts
    at regular intervals (like audio delay with feedback).

    Single-pass streaming operation.

    Output is the same duration as input.

    delay: echo delay in seconds (default 0.0).
           0 = motion blur (blend with previous frame — single accumulator).
           0.5 = ghosts appear 0.5s behind, 1.0s behind (fainter), etc.
           1.0 = widely spaced, distinct apparitions.
    trail: echo strength / feedback, 0–1 (default 0.8).
           0 = no echo (passthrough).
           0.5 = moderate ghosts, visible but transparent.
           0.8 = strong ghosts, persistent.
           0.95 = extreme — ghosts dominate, new frames barely punch through.
    """
    c = cfg or Config()
    info = probe(src, c)
    delay_frames = max(1, round(delay * info.fps)) if delay > 0 else 1

    mem_mb = delay_frames * info.width * info.height * 3 / 1e6
    mode = "motion blur" if delay <= 0 else f"echo delay={delay_frames}f ({delay:.2f}s)"
    print(f"echo_trail: {info.duration:.1f}s @ {info.fps}fps, "
          f"{info.width}x{info.height}")
    print(f"  {mode}, trail={trail}, buffer={delay_frames}f ({mem_mb:.0f} MB)")

    from collections import deque
    ring: deque[np.ndarray] = deque(maxlen=delay_frames)
    blend_new = 1.0 - trail
    frame_count = 0

    with FrameWriter(dst, info, cfg=c) as writer:
        t0 = _time.monotonic()
        last_log = t0
        for frame in read_frames(src, c):
            f = frame.astype(np.float32)

            if len(ring) < delay_frames:
                # Buffer not yet full — output current frame unblended
                out = f
            else:
                # Blend current frame with the delayed output
                echo_src = ring[0]  # oldest in ring = delay_frames ago
                out = blend_new * f + trail * echo_src

            # Push output into ring (feedback — echoes echo themselves)
            ring.append(out.copy())
            writer.write(np.clip(out, 0, 255).astype(np.uint8))
            frame_count += 1

            now = _time.monotonic()
            if now - last_log >= 5.0:
                elapsed = now - t0
                logger.info("echo_trail: %d frames, elapsed=%.1fs",
                            frame_count, elapsed)
                last_log = now

    elapsed = _time.monotonic() - t0
    print(f"  wrote {frame_count} frames in {elapsed:.1f}s → {dst.name}")

    return dst


@task(name="time-patch")
def time_patch(
    src: Path,
    dst: Path,
    *,
    patch_min: float = 0.05,
    patch_max: float = 0.4,
    seed: Optional[int] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """
    Temporal patchwork — for each frame, paste a randomly sized and
    positioned rectangular crop of the current frame onto the output
    canvas. Everything outside the patch is left over from previous
    frames, creating a mosaic of moments from different points in time.

    Patch size varies randomly per frame within [patch_min, patch_max].

    Single-pass streaming. One canvas buffer (size of one frame).

    Output is the same duration as input.

    patch_min: minimum patch size as fraction of frame dims (default 0.05).
    patch_max: maximum patch size as fraction of frame dims (default 0.4).
    seed:      random seed for reproducibility.
    """
    c = cfg or Config()
    info = probe(src, c)
    rng = np.random.default_rng(seed)

    h, w = info.height, info.width

    print(f"time_patch: {info.duration:.1f}s @ {info.fps}fps, {w}x{h}")
    print(f"  patch range={patch_min:.0%}–{patch_max:.0%}, seed={seed}")

    canvas: Optional[np.ndarray] = None
    frame_count = 0

    with FrameWriter(dst, info, cfg=c) as writer:
        t0 = _time.monotonic()
        last_log = t0
        for frame in read_frames(src, c):
            if canvas is None:
                canvas = frame.copy()
            else:
                # Random patch size this frame
                frac = rng.uniform(patch_min, patch_max)
                ph = max(1, round(h * frac))
                pw = max(1, round(w * frac))
                # Random position
                y = rng.integers(0, h - ph + 1)
                x = rng.integers(0, w - pw + 1)
                canvas[y:y + ph, x:x + pw] = frame[y:y + ph, x:x + pw]

            writer.write(canvas)
            frame_count += 1

            now = _time.monotonic()
            if now - last_log >= 5.0:
                elapsed = now - t0
                logger.info("time_patch: %d frames, elapsed=%.1fs",
                            frame_count, elapsed)
                last_log = now

    elapsed = _time.monotonic() - t0
    print(f"  wrote {frame_count} frames in {elapsed:.1f}s → {dst.name}")

    return dst
