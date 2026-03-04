"""
Temporal manipulation tasks — time scrubbing, drift looping, ping-pong,
echo trails, temporal patchwork, slit-scan, temporal tile, smear, bloom,
stack, slip, temporal sort, extrema hold, feedback transform.

Each effect has a _process_* helper (operates on in-memory frame buffers)
and a @task wrapper (handles file I/O). The fused_time_chain task applies
multiple effects in a single decode-encode pass.
"""

from __future__ import annotations

import logging
import time as _time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
from prefect import task

from ..config import Config
from ..ffmpeg import FrameWriter, probe, read_frames


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

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

    # Sharpen zero crossings — the raw noise drifts through zero slowly
    # (lingering at low-speed reverse points). Signed power < 1 pushes
    # values toward ±1, making the playhead snap between forward and
    # reverse instead of decelerating through the turn.
    noise = np.sign(noise) * np.abs(noise) ** 0.5

    # Velocity: base forward speed + scaled variation
    # Scale factor 3x so intensity=0.5 gives strong scrubbing (~30% reverse)
    # and intensity=1.0 is extreme (playhead barely drifts forward)
    velocity = 1.0 + 3.0 * intensity * noise

    # Integrate to get position
    position = np.cumsum(velocity)
    # Shift so position[0] = 0
    position -= position[0]

    # Wrap to valid frame range (mod n) so the playhead loops
    # instead of freezing on the last frame when it runs ahead.
    position = position % n_frames

    return position


class FrameBuffer:
    """List-like frame buffer backed by RAM or a memory-mapped temp file.

    Decision logic (on creation):
      estimated size > cfg.max_ram_mb       → MemoryError (pre-flight gate)
      estimated size > cfg.memmap_threshold_mb → np.memmap (disk-backed)
      otherwise                              → np.ndarray  (pure RAM)

    Supports len(), indexing, iteration, append(), and context-manager
    cleanup — a drop-in replacement for list[np.ndarray] in _process_*
    helpers.
    """

    def __init__(
        self,
        capacity: int,
        height: int,
        width: int,
        *,
        cfg: Optional[Config] = None,
    ):
        c = cfg or Config()
        self._capacity = capacity
        self._count = 0
        self._tmppath: Optional[Path] = None

        size_mb = capacity * height * width * 3 / 1e6
        if size_mb > c.max_ram_mb:
            raise MemoryError(
                f"Frame buffer would use ~{size_mb:.0f} MB "
                f"(limit {c.max_ram_mb} MB) — clip too long for in-memory processing"
            )

        shape = (capacity, height, width, 3)
        if size_mb > c.memmap_threshold_mb:
            import tempfile
            work = c.work_dir
            work.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(suffix=".mmap", dir=str(work))
            import os as _os
            _os.close(fd)
            self._tmppath = Path(tmp)
            self._data = np.memmap(
                str(self._tmppath), dtype=np.uint8, mode="w+", shape=shape,
            )
            logger.info("FrameBuffer: memmap %.0f MB → %s", size_mb, self._tmppath.name)
        else:
            self._data = np.empty(shape, dtype=np.uint8)

    # ── Sequence interface ─────────────────────────────────────────────
    def __len__(self) -> int:
        return self._count

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self._data[i] for i in range(*idx.indices(self._count))]
        if idx < 0:
            idx += self._count
        if not 0 <= idx < self._count:
            raise IndexError(idx)
        return self._data[idx]

    def __setitem__(self, idx: int, value):
        if idx < 0:
            idx += self._count
        self._data[idx] = value

    def __iter__(self):
        for i in range(self._count):
            yield self._data[i]

    def append(self, frame: np.ndarray):
        if self._count >= self._capacity:
            raise RuntimeError(
                f"FrameBuffer full ({self._capacity} frames)"
            )
        self._data[self._count] = frame
        self._count += 1

    # ── Utilities ──────────────────────────────────────────────────────
    @property
    def is_memmap(self) -> bool:
        return self._tmppath is not None

    def cleanup(self):
        """Release memmap and delete temp file (no-op for RAM buffers)."""
        if self._tmppath is not None:
            del self._data
            try:
                self._tmppath.unlink(missing_ok=True)
            except OSError:
                pass
            self._tmppath = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.cleanup()
        return False


def _load_frames(src: Path, cfg: Config) -> FrameBuffer:
    """Load all frames into a FrameBuffer (RAM or memmap depending on size).

    Raises MemoryError before decoding if the estimated buffer exceeds
    cfg.max_ram_mb.
    """
    info = probe(src, cfg)
    n_est = int(info.duration * info.fps) + 1
    buf = FrameBuffer(n_est, info.height, info.width, cfg=cfg)
    for frame in read_frames(src, cfg):
        buf.append(frame)
    if len(buf) == 0:
        buf.cleanup()
        raise ValueError(f"No frames read from {src}")
    return buf


def _write_frames(
    frames: list[np.ndarray],
    dst: Path,
    info,
    cfg: Config,
    label: str,
) -> Path:
    """Write a frame buffer to a video file with progress logging."""
    n = len(frames)
    with FrameWriter(dst, info, cfg=cfg) as writer:
        t0 = _time.monotonic()
        last_log = t0
        for i, frame in enumerate(frames):
            writer.write(frame)
            now = _time.monotonic()
            if now - last_log >= 5.0:
                pct = (i + 1) / n * 100
                logger.info("%s: %.0f%% (%d/%d) elapsed=%.1fs",
                            label, pct, i + 1, n, now - t0)
                last_log = now
    elapsed = _time.monotonic() - t0
    print(f"  wrote {n} frames in {elapsed:.1f}s → {dst.name}")
    return dst


# ---------------------------------------------------------------------------
# 1. Time Scrub
# ---------------------------------------------------------------------------

def _process_scrub(
    frames: list[np.ndarray],
    fps: float,
    *,
    smoothness: float = 2.0,
    intensity: float = 0.5,
    seed: Optional[int] = None,
) -> list[np.ndarray]:
    """Remap timeline via a smooth random playhead curve."""
    n = len(frames)
    position = _generate_playhead_curve(n, fps, smoothness, intensity, seed)
    return [frames[int(round(position[i])) % n] for i in range(n)]


@task(name="time-scrub", tags=["ram-heavy"])
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

    with _load_frames(src, c) as frames:
        n_loaded = len(frames)
        print(f"  loaded {n_loaded} frames ({n_loaded * info.width * info.height * 3 / 1e6:.0f} MB)")

        position = _generate_playhead_curve(n_loaded, info.fps, smoothness, intensity, seed)
        velocity = np.diff(position)
        pct_reverse = (velocity < 0).sum() / max(len(velocity), 1) * 100
        print(f"  playhead: speed range [{velocity.min():.2f}, {velocity.max():.2f}], "
              f"{pct_reverse:.0f}% reverse")

        output = _process_scrub(frames, info.fps, smoothness=smoothness, intensity=intensity, seed=seed)
        return _write_frames(output, dst, info, c, "time_scrub")


# ---------------------------------------------------------------------------
# 2. Drift Loop
# ---------------------------------------------------------------------------

def _process_drift(
    frames: list[np.ndarray],
    fps: float,
    *,
    loop_dur: float = 0.5,
    drift: Optional[float] = None,
    seed: Optional[int] = None,
) -> list[np.ndarray]:
    """Loop a fixed-length window with per-cycle drift."""
    rng = np.random.default_rng(seed)
    n = len(frames)
    loop_frames = max(1, round(loop_dur * fps))
    loop_frames = min(loop_frames, n)

    n_cycles = n / loop_frames
    if drift is None:
        mag = max(1, round(0.1 * loop_frames))
        drift_frames = mag if rng.random() > 0.5 else -mag
    else:
        raw = round(drift * fps)
        drift_frames = raw if raw != 0 else (1 if drift >= 0 else -1)

    total_drift = abs(drift_frames) * n_cycles
    if drift_frames >= 0:
        max_start = max(0, n - loop_frames - total_drift)
    else:
        max_start = n - loop_frames
        min_start = min(n - loop_frames, total_drift)
        max_start = max(max_start, min_start)

    start_frame = rng.integers(0, max(1, int(max_start) + 1))

    output = []
    for out_idx in range(n):
        cycle = out_idx // loop_frames
        pos_in_loop = out_idx % loop_frames
        window_start = start_frame + cycle * drift_frames
        src_idx = (window_start + pos_in_loop) % n
        output.append(frames[src_idx])
    return output


@task(name="drift-loop", tags=["ram-heavy"])
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

    with _load_frames(src, c) as frames:
        n = len(frames)
        fps = info.fps

        loop_frames = max(1, round(loop_dur * fps))
        loop_frames = min(loop_frames, n)
        n_cycles = n / loop_frames

        print(f"drift_loop: {n} frames, {info.duration:.1f}s @ {fps}fps")
        mem_mb = n * info.width * info.height * 3 / 1e6
        print(f"  loaded {n} frames ({mem_mb:.0f} MB)")

        output = _process_drift(frames, fps, loop_dur=loop_dur, drift=drift, seed=seed)
        return _write_frames(output, dst, info, c, "drift_loop")


# ---------------------------------------------------------------------------
# 3. Ping-Pong
# ---------------------------------------------------------------------------

def _process_ping_pong(
    frames: list[np.ndarray],
    fps: float,
    *,
    window: float = 0.5,
    seed: Optional[int] = None,
) -> list[np.ndarray]:
    """Play a short subsegment forward-backward continuously."""
    rng = np.random.default_rng(seed)
    n = len(frames)
    win_frames = max(2, round(window * fps))
    win_frames = min(win_frames, n)

    max_start = n - win_frames
    start = rng.integers(0, max(1, max_start + 1))

    fwd = list(range(start, start + win_frames))
    bwd = list(range(start + win_frames - 2, start, -1))
    cycle = fwd + bwd
    cycle_len = len(cycle)

    return [frames[cycle[out_idx % cycle_len]] for out_idx in range(n)]


@task(name="ping-pong", tags=["ram-heavy"])
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

    with _load_frames(src, c) as frames:
        n = len(frames)
        fps = info.fps
        win_frames = max(2, round(window * fps))
        win_frames = min(win_frames, n)

        print(f"ping_pong: {n} frames, {info.duration:.1f}s @ {fps}fps")
        mem_mb = n * info.width * info.height * 3 / 1e6
        print(f"  loaded {n} frames ({mem_mb:.0f} MB)")

        output = _process_ping_pong(frames, fps, window=window, seed=seed)
        return _write_frames(output, dst, info, c, "ping_pong")


# ---------------------------------------------------------------------------
# 4. Echo Trail
# ---------------------------------------------------------------------------

def _process_echo(
    frames: list[np.ndarray],
    fps: float,
    *,
    delay: float = 0.0,
    trail: float = 0.8,
) -> list[np.ndarray]:
    """Temporal echo with feedback via ring buffer."""
    delay_frames = max(1, round(delay * fps)) if delay > 0 else 1
    ring: deque[np.ndarray] = deque(maxlen=delay_frames)
    blend_new = 1.0 - trail
    output = []
    for frame in frames:
        f = frame.astype(np.float32)
        if len(ring) < delay_frames:
            out = f
        else:
            echo_src = ring[0]
            out = blend_new * f + trail * echo_src
        ring.append(out.copy())
        output.append(np.clip(out, 0, 255).astype(np.uint8))
    return output


@task(name="echo-trail", tags=["ram-heavy"])
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

    with _load_frames(src, c) as frames:
        output = _process_echo(frames, info.fps, delay=delay, trail=trail)
        return _write_frames(output, dst, info, c, "echo_trail")


# ---------------------------------------------------------------------------
# 5. Time Patch
# ---------------------------------------------------------------------------

def _process_patch(
    frames: list[np.ndarray],
    fps: float,
    *,
    patch_min: float = 0.05,
    patch_max: float = 0.4,
    seed: Optional[int] = None,
) -> list[np.ndarray]:
    """Paste random crops of current frame onto an accumulating canvas."""
    rng = np.random.default_rng(seed)
    h, w = frames[0].shape[:2]
    canvas: Optional[np.ndarray] = None
    output = []
    for frame in frames:
        if canvas is None:
            canvas = frame.copy()
        else:
            frac = rng.uniform(patch_min, patch_max)
            ph = max(1, round(h * frac))
            pw = max(1, round(w * frac))
            y = rng.integers(0, h - ph + 1)
            x = rng.integers(0, w - pw + 1)
            canvas[y:y + ph, x:x + pw] = frame[y:y + ph, x:x + pw]
        output.append(canvas.copy())
    return output


@task(name="time-patch", tags=["ram-heavy"])
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

    print(f"time_patch: {info.duration:.1f}s @ {info.fps}fps, {info.width}x{info.height}")
    print(f"  patch range={patch_min:.0%}–{patch_max:.0%}, seed={seed}")

    with _load_frames(src, c) as frames:
        output = _process_patch(frames, info.fps, patch_min=patch_min, patch_max=patch_max, seed=seed)
        return _write_frames(output, dst, info, c, "time_patch")


# ---------------------------------------------------------------------------
# 6. Slit Scan
# ---------------------------------------------------------------------------

def _process_slit_scan(
    frames: list[np.ndarray],
    fps: float,
    *,
    axis: str = "horizontal",
    scan_speed: float = 1.0,
    seed: Optional[int] = None,
) -> list[np.ndarray]:
    """Each row/column samples a different point in time."""
    rng = np.random.default_rng(seed)
    n = len(frames)
    h, w = frames[0].shape[:2]
    n_slices = h if axis == "horizontal" else w
    spread = max(1, int(n * scan_speed))
    start_offset = rng.integers(0, max(1, n))
    slice_offsets = np.round(np.linspace(0, spread, n_slices)).astype(int)

    output = []
    for out_idx in range(n):
        out_frame = np.empty((h, w, 3), dtype=np.uint8)
        for s in range(n_slices):
            src_idx = (out_idx + start_offset + slice_offsets[s]) % n
            if axis == "horizontal":
                out_frame[s, :, :] = frames[src_idx][s, :, :]
            else:
                out_frame[:, s, :] = frames[src_idx][:, s, :]
        output.append(out_frame)
    return output


@task(name="slit-scan", tags=["ram-heavy"])
def slit_scan(
    src: Path,
    dst: Path,
    *,
    axis: str = "horizontal",
    scan_speed: float = 1.0,
    seed: Optional[int] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """
    Slit-scan — each row (or column) of the output samples a different point
    in time. Horizontal motion becomes vertical distortion and vice versa.

    Output is the same duration as input.

    axis:       "horizontal" = rows sample different times (default),
                "vertical" = columns sample different times.
    scan_speed: how many frames of offset between first and last row/col
                as a fraction of total frames (default 1.0 = full span).
    seed:       random seed (controls start offset).
    """
    c = cfg or Config()
    info = probe(src, c)

    with _load_frames(src, c) as frames:
        n = len(frames)

        print(f"slit_scan: {n} frames, {info.duration:.1f}s @ {info.fps}fps")
        print(f"  axis={axis}, scan_speed={scan_speed}")
        mem_mb = n * info.width * info.height * 3 / 1e6
        print(f"  loaded {n} frames ({mem_mb:.0f} MB)")

        output = _process_slit_scan(frames, info.fps, axis=axis, scan_speed=scan_speed, seed=seed)
        return _write_frames(output, dst, info, c, "slit_scan")


# ---------------------------------------------------------------------------
# 7. Temporal Tile
# ---------------------------------------------------------------------------

def _process_temporal_tile(
    frames: list[np.ndarray],
    fps: float,
    *,
    grid: int = 4,
    offset_scale: float = 1.0,
    seed: Optional[int] = None,
) -> list[np.ndarray]:
    """Divide frame into grid tiles, each showing a different time offset."""
    rng = np.random.default_rng(seed)
    n = len(frames)
    h, w = frames[0].shape[:2]
    n_tiles = grid * grid
    max_offset = max(1, int(n * offset_scale))
    tile_offsets = rng.integers(0, max_offset, size=n_tiles)
    row_edges = np.linspace(0, h, grid + 1, dtype=int)
    col_edges = np.linspace(0, w, grid + 1, dtype=int)

    output = []
    for out_idx in range(n):
        out_frame = np.empty((h, w, 3), dtype=np.uint8)
        for ti in range(n_tiles):
            r, ci = divmod(ti, grid)
            y0, y1 = row_edges[r], row_edges[r + 1]
            x0, x1 = col_edges[ci], col_edges[ci + 1]
            src_idx = (out_idx + tile_offsets[ti]) % n
            out_frame[y0:y1, x0:x1, :] = frames[src_idx][y0:y1, x0:x1, :]
        output.append(out_frame)
    return output


@task(name="temporal-tile", tags=["ram-heavy"])
def temporal_tile(
    src: Path,
    dst: Path,
    *,
    grid: int = 4,
    offset_scale: float = 1.0,
    seed: Optional[int] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """
    Temporal tile — divide the frame into a grid, each tile shows content
    from a different point in time. Temporal mosaic / surveillance wall.

    Output is the same duration as input.

    grid:          number of tiles per axis (grid x grid). Default 4.
    offset_scale:  max time offset per tile as fraction of total frames
                   (default 1.0 = tiles can span entire clip).
    seed:          random seed (controls per-tile offsets).
    """
    c = cfg or Config()
    info = probe(src, c)

    with _load_frames(src, c) as frames:
        n = len(frames)

        print(f"temporal_tile: {n} frames, {info.duration:.1f}s @ {info.fps}fps")
        print(f"  grid={grid}x{grid}, offset_scale={offset_scale}")
        mem_mb = n * info.width * info.height * 3 / 1e6
        print(f"  loaded {n} frames ({mem_mb:.0f} MB)")

        output = _process_temporal_tile(frames, info.fps, grid=grid, offset_scale=offset_scale, seed=seed)
        return _write_frames(output, dst, info, c, "temporal_tile")


# ---------------------------------------------------------------------------
# 8. Quad Loop
# ---------------------------------------------------------------------------

def _process_quad_loop(
    frames: list[np.ndarray],
    fps: float,
    *,
    loop_dur: float = 1.0,
    offset_scale: float = 0.5,
    layout: str = "grid_2x2",
    seed: Optional[int] = None,
) -> list[np.ndarray]:
    """4 independent loops at polyrhythmic rates composited into quadrants."""
    import cv2

    rng = np.random.default_rng(seed)
    n = len(frames)
    h, w = frames[0].shape[:2]

    multipliers = [1.0, 1.25, 0.75, 1.5]
    base_loop_frames = max(2, int(loop_dur * fps))

    quad_loops = []
    max_offset = max(1, int(n * offset_scale))
    for mult in multipliers:
        loop_len = max(2, int(base_loop_frames * mult))
        start = int(rng.integers(0, max_offset))
        quad_loops.append((loop_len, start))

    if layout == "grid_2x2":
        h2, w2 = h // 2, w // 2
        regions = [
            (0, 0, h2, w2),
            (0, w2, h2, w - w2),
            (h2, 0, h - h2, w2),
            (h2, w2, h - h2, w - w2),
        ]
    elif layout == "horizontal_bands":
        band_h = h // 4
        regions = [
            (0, 0, band_h, w),
            (band_h, 0, band_h, w),
            (band_h * 2, 0, band_h, w),
            (band_h * 2 + band_h, 0, h - band_h * 3, w),
        ]
    else:  # vertical_bands
        band_w = w // 4
        regions = [
            (0, 0, h, band_w),
            (0, band_w, h, band_w),
            (0, band_w * 2, h, band_w),
            (0, band_w * 3, h, w - band_w * 3),
        ]

    # Pre-process source frames for each quadrant/band
    quad_prepared: list[list[np.ndarray]] = []
    if layout == "grid_2x2":
        # 2x2: resize full frame to fit each quadrant (aspect ratio preserved)
        for q_idx, (y0, x0, qh, qw) in enumerate(regions):
            resized = [cv2.resize(frame, (qw, qh), interpolation=cv2.INTER_AREA)
                       for frame in frames]
            quad_prepared.append(resized)
    else:
        # Band layouts: center-crop instead of squishing
        for q_idx, (y0, x0, qh, qw) in enumerate(regions):
            if layout == "horizontal_bands":
                # Crop center vertical strip to band height
                crop_y0 = (h - qh) // 2
                cropped = [frame[crop_y0:crop_y0 + qh, :] for frame in frames]
            else:  # vertical_bands
                # Crop center horizontal strip to band width
                crop_x0 = (w - qw) // 2
                cropped = [frame[:, crop_x0:crop_x0 + qw] for frame in frames]
            quad_prepared.append(cropped)

    output = []
    for out_idx in range(n):
        out_frame = np.empty((h, w, 3), dtype=np.uint8)
        for q_idx, (y0, x0, qh, qw) in enumerate(regions):
            loop_len, start = quad_loops[q_idx]
            pos = (out_idx + start) % loop_len
            src_idx = pos % n
            out_frame[y0:y0 + qh, x0:x0 + qw] = quad_prepared[q_idx][src_idx]
        output.append(out_frame)
    return output


@task(name="quad-loop", tags=["ram-heavy"])
def quad_loop(
    src: Path,
    dst: Path,
    *,
    loop_dur: float = 1.0,
    offset_scale: float = 0.5,
    layout: str = "grid_2x2",
    seed: Optional[int] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """
    Quad loop — 4 independent short loops at different speeds/offsets,
    composited into a 2x2 grid (or banded layout).

    Each quadrant shows the FULL source frame (resized to fit), looping
    at a polyrhythmic rate relative to loop_dur.

    loop_dur:      base loop length in seconds.
    offset_scale:  how far apart start offsets are (0-1, fraction of total).
    layout:        "grid_2x2", "horizontal_bands", or "vertical_bands".
    seed:          random seed for start offsets.
    """
    c = cfg or Config()
    info = probe(src, c)

    with _load_frames(src, c) as frames:
        n = len(frames)

        print(f"quad_loop: {n} frames, {info.duration:.1f}s @ {info.fps}fps")
        print(f"  layout={layout}, loop_dur={loop_dur:.2f}s, offset_scale={offset_scale:.2f}")
        mem_mb = n * info.width * info.height * 3 / 1e6
        print(f"  loaded {n} frames ({mem_mb:.0f} MB)")

        output = _process_quad_loop(
            frames, info.fps, loop_dur=loop_dur, offset_scale=offset_scale,
            layout=layout, seed=seed,
        )
        return _write_frames(output, dst, info, c, "quad_loop")


# ---------------------------------------------------------------------------
# 9. Smear
# ---------------------------------------------------------------------------

def _process_smear(
    frames: list[np.ndarray],
    fps: float,
    *,
    threshold: float = 0.1,
) -> list[np.ndarray]:
    """Pixels only update when they change beyond a threshold."""
    abs_threshold = threshold * 255.0
    canvas: Optional[np.ndarray] = None
    output = []
    for frame in frames:
        if canvas is None:
            canvas = frame.copy()
        else:
            diff = np.abs(
                frame.astype(np.int16) - canvas.astype(np.int16)
            ).max(axis=2)
            mask = diff > abs_threshold
            canvas[mask] = frame[mask]
        output.append(canvas.copy())
    return output


@task(name="smear", tags=["ram-heavy"])
def smear(
    src: Path,
    dst: Path,
    *,
    threshold: float = 0.1,
    cfg: Optional[Config] = None,
) -> Path:
    """
    Smear — pixels only update when they change beyond a threshold.
    Static areas freeze, motion areas refresh. Creates painterly erosion
    where the image accumulates ghosts of stillness.

    Single-pass streaming. One canvas buffer.

    Output is the same duration as input.

    threshold: change threshold as fraction of 255 (default 0.1).
               0.05 = very sensitive (most pixels update).
               0.2 = sticky (only strong motion breaks through).
               0.5 = extreme (only drastic changes show).
    """
    c = cfg or Config()
    info = probe(src, c)

    print(f"smear: {info.duration:.1f}s @ {info.fps}fps, {info.width}x{info.height}")
    print(f"  threshold={threshold}")

    with _load_frames(src, c) as frames:
        output = _process_smear(frames, info.fps, threshold=threshold)
        return _write_frames(output, dst, info, c, "smear")


# ---------------------------------------------------------------------------
# 10. Bloom
# ---------------------------------------------------------------------------

def _process_bloom(
    frames: list[np.ndarray],
    fps: float,
    *,
    sensitivity: float = 0.1,
) -> list[np.ndarray]:
    """Frame differencing — only show pixels that changed."""
    gain = 1.0 / max(sensitivity, 0.01)
    prev: Optional[np.ndarray] = None
    output = []
    for frame in frames:
        if prev is None:
            output.append(np.zeros_like(frame))
        else:
            diff = np.abs(frame.astype(np.float32) - prev.astype(np.float32))
            output.append(np.clip(diff * gain, 0, 255).astype(np.uint8))
        prev = frame.copy()
    return output


@task(name="bloom", tags=["ram-heavy"])
def bloom(
    src: Path,
    dst: Path,
    *,
    sensitivity: float = 0.1,
    cfg: Optional[Config] = None,
) -> Path:
    """
    Bloom — frame differencing: only show pixels that changed between
    consecutive frames. Static areas go black, motion glows.
    Temporal edge detection.

    Single-pass streaming.

    Output is the same duration as input.

    sensitivity: difference scaling factor (default 0.1).
                 Lower = only strong motion shows.
                 Higher = amplifies subtle motion.
                 The raw difference is multiplied by (1/sensitivity).
    """
    c = cfg or Config()
    info = probe(src, c)

    print(f"bloom: {info.duration:.1f}s @ {info.fps}fps, {info.width}x{info.height}")
    print(f"  sensitivity={sensitivity}, gain={1.0 / max(sensitivity, 0.01):.1f}x")

    with _load_frames(src, c) as frames:
        output = _process_bloom(frames, info.fps, sensitivity=sensitivity)
        return _write_frames(output, dst, info, c, "bloom")


# ---------------------------------------------------------------------------
# 11. Frame Stack
# ---------------------------------------------------------------------------

def _process_frame_stack(
    frames: list[np.ndarray],
    fps: float,
    *,
    window: int = 8,
    mode: str = "mean",
) -> list[np.ndarray]:
    """Sliding window average/max/min of frames."""
    ring: deque[np.ndarray] = deque(maxlen=window)
    reduce_fn = {
        "mean": lambda buf: np.mean(buf, axis=0).astype(np.uint8),
        "max": lambda buf: np.max(buf, axis=0).astype(np.uint8),
        "min": lambda buf: np.min(buf, axis=0).astype(np.uint8),
    }[mode]
    output = []
    for frame in frames:
        ring.append(frame.astype(np.float32) if mode == "mean" else frame)
        buf = np.stack(list(ring), axis=0)
        output.append(reduce_fn(buf))
    return output


@task(name="frame-stack", tags=["ram-heavy"])
def frame_stack(
    src: Path,
    dst: Path,
    *,
    window: int = 8,
    mode: str = "mean",
    cfg: Optional[Config] = None,
) -> Path:
    """
    Stack — average (or max/min) a sliding window of frames together.
    Long-exposure photography for video. Equal-weight window creates
    a dreamy blur rather than exponential echo trails.

    Streaming with a ring buffer of `window` frames.

    Output is the same duration as input.

    window: number of frames to accumulate (default 8).
    mode:   "mean" (dreamy blur), "max" (brightest survives),
            "min" (darkest survives). Default "mean".
    """
    c = cfg or Config()
    info = probe(src, c)

    print(f"frame_stack: {info.duration:.1f}s @ {info.fps}fps, "
          f"{info.width}x{info.height}")
    mem_mb = window * info.width * info.height * 3 / 1e6
    print(f"  window={window} frames, mode={mode} ({mem_mb:.0f} MB buffer)")

    with _load_frames(src, c) as frames:
        output = _process_frame_stack(frames, info.fps, window=window, mode=mode)
        return _write_frames(output, dst, info, c, "frame_stack")


# ---------------------------------------------------------------------------
# 12. Slip
# ---------------------------------------------------------------------------

def _process_slip(
    frames: list[np.ndarray],
    fps: float,
    *,
    n_bands: int = 8,
    max_slip: float = 0.5,
    axis: str = "horizontal",
    seed: Optional[int] = None,
) -> list[np.ndarray]:
    """Offset bands of scanlines by different amounts in time."""
    rng = np.random.default_rng(seed)
    n = len(frames)
    h, w = frames[0].shape[:2]
    max_offset = max(1, int(n * max_slip))
    band_offsets = rng.integers(-max_offset, max_offset + 1, size=n_bands)
    dim = h if axis == "horizontal" else w
    band_edges = np.linspace(0, dim, n_bands + 1, dtype=int)

    output = []
    for out_idx in range(n):
        out_frame = np.empty((h, w, 3), dtype=np.uint8)
        for b in range(n_bands):
            src_idx = (out_idx + band_offsets[b]) % n
            if axis == "horizontal":
                y0, y1 = band_edges[b], band_edges[b + 1]
                out_frame[y0:y1, :, :] = frames[src_idx][y0:y1, :, :]
            else:
                x0, x1 = band_edges[b], band_edges[b + 1]
                out_frame[:, x0:x1, :] = frames[src_idx][:, x0:x1, :]
        output.append(out_frame)
    return output


@task(name="slip", tags=["ram-heavy"])
def slip(
    src: Path,
    dst: Path,
    *,
    n_bands: int = 8,
    max_slip: float = 0.5,
    axis: str = "horizontal",
    seed: Optional[int] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """
    Slip — offset bands of scanlines by different amounts in time.
    Each horizontal (or vertical) band shows a slightly different moment,
    like temporal interlacing artifacts.

    All source frames loaded into memory.

    Output is the same duration as input.

    n_bands:   number of bands to split the frame into (default 8).
    max_slip:  max time offset per band as fraction of total frames
               (default 0.5).
    axis:      "horizontal" = rows banded (default),
               "vertical" = columns banded.
    seed:      random seed (controls per-band offsets).
    """
    c = cfg or Config()
    info = probe(src, c)

    with _load_frames(src, c) as frames:
        n = len(frames)

        print(f"slip: {n} frames, {info.duration:.1f}s @ {info.fps}fps")
        print(f"  {n_bands} bands ({axis}), max_slip={max_slip}")
        mem_mb = n * info.width * info.height * 3 / 1e6
        print(f"  loaded {n} frames ({mem_mb:.0f} MB)")

        output = _process_slip(frames, info.fps, n_bands=n_bands, max_slip=max_slip, axis=axis, seed=seed)
        return _write_frames(output, dst, info, c, "slip")


# ---------------------------------------------------------------------------
# 13. Flow Warp
# ---------------------------------------------------------------------------

def _process_flow_warp(
    frames: list[np.ndarray],
    fps: float,
    *,
    amplify: float = 3.0,
    smooth: int = 15,
    seed: Optional[int] = None,
) -> list[np.ndarray]:
    """Optical-flow motion exaggeration."""
    import cv2

    smooth = max(3, smooth | 1)
    h, w = frames[0].shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                  np.arange(h, dtype=np.float32))
    prev_gray: Optional[np.ndarray] = None
    output = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if prev_gray is None:
            output.append(frame)
        else:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )
            flow[:, :, 0] = cv2.GaussianBlur(flow[:, :, 0], (smooth, smooth), 0)
            flow[:, :, 1] = cv2.GaussianBlur(flow[:, :, 1], (smooth, smooth), 0)
            map_x = grid_x + flow[:, :, 0] * amplify
            map_y = grid_y + flow[:, :, 1] * amplify
            warped = cv2.remap(
                frame, map_x, map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )
            output.append(warped)
        prev_gray = gray
    return output


@task(name="flow-warp", tags=["ram-heavy"])
def flow_warp(
    src: Path,
    dst: Path,
    *,
    amplify: float = 3.0,
    smooth: int = 15,
    seed: Optional[int] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """
    Optical-flow motion exaggeration — amplify existing motion in video.

    Computes dense optical flow between consecutive frames, then warps
    each frame by amplified flow vectors. Things that are already moving
    move MORE. Static areas stay still.

    Single-pass streaming. One previous-frame buffer.

    Output is the same duration as input.

    amplify: flow multiplier (1.0 = natural, 3.0 = 3x exaggerated,
             negative = reverse motion). Default 3.0.
    smooth:  Gaussian blur kernel size for flow field smoothing.
             Higher = smoother warps, fewer artifacts. Default 15.
             Must be odd.
    """
    c = cfg or Config()
    info = probe(src, c)

    print(f"flow_warp: {info.duration:.1f}s @ {info.fps}fps, {info.width}x{info.height}")
    print(f"  amplify={amplify}, smooth={smooth}")

    with _load_frames(src, c) as frames:
        output = _process_flow_warp(frames, info.fps, amplify=amplify, smooth=smooth, seed=seed)
        return _write_frames(output, dst, info, c, "flow_warp")


# ---------------------------------------------------------------------------
# 14. Temporal Sort
# ---------------------------------------------------------------------------

def _process_temporal_sort(
    frames: list[np.ndarray],
    fps: float,
    *,
    mode: str = "luminance",
    direction: str = "ascending",
    seed: Optional[int] = None,
) -> list[np.ndarray]:
    """Sort pixels along time axis by luminance or channel."""
    n = len(frames)
    h, w = frames[0].shape[:2]

    vol = np.stack(frames, axis=0)  # (T, H, W, 3)

    if mode == "luminance":
        key = (0.299 * vol[:, :, :, 0].astype(np.float32)
               + 0.587 * vol[:, :, :, 1].astype(np.float32)
               + 0.114 * vol[:, :, :, 2].astype(np.float32))
    elif mode in ("red", "green", "blue"):
        ch = {"red": 0, "green": 1, "blue": 2}[mode]
        key = vol[:, :, :, ch].astype(np.float32)
    else:
        key = vol[:, :, :, 0].astype(np.float32)

    order = np.argsort(key, axis=0)
    if direction == "descending":
        order = order[::-1]

    sorted_vol = np.empty_like(vol)
    for ch in range(3):
        channel = vol[:, :, :, ch]
        sorted_vol[:, :, :, ch] = np.take_along_axis(channel, order, axis=0)

    return [sorted_vol[i] for i in range(n)]


@task(name="temporal-sort", tags=["ram-heavy"])
def temporal_sort(
    src: Path,
    dst: Path,
    *,
    mode: str = "luminance",
    direction: str = "ascending",
    seed: Optional[int] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """
    Temporal sort — for each pixel position, sort all values across time
    by luminance (or by channel), then play the sorted sequence back.

    Every pixel independently transitions from its darkest to brightest
    moment (ascending) or vice versa (descending). Static parts stay
    constant; moving parts create a strange chromatic dissolve that has
    no relationship to the original timeline.

    Output is the same duration as input.

    mode:      "luminance" (sort by Y), "red", "green", "blue"
               (sort by that channel). Default "luminance".
    direction: "ascending" (dark→bright) or "descending" (bright→dark).
               Default "ascending".
    seed:      unused, kept for interface consistency.
    """
    c = cfg or Config()
    info = probe(src, c)

    with _load_frames(src, c) as frames:
        n = len(frames)

        print(f"temporal_sort: {n} frames, {info.duration:.1f}s @ {info.fps}fps")
        print(f"  mode={mode}, direction={direction}")
        mem_mb = n * info.width * info.height * 3 / 1e6
        print(f"  loaded {n} frames ({mem_mb:.0f} MB)")

        t0 = _time.monotonic()
        output = _process_temporal_sort(frames, info.fps, mode=mode, direction=direction, seed=seed)
        sort_elapsed = _time.monotonic() - t0
        print(f"  sorted in {sort_elapsed:.1f}s")

        return _write_frames(output, dst, info, c, "temporal_sort")


# ---------------------------------------------------------------------------
# 14b. Temporal FFT Filtering
# ---------------------------------------------------------------------------

def _smooth_gate(freqs: np.ndarray, cutoff: float, below: bool) -> np.ndarray:
    """Smooth sigmoid frequency gate. below=True passes frequencies below cutoff."""
    k = 30.0
    if below:
        return 1.0 / (1.0 + np.exp(k * (freqs - cutoff)))
    else:
        return 1.0 / (1.0 + np.exp(-k * (freqs - cutoff)))


def _process_temporal_fft(
    frames: list[np.ndarray],
    fps: float,
    *,
    filter_type: str = "low_pass",
    cutoff_low: float = 0.1,
    cutoff_high: float = 0.5,
    preserve_dc: bool = True,
    seed: Optional[int] = None,
) -> list[np.ndarray]:
    """Filter frame volume along time axis via FFT."""
    n = len(frames)
    if n < 4:
        return list(frames)

    # Stack to (T, H, W, 3) float32 volume
    vol = np.stack(frames, axis=0).astype(np.float32)

    # FFT along time axis (axis=0) — per-pixel, per-channel
    spectrum = np.fft.rfft(vol, axis=0)  # (T//2+1, H, W, 3) complex

    # Save DC bin before filtering
    dc = spectrum[0].copy() if preserve_dc else None

    # Build frequency mask (T//2+1,) — broadcast across spatial dims
    n_bins = spectrum.shape[0]
    freqs = np.linspace(0, 1, n_bins)  # 0 = DC, 1 = Nyquist

    if filter_type == "low_pass":
        mask = _smooth_gate(freqs, cutoff_low, below=True)
    elif filter_type == "high_pass":
        mask = _smooth_gate(freqs, cutoff_low, below=False)
    elif filter_type == "band_pass":
        mask = (_smooth_gate(freqs, cutoff_low, below=False)
                * _smooth_gate(freqs, cutoff_high, below=True))
    elif filter_type == "notch":
        mask = 1.0 - (_smooth_gate(freqs, cutoff_low, below=False)
                      * _smooth_gate(freqs, cutoff_high, below=True))
    else:
        mask = np.ones(n_bins)

    # Apply mask (broadcast: (n_bins, 1, 1, 1))
    spectrum *= mask[:, np.newaxis, np.newaxis, np.newaxis]

    # Restore DC if requested
    if dc is not None:
        spectrum[0] = dc

    # Inverse FFT, clip, convert
    filtered = np.fft.irfft(spectrum, n=n, axis=0)
    filtered = np.clip(filtered, 0, 255).astype(np.uint8)

    return [filtered[i] for i in range(n)]


@task(name="temporal-fft", tags=["ram-heavy"])
def temporal_fft(
    src: Path,
    dst: Path,
    *,
    filter_type: str = "low_pass",
    cutoff_low: float = 0.1,
    cutoff_high: float = 0.5,
    preserve_dc: bool = True,
    seed: Optional[int] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """
    Temporal FFT filtering — manipulate frequency content along the time axis.

    Applies FFT per-pixel along the time dimension, filters in the frequency
    domain, then inverse-transforms back. Produces effects impossible to
    achieve frame-by-frame.

    filter_type: "low_pass" (ghostly blur), "high_pass" (flicker isolation),
                 "band_pass" (rhythmic extraction), "notch" (deflicker).
    cutoff_low:  normalized frequency 0–1 (fraction of Nyquist).
    cutoff_high: upper bound for band_pass/notch.
    preserve_dc: keep DC bin to preserve average brightness. Default True.
    seed:        unused, kept for interface consistency.
    """
    c = cfg or Config()
    info = probe(src, c)

    with _load_frames(src, c) as frames:
        n = len(frames)

        print(f"temporal_fft: {n} frames, {info.duration:.1f}s @ {info.fps}fps")
        print(f"  filter_type={filter_type}, cutoff_low={cutoff_low:.2f}, "
              f"cutoff_high={cutoff_high:.2f}")
        mem_mb = n * info.width * info.height * 3 / 1e6
        print(f"  loaded {n} frames ({mem_mb:.0f} MB)")

        t0 = _time.monotonic()
        output = _process_temporal_fft(
            frames, info.fps,
            filter_type=filter_type, cutoff_low=cutoff_low,
            cutoff_high=cutoff_high, preserve_dc=preserve_dc, seed=seed,
        )
        elapsed = _time.monotonic() - t0
        print(f"  filtered in {elapsed:.1f}s")

        return _write_frames(output, dst, info, c, "temporal_fft")


# ---------------------------------------------------------------------------
# 14b. Temporal Gradient
# ---------------------------------------------------------------------------

def _process_temporal_gradient(
    frames: list[np.ndarray],
    fps: float,
    *,
    order: int = 1,
    seed: Optional[int] = None,
) -> list[np.ndarray]:
    """Per-pixel temporal derivative — only motion survives.

    order=1: first derivative (velocity — edges of motion).
    order=2: second derivative (acceleration — direction changes).
    """
    n = len(frames)
    if n < order + 1:
        return list(frames)

    vol = np.stack(frames, axis=0).astype(np.float32)

    # np.diff along time axis, repeated for higher orders
    diff = vol
    for _ in range(order):
        diff = np.diff(diff, axis=0)  # (T-order, H, W, 3)

    # Shift to mid-gray (128) so negative changes are visible,
    # then scale to use the full 0–255 range.
    # Use robust scaling: scale by 2*std, clipped to [0, 255].
    std = diff.std()
    if std > 0:
        diff = diff / (2.0 * std)  # ~95% of values in [-1, 1]
    diff = (diff * 127.5) + 128.0
    diff = np.clip(diff, 0, 255).astype(np.uint8)

    # Pad back to original frame count by repeating first/last frames
    result = list(diff[i] for i in range(diff.shape[0]))
    # Pad start
    for _ in range(order):
        result.insert(0, result[0])
    return result


@task(name="temporal-gradient", tags=["ram-heavy"])
def temporal_gradient(
    src: Path,
    dst: Path,
    *,
    order: int = 1,
    seed: Optional[int] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """Temporal gradient — per-pixel temporal derivative.

    Only motion survives. Static regions go to mid-gray.
    order=1: velocity (motion edges). order=2: acceleration (direction changes).
    """
    c = cfg or Config()
    info = probe(src, c)

    with _load_frames(src, c) as frames:
        n = len(frames)
        print(f"temporal_gradient: {n} frames, {info.duration:.1f}s @ {info.fps}fps")
        print(f"  order={order}")

        t0 = _time.monotonic()
        output = _process_temporal_gradient(frames, info.fps, order=order, seed=seed)
        elapsed = _time.monotonic() - t0
        print(f"  processed in {elapsed:.1f}s")

        return _write_frames(output, dst, info, c, "temporal_gradient")


# ---------------------------------------------------------------------------
# 14c. Temporal Median
# ---------------------------------------------------------------------------

def _process_temporal_median(
    frames: list[np.ndarray],
    fps: float,
    *,
    window: int = 7,
    seed: Optional[int] = None,
) -> list[np.ndarray]:
    """Rolling median along time axis — removes transient motion.

    window: number of frames in the median kernel (must be odd).
    Small window (3–5): softens motion while keeping structure.
    Large window (15–31): ghost-like smear, only persistent features remain.
    """
    n = len(frames)
    if n < 3:
        return list(frames)

    # Ensure odd window
    window = max(3, window | 1)  # force odd
    window = min(window, n)      # can't exceed frame count

    vol = np.stack(frames, axis=0)  # (T, H, W, 3) uint8

    # Rolling median along time axis using numpy — no scipy dependency.
    # Pad the volume temporally with edge values, then slide a window.
    half = window // 2
    padded = np.pad(vol, ((half, half), (0, 0), (0, 0), (0, 0)), mode="edge")
    # Sliding window view: (T, window, H, W, 3) — zero-copy
    shape = (n, window) + vol.shape[1:]
    strides = (padded.strides[0],) + padded.strides
    windowed = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
    # Median along the window axis (axis=1)
    filtered = np.median(windowed, axis=1).astype(np.uint8)

    return [filtered[i] for i in range(n)]


@task(name="temporal-median", tags=["ram-heavy"])
def temporal_median(
    src: Path,
    dst: Path,
    *,
    window: int = 7,
    seed: Optional[int] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """Temporal median — rolling median along time axis.

    Removes transient motion; only features persisting for window/2 frames survive.
    window: median kernel size in frames (odd, default 7).
    """
    c = cfg or Config()
    info = probe(src, c)

    with _load_frames(src, c) as frames:
        n = len(frames)
        print(f"temporal_median: {n} frames, {info.duration:.1f}s @ {info.fps}fps")
        print(f"  window={window}")

        t0 = _time.monotonic()
        output = _process_temporal_median(frames, info.fps, window=window, seed=seed)
        elapsed = _time.monotonic() - t0
        print(f"  processed in {elapsed:.1f}s")

        return _write_frames(output, dst, info, c, "temporal_median")


# ---------------------------------------------------------------------------
# 14d. Axis Swap
# ---------------------------------------------------------------------------

def _process_axis_swap(
    frames: list[np.ndarray],
    fps: float,
    *,
    axis: str = "horizontal",
    seed: Optional[int] = None,
) -> list[np.ndarray]:
    """Swap time axis with spatial axis — view the volume from the side.

    axis="horizontal": swap T↔X. Each output frame is a temporal cross-section
        at a different x-position. Horizontal axis shows timeline.
    axis="vertical": swap T↔Y. Each output frame is a temporal cross-section
        at a different y-position. Vertical axis shows timeline.
    """
    n = len(frames)
    if n < 2:
        return list(frames)

    H, W = frames[0].shape[:2]
    vol = np.stack(frames, axis=0)  # (T, H, W, 3) uint8
    T = n

    output = np.empty_like(vol)  # (T, H, W, 3)

    if axis == "horizontal":
        # Swap T↔X: output[t, y, x] = vol[x_to_t[x], y, t_to_x[t]]
        t_to_x = np.round(np.linspace(0, W - 1, T)).astype(np.intp)
        x_to_t = np.round(np.linspace(0, T - 1, W)).astype(np.intp)
        for t in range(T):
            col = t_to_x[t]
            # vol[x_to_t, :, col, :] → (W, H, 3), transpose to (H, W, 3)
            output[t] = vol[x_to_t, :, col, :].transpose(1, 0, 2)
    else:
        # Swap T↔Y: output[t, y, x] = vol[y_to_t[y], t_to_y[t], x]
        t_to_y = np.round(np.linspace(0, H - 1, T)).astype(np.intp)
        y_to_t = np.round(np.linspace(0, T - 1, H)).astype(np.intp)
        for t in range(T):
            row = t_to_y[t]
            # vol[y_to_t, row, :, :] → (H, W, 3) — directly correct shape
            output[t] = vol[y_to_t, row, :, :]

    return [output[i] for i in range(T)]


# ---------------------------------------------------------------------------
# 19. Temporal Morph (morphological ops along time)
# ---------------------------------------------------------------------------

def _process_temporal_morph(
    frames: list[np.ndarray],
    fps: float,
    *,
    operation: str = "dilate",
    window: int = 5,
    seed: Optional[int] = None,
) -> list[np.ndarray]:
    """Morphological operations along the time axis.

    dilate:  pixel stays bright if bright in ANY frame within window.
    erode:   pixel stays bright only if bright in ALL frames within window.
    open:    erode then dilate — removes bright transients.
    close:   dilate then erode — fills dark transients.
    """
    n = len(frames)
    if n < 3:
        return list(frames)

    vol = np.stack(frames, axis=0)  # (T, H, W, 3) uint8
    T = vol.shape[0]
    window = max(3, window | 1)  # force odd
    window = min(window, T)
    half = window // 2

    def _sliding_op(data: np.ndarray, op) -> np.ndarray:
        padded = np.pad(data, ((half, half), (0, 0), (0, 0), (0, 0)), mode="edge")
        shape = (T, window) + data.shape[1:]
        strides = (padded.strides[0],) + padded.strides
        windowed = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
        return op(windowed, axis=1).astype(np.uint8)

    if operation == "dilate":
        result = _sliding_op(vol, np.max)
    elif operation == "erode":
        result = _sliding_op(vol, np.min)
    elif operation == "open":
        result = _sliding_op(_sliding_op(vol, np.min), np.max)
    elif operation == "close":
        result = _sliding_op(_sliding_op(vol, np.max), np.min)
    else:
        result = vol

    return [result[i] for i in range(T)]


# ---------------------------------------------------------------------------
# 20. Depth Slice (angled scan through spacetime)
# ---------------------------------------------------------------------------

def _process_depth_slice(
    frames: list[np.ndarray],
    fps: float,
    *,
    angle: float = 45.0,
    axis: str = "horizontal",
    seed: Optional[int] = None,
) -> list[np.ndarray]:
    """Sweep a plane through the (T,X,Y) volume at an arbitrary angle.

    angle=0 → identity (pure spatial). angle=90 → axis_swap.
    Intermediate angles blend time and space continuously.
    """
    n = len(frames)
    if n < 2:
        return list(frames)

    vol = np.stack(frames, axis=0)  # (T, H, W, 3)
    T, H, W = vol.shape[:3]
    output = np.empty_like(vol)

    rad = np.radians(np.clip(angle, 0.0, 90.0))
    cos_a = np.cos(rad)
    sin_a = np.sin(rad)

    if axis == "horizontal":
        x_norm = np.linspace(0, 1, W)  # (W,)
        for t in range(T):
            t_norm = t / max(T - 1, 1)
            t_sample = t_norm * cos_a + x_norm * sin_a
            t_idx = np.clip((t_sample * (T - 1)).astype(np.intp), 0, T - 1)
            output[t] = vol[t_idx, :, np.arange(W)].transpose(1, 0, 2)
    else:
        y_norm = np.linspace(0, 1, H)  # (H,)
        for t in range(T):
            t_norm = t / max(T - 1, 1)
            t_sample = t_norm * cos_a + y_norm * sin_a
            t_idx = np.clip((t_sample * (T - 1)).astype(np.intp), 0, T - 1)
            output[t] = vol[t_idx, np.arange(H), :]

    return [output[i] for i in range(T)]


# ---------------------------------------------------------------------------
# 21. Temporal Equalize (per-pixel histogram equalization along time)
# ---------------------------------------------------------------------------

def _process_temporal_equalize(
    frames: list[np.ndarray],
    fps: float,
    *,
    strength: float = 1.0,
    seed: Optional[int] = None,
) -> list[np.ndarray]:
    """Per-pixel histogram equalization along the time axis.

    Each pixel is individually rank-normalized across its temporal history,
    forcing every pixel to use its full dynamic range over time.
    """
    n = len(frames)
    if n < 3:
        return list(frames)

    vol = np.stack(frames, axis=0).astype(np.float32)  # (T, H, W, 3)
    T = vol.shape[0]

    order = np.argsort(vol, axis=0)
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(T)[:, None, None, None], axis=0)

    equalized = (ranks.astype(np.float32) / max(T - 1, 1)) * 255.0
    result = vol * (1.0 - strength) + equalized * strength
    result = np.clip(result, 0, 255).astype(np.uint8)
    return [result[i] for i in range(T)]


# ---------------------------------------------------------------------------
# 22. Temporal Displace (self-referential time warp)
# ---------------------------------------------------------------------------

def _process_temporal_displace(
    frames: list[np.ndarray],
    fps: float,
    *,
    amount: float = 0.5,
    channel: str = "luma",
    seed: Optional[int] = None,
) -> list[np.ndarray]:
    """Use each pixel's brightness to index into the time axis.

    Bright pixels show future frames, dark pixels show past frames.
    amount controls how far in time the displacement can reach.
    """
    n = len(frames)
    if n < 3:
        return list(frames)

    vol = np.stack(frames, axis=0).astype(np.float32)  # (T, H, W, 3)
    T, H, W = vol.shape[:3]

    ch_map = {"r": 0, "g": 1, "b": 2}
    if channel in ch_map:
        disp = vol[..., ch_map[channel]]
    else:  # luma
        disp = 0.2126 * vol[..., 0] + 0.7152 * vol[..., 1] + 0.0722 * vol[..., 2]
    disp = disp / 255.0  # (T, H, W) in [0, 1]

    offset = (disp - 0.5) * 2.0 * amount * T
    t_base = np.arange(T)[:, None, None]
    lookup = np.clip(t_base + offset, 0, T - 1).astype(np.intp)

    y_idx = np.arange(H)[None, :, None]
    x_idx = np.arange(W)[None, None, :]
    result = vol[lookup, y_idx, x_idx, :]
    return [np.clip(result[i], 0, 255).astype(np.uint8) for i in range(T)]


# ---------------------------------------------------------------------------
# 23. Spectral Remix (FFT frequency bin rearrangement)
# ---------------------------------------------------------------------------

def _process_spectral_remix(
    frames: list[np.ndarray],
    fps: float,
    *,
    mode: str = "swap",
    amount: float = 0.3,
    seed: Optional[int] = None,
) -> list[np.ndarray]:
    """FFT along time, rearrange frequency bins, IFFT back.

    swap:    exchange low and high halves.
    reverse: reverse the frequency bin order.
    rotate:  circular shift bins by quarter.
    shuffle: random permutation of bins.

    amount:  blend 0=original 1=fully rearranged. Low values (0.2-0.4)
             give subtle temporal texture shift without flicker.
    """
    n = len(frames)
    if n < 4:
        return list(frames)

    vol = np.stack(frames, axis=0).astype(np.float32)
    spectrum = np.fft.rfft(vol, axis=0)
    dc = spectrum[0:1].copy()

    bins_orig = spectrum[1:]
    nb = len(bins_orig)
    if nb < 2:
        return list(frames)

    if mode == "swap":
        mid = nb // 2
        bins_new = np.concatenate([bins_orig[mid:], bins_orig[:mid]], axis=0)
    elif mode == "reverse":
        bins_new = bins_orig[::-1]
    elif mode == "rotate":
        shift = nb // 4 or 1
        bins_new = np.concatenate([bins_orig[shift:], bins_orig[:shift]], axis=0)
    elif mode == "shuffle":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(nb)
        bins_new = bins_orig[perm]
    else:
        bins_new = bins_orig

    # Blend original and rearranged to control intensity
    bins_blended = (1.0 - amount) * bins_orig + amount * bins_new

    spectrum_new = np.concatenate([dc, bins_blended], axis=0)
    filtered = np.fft.irfft(spectrum_new, n=n, axis=0)
    return [np.clip(filtered[i], 0, 255).astype(np.uint8) for i in range(n)]


# ---------------------------------------------------------------------------
# 24. Phase Scramble (randomize phases, preserve magnitudes)
# ---------------------------------------------------------------------------

def _process_phase_scramble(
    frames: list[np.ndarray],
    fps: float,
    *,
    amount: float = 1.0,
    seed: Optional[int] = None,
) -> list[np.ndarray]:
    """FFT along time, randomize phases while keeping magnitudes.

    Preserves the power spectrum (same temporal texture/statistics)
    but destroys temporal coherence.
    amount=1.0: complete phase randomization.
    amount=0.3: dreamlike desynchronization.
    """
    n = len(frames)
    if n < 4:
        return list(frames)

    rng = np.random.default_rng(seed)
    vol = np.stack(frames, axis=0).astype(np.float32)
    spectrum = np.fft.rfft(vol, axis=0)

    magnitudes = np.abs(spectrum)
    phases = np.angle(spectrum)

    random_phases = rng.uniform(-np.pi, np.pi,
                                size=phases.shape).astype(np.float32)
    new_phases = phases + amount * random_phases

    spectrum_new = magnitudes * np.exp(1j * new_phases)
    filtered = np.fft.irfft(spectrum_new, n=n, axis=0)
    return [np.clip(filtered[i], 0, 255).astype(np.uint8) for i in range(n)]


@task(name="axis-swap", tags=["ram-heavy"])
def axis_swap(
    src: Path,
    dst: Path,
    *,
    axis: str = "horizontal",
    seed: Optional[int] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """Axis swap — view the frame volume from the side.

    Swaps the time axis with a spatial axis. Each output frame shows what used
    to be a spatial cross-section, with time stretched across the frame.
    axis: "horizontal" (swap T↔X) or "vertical" (swap T↔Y).
    """
    c = cfg or Config()
    info = probe(src, c)

    with _load_frames(src, c) as frames:
        n = len(frames)
        print(f"axis_swap: {n} frames, {info.duration:.1f}s @ {info.fps}fps")
        print(f"  axis={axis}")

        t0 = _time.monotonic()
        output = _process_axis_swap(frames, info.fps, axis=axis, seed=seed)
        elapsed = _time.monotonic() - t0
        print(f"  processed in {elapsed:.1f}s")

        return _write_frames(output, dst, info, c, "axis_swap")


@task(name="temporal-morph", tags=["ram-heavy"])
def temporal_morph(
    src: Path,
    dst: Path,
    *,
    operation: str = "dilate",
    window: int = 5,
    seed: Optional[int] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """Morphological operations along the time axis."""
    c = cfg or Config()
    info = probe(src, c)
    with _load_frames(src, c) as frames:
        n = len(frames)
        print(f"temporal_morph: {n} frames, {info.duration:.1f}s @ {info.fps}fps")
        print(f"  operation={operation}, window={window}")
        t0 = _time.monotonic()
        output = _process_temporal_morph(frames, info.fps, operation=operation,
                                         window=window, seed=seed)
        print(f"  processed in {_time.monotonic() - t0:.1f}s")
        return _write_frames(output, dst, info, c, "temporal_morph")


@task(name="depth-slice", tags=["ram-heavy"])
def depth_slice(
    src: Path,
    dst: Path,
    *,
    angle: float = 45.0,
    axis: str = "horizontal",
    seed: Optional[int] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """Angled scan plane through spacetime volume."""
    c = cfg or Config()
    info = probe(src, c)
    with _load_frames(src, c) as frames:
        n = len(frames)
        print(f"depth_slice: {n} frames, {info.duration:.1f}s @ {info.fps}fps")
        print(f"  angle={angle}, axis={axis}")
        t0 = _time.monotonic()
        output = _process_depth_slice(frames, info.fps, angle=angle,
                                       axis=axis, seed=seed)
        print(f"  processed in {_time.monotonic() - t0:.1f}s")
        return _write_frames(output, dst, info, c, "depth_slice")


@task(name="temporal-equalize", tags=["ram-heavy"])
def temporal_equalize(
    src: Path,
    dst: Path,
    *,
    strength: float = 1.0,
    seed: Optional[int] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """Per-pixel histogram equalization along time axis."""
    c = cfg or Config()
    info = probe(src, c)
    with _load_frames(src, c) as frames:
        n = len(frames)
        print(f"temporal_equalize: {n} frames, {info.duration:.1f}s @ {info.fps}fps")
        print(f"  strength={strength}")
        t0 = _time.monotonic()
        output = _process_temporal_equalize(frames, info.fps, strength=strength,
                                             seed=seed)
        print(f"  processed in {_time.monotonic() - t0:.1f}s")
        return _write_frames(output, dst, info, c, "temporal_equalize")


@task(name="temporal-displace", tags=["ram-heavy"])
def temporal_displace(
    src: Path,
    dst: Path,
    *,
    amount: float = 0.5,
    channel: str = "luma",
    seed: Optional[int] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """Self-referential time warp — brightness indexes into time."""
    c = cfg or Config()
    info = probe(src, c)
    with _load_frames(src, c) as frames:
        n = len(frames)
        print(f"temporal_displace: {n} frames, {info.duration:.1f}s @ {info.fps}fps")
        print(f"  amount={amount}, channel={channel}")
        t0 = _time.monotonic()
        output = _process_temporal_displace(frames, info.fps, amount=amount,
                                             channel=channel, seed=seed)
        print(f"  processed in {_time.monotonic() - t0:.1f}s")
        return _write_frames(output, dst, info, c, "temporal_displace")


@task(name="spectral-remix", tags=["ram-heavy"])
def spectral_remix(
    src: Path,
    dst: Path,
    *,
    mode: str = "swap",
    amount: float = 0.3,
    seed: Optional[int] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """FFT frequency bin rearrangement along time axis."""
    c = cfg or Config()
    info = probe(src, c)
    with _load_frames(src, c) as frames:
        n = len(frames)
        print(f"spectral_remix: {n} frames, {info.duration:.1f}s @ {info.fps}fps")
        print(f"  mode={mode}, amount={amount}")
        t0 = _time.monotonic()
        output = _process_spectral_remix(frames, info.fps, mode=mode, amount=amount, seed=seed)
        print(f"  processed in {_time.monotonic() - t0:.1f}s")
        return _write_frames(output, dst, info, c, "spectral_remix")


@task(name="phase-scramble", tags=["ram-heavy"])
def phase_scramble(
    src: Path,
    dst: Path,
    *,
    amount: float = 1.0,
    seed: Optional[int] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """Randomize temporal phases while preserving frequency magnitudes."""
    c = cfg or Config()
    info = probe(src, c)
    with _load_frames(src, c) as frames:
        n = len(frames)
        print(f"phase_scramble: {n} frames, {info.duration:.1f}s @ {info.fps}fps")
        print(f"  amount={amount}")
        t0 = _time.monotonic()
        output = _process_phase_scramble(frames, info.fps, amount=amount, seed=seed)
        print(f"  processed in {_time.monotonic() - t0:.1f}s")
        return _write_frames(output, dst, info, c, "phase_scramble")


# ---------------------------------------------------------------------------
# 15. Extrema Hold
# ---------------------------------------------------------------------------

def _process_extrema_hold(
    frames: list[np.ndarray],
    fps: float,
    *,
    mode: str = "max",
    decay: float = 0.0,
    seed: Optional[int] = None,
) -> list[np.ndarray]:
    """Per-pixel max/min accumulation with optional decay."""
    canvas_max: Optional[np.ndarray] = None
    canvas_min: Optional[np.ndarray] = None
    output = []
    for frame in frames:
        f = frame.astype(np.float32)
        if canvas_max is None:
            canvas_max = f.copy()
            canvas_min = f.copy()
        else:
            if decay > 0:
                canvas_max += (f - canvas_max) * decay
                canvas_min += (f - canvas_min) * decay
            np.maximum(canvas_max, f, out=canvas_max)
            np.minimum(canvas_min, f, out=canvas_min)

        if mode == "max":
            out = canvas_max
        elif mode == "min":
            out = canvas_min
        else:  # "both"
            out = np.stack([
                canvas_max[:, :, 0],
                f[:, :, 1],
                canvas_min[:, :, 2],
            ], axis=2)
        output.append(np.clip(out, 0, 255).astype(np.uint8))
    return output


@task(name="extrema-hold", tags=["ram-heavy"])
def extrema_hold(
    src: Path,
    dst: Path,
    *,
    mode: str = "max",
    decay: float = 0.0,
    seed: Optional[int] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """
    Extrema hold — each pixel accumulates the brightest (or darkest)
    value it has reached so far. Motion leaves permanent trails;
    static areas stay put. Like long-exposure photography for video.

    With decay > 0, the held extrema slowly relax back toward the
    current frame, so trails fade over time rather than being permanent.

    Single-pass streaming. One canvas buffer.

    Output is the same duration as input.

    mode:  "max" (brightest survives — bright trails on dark background),
           "min" (darkest survives — dark trails on bright background),
           "both" (R=max hold, G=current, B=min hold — split channels).
           Default "max".
    decay: how fast extrema relax back toward current frame, 0–1.
           0 = permanent hold (pure accumulation).
           0.01 = very slow fade (trails last many seconds).
           0.1 = moderate fade.
           Default 0.0.
    seed:  unused, kept for interface consistency.
    """
    c = cfg or Config()
    info = probe(src, c)

    print(f"extrema_hold: {info.duration:.1f}s @ {info.fps}fps, "
          f"{info.width}x{info.height}")
    print(f"  mode={mode}, decay={decay}")

    with _load_frames(src, c) as frames:
        output = _process_extrema_hold(frames, info.fps, mode=mode, decay=decay, seed=seed)
        return _write_frames(output, dst, info, c, "extrema_hold")


# ---------------------------------------------------------------------------
# 16. Feedback Transform
# ---------------------------------------------------------------------------

def _process_feedback_transform(
    frames: list[np.ndarray],
    fps: float,
    *,
    transform: str = "zoom",
    amount: float = 0.02,
    mix: float = 0.7,
    seed: Optional[int] = None,
) -> list[np.ndarray]:
    """Recursive spatial transform on previous output blended with current."""
    import cv2

    h, w = frames[0].shape[:2]
    cx, cy = w / 2.0, h / 2.0
    prev_out: Optional[np.ndarray] = None
    output = []
    for frame in frames:
        f = frame.astype(np.float32)
        if prev_out is None:
            out = f
        else:
            if transform == "zoom":
                s = 1.0 + amount
                M = cv2.getRotationMatrix2D((cx, cy), 0, s)
            elif transform == "rotate":
                deg = amount * (180.0 / 3.14159265)
                M = cv2.getRotationMatrix2D((cx, cy), deg, 1.0)
            elif transform == "spiral":
                s = 1.0 + amount
                deg = amount * 0.5 * (180.0 / 3.14159265)
                M = cv2.getRotationMatrix2D((cx, cy), deg, s)
            else:  # "shift"
                M = np.float64([[1, 0, amount * w],
                                [0, 1, 0]])
            warped = cv2.warpAffine(
                prev_out, M, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )
            out = (1.0 - mix) * f + mix * warped
        prev_out = out.copy()
        output.append(np.clip(out, 0, 255).astype(np.uint8))
    return output


@task(name="feedback-transform", tags=["ram-heavy"])
def feedback_transform(
    src: Path,
    dst: Path,
    *,
    transform: str = "zoom",
    amount: float = 0.02,
    mix: float = 0.7,
    seed: Optional[int] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """
    Feedback with spatial transform — each output frame blends the
    current input with a spatially-transformed version of the previous
    output. Creates infinite regression trails, spiral echoes, fractal
    temporal stacking.

    Different from echo because feedback compounds geometrically —
    each frame contains ghosts of ALL previous frames, not just a
    windowed average.

    Single-pass streaming. One previous-output buffer.

    Output is the same duration as input.

    transform: spatial transform applied to the feedback buffer.
               "zoom" = slow zoom in/out (default).
               "rotate" = slow rotation.
               "spiral" = zoom + rotation combined.
               "shift" = horizontal drift.
    amount:    magnitude of the spatial transform per frame.
               For zoom: fraction of zoom per frame (0.02 = 2% zoom/frame).
               For rotate: radians per frame (0.02 ~ 1.1 deg/frame).
               For shift: fraction of width per frame.
               Default 0.02.
    mix:       blend ratio of feedback vs current frame, 0-1.
               0 = no feedback (passthrough).
               0.5 = equal blend.
               0.7 = strong feedback, faint new content.
               0.9 = extreme — trails dominate.
               Default 0.7.
    seed:      unused, kept for interface consistency.
    """
    c = cfg or Config()
    info = probe(src, c)

    print(f"feedback_transform: {info.duration:.1f}s @ {info.fps}fps, "
          f"{info.width}x{info.height}")
    print(f"  transform={transform}, amount={amount}, mix={mix}")

    with _load_frames(src, c) as frames:
        output = _process_feedback_transform(
            frames, info.fps, transform=transform, amount=amount, mix=mix, seed=seed,
        )
        return _write_frames(output, dst, info, c, "feedback_transform")


# ---------------------------------------------------------------------------
# Scan refresh — CRT phosphor beam effect
# ---------------------------------------------------------------------------

def _process_scan_refresh(
    frames,
    fps: float,
    *,
    speed: float = 0.5,
    decay: float = 3.0,
    beam_width: float = 0.02,
    axis: str = "horizontal",
    seed: Optional[int] = None,
) -> list[np.ndarray]:
    """CRT phosphor scan refresh — beam sweeps across frame, refreshing
    content as it passes. Behind the beam, phosphor decays exponentially.

    Brightness-preserving: the beam writes pixels *boosted* above their true
    value, and the exponential decay pulls them below it. Over one full scan
    cycle the time-average equals the original pixel value, so mean brightness
    is unchanged. Requires float32 phosphor buffer (values can exceed 255
    mid-cycle before decay brings them back).

    Streaming: only needs a single float32 phosphor canvas.

    speed:      beam sweep rate in cycles per second (full sweeps).
    decay:      phosphor decay rate. Higher = faster fade, tighter trail.
                ~1.0 = long slow glow, ~2.0 = medium trail, ~3.0 = tight band.
    beam_width: fraction of the scan dimension refreshed per pass (0-1).
                0.02 = thin line, 0.1 = wide band.
    axis:       "horizontal" = top-to-bottom scan, "vertical" = left-to-right.
    """
    n = len(frames)
    if n == 0:
        return []

    h, w = frames[0].shape[:2]
    scan_dim = h if axis == "horizontal" else w
    beam_px = max(1, int(beam_width * scan_dim))
    decay = max(0.1, decay)  # avoid divide-by-zero

    # ── Brightness-preserving boost factor ──
    # Fade function: exp(-decay * dist / scan_dim).  Over one full scan cycle
    # dist traverses 0 → scan_dim, so total decay = exp(-decay).
    # Time-average of V * boost * exp(-decay * x) for x ∈ [0,1):
    #   avg = V * boost * (1 - exp(-decay)) / decay
    # Setting avg = V:  boost = decay / (1 - exp(-decay))
    boost = decay / (1.0 - np.exp(-decay))

    # Phosphor buffer starts black
    phosphor = np.zeros((h, w, 3), dtype=np.float32)
    output = []
    prev_pos = 0.0

    for t in range(n):
        current = frames[t].astype(np.float32)

        # Beam position: cycles through scan dimension
        scan_pos = (t * speed * scan_dim / fps) % scan_dim

        # Refresh all lines/cols the beam has swept since last frame.
        # This closes gaps when the beam advances faster than beam_width.
        if t == 0:
            write_start = int(scan_pos)
            write_end = int(scan_pos) + beam_px
        else:
            write_start = int(prev_pos)
            write_end = int(scan_pos) + beam_px
            # Handle wrap-around: if beam crossed the boundary, extend end
            if scan_pos < prev_pos:
                write_end += scan_dim

        # Write boosted pixel values at beam position
        if axis == "horizontal":
            for idx in range(write_start, write_end):
                row = idx % scan_dim
                phosphor[row] = current[row] * boost
        else:
            for idx in range(write_start, write_end):
                col = idx % scan_dim
                phosphor[:, col] = current[:, col] * boost

        prev_pos = scan_pos

        # Build a per-row (or per-col) decay mask based on distance from beam
        coords = np.arange(scan_dim, dtype=np.float32)
        # Distance behind the beam (wrapping). 0 = at beam, scan_dim = furthest.
        dist = (scan_pos - coords) % scan_dim
        # Decay factor: 1.0 at beam, exponential fall-off behind.
        fade = np.exp(-decay * dist / scan_dim)

        if axis == "horizontal":
            mask = fade[:, np.newaxis, np.newaxis]
        else:
            mask = fade[np.newaxis, :, np.newaxis]

        result = phosphor * mask
        output.append(np.clip(result, 0, 255).astype(np.uint8))

    return output


@task(name="scan-refresh", tags=["ram-heavy"])
def scan_refresh(
    src: Path,
    dst: Path,
    *,
    speed: float = 0.5,
    decay: float = 3.0,
    beam_width: float = 0.02,
    axis: str = "horizontal",
    seed: Optional[int] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """CRT phosphor scan refresh effect.

    A beam sweeps across the frame. Where it passes, the current frame
    content is written to a phosphor buffer. Behind the beam, the phosphor
    decays toward black — creating a glowing trail of progressively older
    content fading away.

    Streaming: phosphor buffer + current frame only.
    Output is the same duration as input.
    """
    c = cfg or Config()
    info = probe(src, c)

    print(f"scan_refresh: {info.duration:.1f}s @ {info.fps}fps, "
          f"{info.width}x{info.height}")
    print(f"  speed={speed}, decay={decay}, beam_width={beam_width}, axis={axis}")

    with _load_frames(src, c) as frames:
        output = _process_scan_refresh(
            frames, info.fps, speed=speed, decay=decay,
            beam_width=beam_width, axis=axis, seed=seed,
        )
        return _write_frames(output, dst, info, c, "scan_refresh")


# ---------------------------------------------------------------------------
# Fused time effect chain
# ---------------------------------------------------------------------------

def _apply_time_effect(
    frames: list[np.ndarray],
    step,
    fps: float,
    seed: Optional[int],
) -> list[np.ndarray]:
    """Dispatch a recipe step dataclass to the corresponding _process_* helper."""
    # Local imports to avoid circular dependency (recipe.py doesn't import time.py)
    from ..recipe import (
        ScrubStep, DriftStep, PingPongStep, EchoStep, PatchStep,
        SlitScanStep, TemporalTileStep, QuadLoopStep,
        SmearStep, BloomStep, StackStep, SlipStep,
        FlowWarpStep, TemporalSortStep, ExtremaHoldStep, FeedbackTransformStep,
        ScanRefreshStep, TemporalFFTStep,
        TemporalGradientStep, TemporalMedianStep, AxisSwapStep,
        TemporalMorphStep, DepthSliceStep, TemporalEqualizeStep,
        TemporalDisplaceStep, SpectralRemixStep, PhaseScrambleStep,
    )
    match step:
        case ScrubStep(smoothness=smoothness, intensity=intensity):
            return _process_scrub(frames, fps, smoothness=smoothness, intensity=intensity, seed=seed)
        case DriftStep(loop_dur=loop_dur, drift=drift):
            return _process_drift(frames, fps, loop_dur=loop_dur, drift=drift, seed=seed)
        case PingPongStep(window=window):
            return _process_ping_pong(frames, fps, window=window, seed=seed)
        case EchoStep(delay=delay, trail=trail):
            return _process_echo(frames, fps, delay=delay, trail=trail)
        case PatchStep(patch_min=patch_min, patch_max=patch_max):
            return _process_patch(frames, fps, patch_min=patch_min, patch_max=patch_max, seed=seed)
        case SlitScanStep(axis=axis, scan_speed=scan_speed):
            return _process_slit_scan(frames, fps, axis=axis, scan_speed=scan_speed, seed=seed)
        case TemporalTileStep(grid=grid, offset_scale=offset_scale):
            return _process_temporal_tile(frames, fps, grid=grid, offset_scale=offset_scale, seed=seed)
        case QuadLoopStep(loop_dur=loop_dur, offset_scale=offset_scale, layout=layout):
            return _process_quad_loop(frames, fps, loop_dur=loop_dur, offset_scale=offset_scale, layout=layout, seed=seed)
        case SmearStep(threshold=threshold):
            return _process_smear(frames, fps, threshold=threshold)
        case BloomStep(sensitivity=sensitivity):
            return _process_bloom(frames, fps, sensitivity=sensitivity)
        case StackStep(window=window, mode=mode):
            return _process_frame_stack(frames, fps, window=window, mode=mode)
        case SlipStep(n_bands=n_bands, max_slip=max_slip, axis=axis):
            return _process_slip(frames, fps, n_bands=n_bands, max_slip=max_slip, axis=axis, seed=seed)
        case FlowWarpStep(amplify=amplify, smooth=smooth):
            return _process_flow_warp(frames, fps, amplify=amplify, smooth=smooth, seed=seed)
        case TemporalSortStep(mode=mode, direction=direction):
            return _process_temporal_sort(frames, fps, mode=mode, direction=direction, seed=seed)
        case ExtremaHoldStep(mode=mode, decay=decay):
            return _process_extrema_hold(frames, fps, mode=mode, decay=decay, seed=seed)
        case FeedbackTransformStep(transform=xform, amount=amount, mix=mix_val):
            return _process_feedback_transform(frames, fps, transform=xform, amount=amount, mix=mix_val, seed=seed)
        case ScanRefreshStep(speed=speed, decay=decay, beam_width=beam_width, axis=axis):
            return _process_scan_refresh(frames, fps, speed=speed, decay=decay, beam_width=beam_width, axis=axis, seed=seed)
        case TemporalFFTStep(filter_type=ft, cutoff_low=cl, cutoff_high=ch,
                             preserve_dc=pdc):
            return _process_temporal_fft(frames, fps, filter_type=ft,
                                         cutoff_low=cl, cutoff_high=ch,
                                         preserve_dc=pdc, seed=seed)
        case TemporalGradientStep(order=order):
            return _process_temporal_gradient(frames, fps, order=order, seed=seed)
        case TemporalMedianStep(window=window):
            return _process_temporal_median(frames, fps, window=window, seed=seed)
        case AxisSwapStep(axis=axis):
            return _process_axis_swap(frames, fps, axis=axis, seed=seed)
        case TemporalMorphStep(operation=operation, window=window):
            return _process_temporal_morph(frames, fps, operation=operation, window=window, seed=seed)
        case DepthSliceStep(angle=angle, axis=axis):
            return _process_depth_slice(frames, fps, angle=angle, axis=axis, seed=seed)
        case TemporalEqualizeStep(strength=strength):
            return _process_temporal_equalize(frames, fps, strength=strength, seed=seed)
        case TemporalDisplaceStep(amount=amount, channel=channel):
            return _process_temporal_displace(frames, fps, amount=amount, channel=channel, seed=seed)
        case SpectralRemixStep(mode=mode, amount=amount):
            return _process_spectral_remix(frames, fps, mode=mode, amount=amount, seed=seed)
        case PhaseScrambleStep(amount=amount):
            return _process_phase_scramble(frames, fps, amount=amount, seed=seed)
        case _:
            raise ValueError(f"Not a time effect step: {type(step).__name__}")


@task(name="fused-time-chain", tags=["ram-heavy"])
def fused_time_chain(
    src: Path,
    dst: Path,
    *,
    steps: list,
    cfg: Optional[Config] = None,
) -> Path:
    """Apply multiple time effects in a single decode-process-encode pass.

    Eliminates intermediate file I/O when consecutive recipe steps are
    all temporal effects. Mirrors _apply_filter_chain for ffmpeg filters.

    steps: list of (step_dataclass, seed) tuples.
    """
    c = cfg or Config()
    info = probe(src, c)
    fps = info.fps

    frames = _load_frames(src, c)
    n_effects = len(steps)
    logger.info("fused_time_chain: %d effects on %d frames (%.1fs @ %.0ffps)",
                n_effects, len(frames), info.duration, fps)
    print(f"fused_time_chain: {n_effects} effects on {len(frames)} frames "
          f"({info.duration:.1f}s @ {fps:.0f}fps)")

    try:
        t0 = _time.monotonic()
        for i, (step, seed) in enumerate(steps):
            step_name = type(step).__name__.replace("Step", "").lower()
            logger.info("  [%d/%d] %s ...", i + 1, n_effects, step_name)
            step_t0 = _time.monotonic()

            result = _apply_time_effect(frames, step, fps, seed)
            # Clean up the FrameBuffer after first effect; subsequent
            # iterations frames is a plain list, cleanup() is a no-op check.
            if isinstance(frames, FrameBuffer):
                frames.cleanup()
            frames = result  # now list[np.ndarray]

            step_elapsed = _time.monotonic() - step_t0
            print(f"  [{i + 1}/{n_effects}] {step_name} done ({step_elapsed:.1f}s, {len(frames)} frames)")

        total_elapsed = _time.monotonic() - t0
        print(f"  all effects in {total_elapsed:.1f}s")

        return _write_frames(frames, dst, info, c, "fused_time_chain")
    finally:
        if isinstance(frames, FrameBuffer):
            frames.cleanup()
