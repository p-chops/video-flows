"""
FFmpeg / FFprobe utilities.

Wraps the raw-pipe patterns from make_channels.py and bake_glitch_stack.py
into reusable functions. All video I/O goes through here.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional

import numpy as np

from .config import Config


# ─── Video metadata ──────────────────────────────────────────────────────────

@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    duration: float  # seconds, 0 if unknown
    codec: str
    bitrate: int = 0  # kbps, 0 if unknown

    @property
    def frame_size(self) -> int:
        """Bytes per raw RGB24 frame."""
        return self.width * self.height * 3


def probe(path: Path, cfg: Optional[Config] = None) -> VideoInfo:
    """Return video stream metadata via ffprobe."""
    bin_ = cfg.ffprobe_bin if cfg else "ffprobe"
    cmd = [bin_, "-v", "quiet", "-print_format", "json",
           "-show_streams", "-show_format", str(path)]
    out = subprocess.check_output(cmd)
    data = json.loads(out)

    vs = next(s for s in data["streams"] if s["codec_type"] == "video")
    num, den = vs["r_frame_rate"].split("/")
    fps = float(num) / float(den)

    duration = float(vs.get("duration", 0))
    if duration == 0:
        duration = float(data.get("format", {}).get("duration", 0))

    # Bitrate: prefer stream-level, fall back to format-level
    bitrate = int(vs.get("bit_rate", 0)) // 1000
    if bitrate == 0:
        bitrate = int(data.get("format", {}).get("bit_rate", 0)) // 1000

    return VideoInfo(
        width=int(vs["width"]),
        height=int(vs["height"]),
        fps=fps,
        duration=duration,
        codec=vs.get("codec_name", "unknown"),
        bitrate=bitrate,
    )


# ─── Frame reader ────────────────────────────────────────────────────────────

def read_frames(path: Path, cfg: Optional[Config] = None
                ) -> Generator[np.ndarray, None, None]:
    """
    Yield raw RGB24 frames as (H, W, 3) uint8 numpy arrays.

    Opens an ffmpeg subprocess that decodes to rawvideo on stdout.
    Caller should consume the generator fully or use it in a with-block
    to ensure the subprocess is cleaned up.
    """
    info = probe(path, cfg)
    bin_ = cfg.ffmpeg_bin if cfg else "ffmpeg"
    loglevel = cfg.ffmpeg_loglevel if cfg else "error"

    proc = subprocess.Popen(
        [bin_, "-loglevel", loglevel,
         "-i", str(path),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE,
    )

    try:
        while True:
            raw = proc.stdout.read(info.frame_size)
            if len(raw) < info.frame_size:
                break
            yield np.frombuffer(raw, dtype=np.uint8).reshape(
                info.height, info.width, 3
            )
    finally:
        proc.stdout.close()
        proc.wait()


# ─── Frame writer ────────────────────────────────────────────────────────────

class FrameWriter:
    """
    Context-managed ffmpeg writer that accepts raw RGB24 frames.

    Usage:
        with FrameWriter(dst, info) as writer:
            writer.write(frame_array)
    """

    def __init__(self, path: Path, info_or_width, height=None, *,
                 fps=None, cfg: Optional[Config] = None):
        self.path = path
        if isinstance(info_or_width, VideoInfo):
            self.info = info_or_width
        else:
            self.info = VideoInfo(
                width=int(info_or_width), height=int(height),
                fps=float(fps or 30.0), duration=0, codec="h264",
            )
        self.cfg = cfg or Config()
        self._proc = None

    def __enter__(self):
        cmd = [
            self.cfg.ffmpeg_bin, "-loglevel", self.cfg.ffmpeg_loglevel, "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{self.info.width}x{self.info.height}",
            "-r", str(self.info.fps),
            "-i", "-",
            "-c:v", self.cfg.default_codec,
            "-pix_fmt", self.cfg.default_pix_fmt,
            "-crf", str(self.cfg.default_crf),
        ]
        if self.cfg.default_video_bitrate is not None:
            br = self.cfg.default_video_bitrate
            cmd += ["-maxrate", f"{br}k", "-bufsize", f"{br * 2}k"]
        cmd.append(str(self.path))
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        return self

    def write(self, frame: np.ndarray):
        """Write a single RGB24 frame (H, W, 3) uint8 array."""
        self._proc.stdin.write(frame.tobytes())

    def write_raw(self, data: bytes):
        """Write raw RGB24 bytes (W * H * 3)."""
        self._proc.stdin.write(data)

    def __exit__(self, *exc):
        if self._proc:
            self._proc.stdin.close()
            self._proc.wait()


# ─── Logged ffmpeg runner ─────────────────────────────────────────────────────

def _parse_progress_time(value: str) -> float:
    """Parse ffmpeg progress out_time like '00:01:23.456000' to seconds."""
    parts = value.strip().split(":")
    if len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    return 0.0


def run_ffmpeg_logged(
    cmd: list[str],
    duration: float,
    logger: logging.Logger,
    label: str = "ffmpeg",
    log_interval: float = 5.0,
) -> None:
    """
    Run an ffmpeg command with periodic progress logging.

    Injects `-progress pipe:1` into the command, streams ffmpeg's
    structured progress output, and logs position/speed every
    `log_interval` seconds.

    duration:     expected total duration (seconds), for percentage calc.
    logger:       Prefect run logger (or any stdlib logger).
    label:        prefix for log messages.
    log_interval: minimum seconds between progress log lines.
    """
    # Inject -progress pipe:1 before the output path (last arg)
    full_cmd = list(cmd[:-1]) + ["-progress", "pipe:1", cmd[-1]]

    t0 = time.monotonic()
    last_log = t0

    proc = subprocess.Popen(
        full_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )

    cur_time = 0.0
    cur_speed = ""
    cur_frame = 0

    for line in proc.stdout:
        line = line.strip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")

        if key == "out_time":
            cur_time = _parse_progress_time(val)
        elif key == "speed":
            cur_speed = val
        elif key == "frame":
            try:
                cur_frame = int(val)
            except ValueError:
                pass
        elif key == "progress":
            # "progress=continue" or "progress=end" — marks end of a block
            now = time.monotonic()
            if now - last_log >= log_interval or val == "end":
                elapsed = now - t0
                pct = (cur_time / duration * 100) if duration > 0 else 0
                logger.info(
                    "%s: %.0f%% (%.1f/%.1fs) frame=%d speed=%s elapsed=%.1fs",
                    label, min(pct, 100), cur_time, duration,
                    cur_frame, cur_speed, elapsed,
                )
                last_log = now

    proc.wait()
    if proc.returncode != 0:
        stderr = proc.stderr.read()
        logger.error("%s: ffmpeg failed (exit %d): %s",
                     label, proc.returncode, stderr.strip())
        raise subprocess.CalledProcessError(
            proc.returncode, full_cmd, stderr=stderr,
        )


# ─── Segment extraction ──────────────────────────────────────────────────────

def extract_segment(src: Path, dst: Path,
                    start: float, duration: float,
                    fps: Optional[float] = None,
                    cfg: Optional[Config] = None):
    """
    Re-encode a time segment from src to dst.
    If fps is given, locks the output frame rate.
    """
    c = cfg or Config()
    cmd = [
        c.ffmpeg_bin, "-y", "-loglevel", c.ffmpeg_loglevel,
        "-ss", f"{start:.3f}", "-i", str(src),
        "-t", f"{duration:.3f}",
        "-an",
        "-c:v", c.default_codec, "-crf", str(c.default_crf),
        "-pix_fmt", c.default_pix_fmt,
    ]
    if c.default_video_bitrate is not None:
        br = c.default_video_bitrate
        cmd += ["-maxrate", f"{br}k", "-bufsize", f"{br * 2}k"]
    if fps:
        cmd += ["-r", str(fps)]
    cmd.append(str(dst))
    subprocess.run(cmd, check=True)


def copy_segment(src: Path, dst: Path,
                  start: float, duration: float,
                  cfg: Optional[Config] = None):
    """
    Copy a time segment from src to dst without re-encoding.

    Near-instant since it copies compressed packets directly.
    Cuts land on the nearest keyframe, so boundaries may drift
    by up to a GOP length (~0.5-2s) — fine for creative workflows.
    """
    c = cfg or Config()
    subprocess.run([
        c.ffmpeg_bin, "-y", "-loglevel", c.ffmpeg_loglevel,
        "-ss", f"{start:.3f}", "-i", str(src),
        "-t", f"{duration:.3f}",
        "-an", "-c:v", "copy",
        str(dst),
    ], check=True)


# ─── Concatenation ───────────────────────────────────────────────────────────

def concat_files(clips: list[Path], dst: Path,
                 cfg: Optional[Config] = None):
    """
    Concatenate clips via the ffmpeg concat demuxer.
    All clips must share codec and pixel format.
    """
    c = cfg or Config()
    list_file = dst.with_suffix(".concat.txt")
    with open(list_file, "w") as f:
        for p in clips:
            f.write(f"file '{p.resolve()}'\n")

    subprocess.run([
        c.ffmpeg_bin, "-y", "-loglevel", c.ffmpeg_loglevel,
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(dst),
    ], check=True)

    list_file.unlink()
