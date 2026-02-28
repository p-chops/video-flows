"""
Pipeline configuration — paths, defaults, project-level settings.

All paths resolve relative to a project root. Override via environment
variables or by passing a Config instance to flows.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import os


@dataclass
class Config:
    # ── Directories ──────────────────────────────────────────────────────
    project_root: Path = field(default_factory=lambda: Path(
        os.environ.get("VP_PROJECT_ROOT", ".")
    ).resolve())

    # Source footage lives here
    source_dir: Path = field(default=None)
    # ISF shader folder — the pipeline scans this for .fs files
    shader_dir: Path = field(default=None)
    # Working directory for intermediate files (segments, masks, etc.)
    work_dir: Path = field(default=None)
    # Final output directory
    output_dir: Path = field(default=None)

    # ── Video defaults ───────────────────────────────────────────────────
    default_codec: str = "libx264"
    default_crf: int = 18
    default_pix_fmt: str = "yuv420p"
    # Max video bitrate in kbps. When set, ffmpeg uses constrained CRF:
    # CRF picks quality, but output never exceeds this bitrate.
    # None = unconstrained (pure CRF).
    default_video_bitrate: Optional[int] = None

    # ── GPU encoding (VideoToolbox) ───────────────────────────────────
    # When True, clean encodes use h264_videotoolbox instead of libx264.
    # Does NOT affect bitrate_crush dirty passes (they need libx264/mpeg2/mpeg4).
    gpu_encode: bool = True
    # VideoToolbox quality scale: 1–100, higher = better quality / larger files.
    # ~60 ≈ CRF 18, ~35 ≈ CRF ~24 (visually acceptable for intermediates).
    vt_quality: int = 60

    # ── FFmpeg ───────────────────────────────────────────────────────────
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"
    ffmpeg_loglevel: str = "error"

    def __post_init__(self):
        if self.source_dir is None:
            self.source_dir = self.project_root / "source"
        if self.shader_dir is None:
            self.shader_dir = self.project_root / "shaders"
        if self.work_dir is None:
            self.work_dir = self.project_root / "work"
        if self.output_dir is None:
            self.output_dir = self.project_root / "output"

    def encode_args(self) -> list[str]:
        """Return codec + quality ffmpeg args for clean encodes.

        When gpu_encode is True, uses h264_videotoolbox with -q:v quality.
        Otherwise uses libx264 with -crf quality.
        Appends -pix_fmt and optional bitrate constraints.
        """
        if self.gpu_encode:
            args = ["-c:v", "h264_videotoolbox", "-q:v", str(self.vt_quality)]
        else:
            args = ["-c:v", self.default_codec, "-crf", str(self.default_crf)]
        args += ["-pix_fmt", self.default_pix_fmt]
        if not self.gpu_encode and self.default_video_bitrate is not None:
            br = self.default_video_bitrate
            args += ["-maxrate", f"{br}k", "-bufsize", f"{br * 2}k"]
        return args

    def ensure_dirs(self):
        """Create all directories if they don't exist."""
        for d in (self.source_dir, self.shader_dir,
                  self.work_dir, self.output_dir):
            d.mkdir(parents=True, exist_ok=True)
