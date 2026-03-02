"""
Color correction tasks — normalization, levels, etc.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Optional

import cv2
import numpy as np
from prefect import task

from ..config import Config
from ..ffmpeg import probe, read_frames


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
    Normalize video levels — clips the darkest and brightest percentiles
    and stretches the remaining range to fill 0–255.

    Uses ffmpeg's colorlevels filter: a static per-pixel LUT operation
    with negligible overhead beyond decode + encode.

    black_point: fraction of darkest values to clip to black (default 0.01).
    white_point: fraction of brightest values to clip to white (default 0.01).
                 Note: this is the fraction clipped from the TOP, not a
                 percentile — so 0.01 means clip the top 1%.
    """
    c = cfg or Config()

    # colorlevels rimin/rimax are input intensity levels [0.0–1.0].
    # rimin = black point (values below become black)
    # rimax = white point (values above become white)
    # Everything between is stretched to fill 0–255.
    bp = black_point
    wp = white_point
    vf = (
        f"colorlevels="
        f"rimin={bp}:gimin={bp}:bimin={bp}:"
        f"rimax={wp}:gimax={wp}:bimax={wp}"
    )

    subprocess.run([
        c.ffmpeg_bin, "-y", "-loglevel", c.ffmpeg_loglevel,
        "-i", str(src),
        "-vf", vf,
        "-an",
        *c.encode_args(),
        str(dst),
    ], check=True)

    return dst


def _probe_brightness(src: Path, cfg: Config, n_samples: int = 10) -> float:
    """Sample frames and return average brightness in [0, 1]."""
    info = probe(src, cfg)
    total_frames = int(info.duration * info.fps)
    sample_interval = max(1, total_frames // n_samples)

    total = 0.0
    count = 0
    for i, frame in enumerate(read_frames(src, cfg=cfg)):
        if i % sample_interval == 0:
            total += float(np.mean(frame)) / 255.0
            count += 1
    return total / max(count, 1)


def _probe_motion(src: Path, cfg: Config, n_samples: int = 10) -> float:
    """Sample frame pairs and return average motion in [0, 1].

    0.0 = completely static, 1.0 = maximum change between frames.
    Uses mean absolute difference of consecutive sampled frames.
    """
    info = probe(src, cfg)
    total_frames = int(info.duration * info.fps)
    sample_interval = max(1, total_frames // n_samples)

    prev = None
    total = 0.0
    count = 0
    for i, frame in enumerate(read_frames(src, cfg=cfg)):
        if i % sample_interval == 0:
            if prev is not None:
                diff = np.mean(np.abs(
                    frame.astype(np.int16) - prev.astype(np.int16)
                )) / 255.0
                total += diff
                count += 1
            prev = frame
    return total / max(count, 1)


# ---------------------------------------------------------------------------
# Multi-feature quality classifier
# ---------------------------------------------------------------------------

@dataclass
class QualityReport:
    """Multi-feature quality assessment of a rendered clip.

    Acts as a feature vector for quality gating.  Rule-based thresholds
    now; drop-in ML classifier later via ``to_array()`` / ``to_dict()``.
    """

    brightness: float        # mean luma [0, 1]
    contrast: float          # p95 − p5 luma [0, 1]
    motion: float            # frame-pair MAD [0, 1]
    temporal_variance: float # std of per-frame brightness
    spatial_entropy: float   # mean Laplacian variance, normalised [0, 1]

    FEATURE_NAMES: ClassVar[list[str]] = [
        "brightness", "contrast", "motion",
        "temporal_variance", "spatial_entropy",
    ]

    def to_array(self) -> np.ndarray:
        """Feature vector (float64) for ML classifiers."""
        return np.array([getattr(self, f) for f in self.FEATURE_NAMES],
                        dtype=np.float64)

    def to_dict(self) -> dict[str, float]:
        """Serialisable dict for logging / training data."""
        return {f: getattr(self, f) for f in self.FEATURE_NAMES}

    def summary(self) -> str:
        return (
            f"brightness={self.brightness:.3f} contrast={self.contrast:.3f} "
            f"motion={self.motion:.4f} temporal_var={self.temporal_variance:.3f} "
            f"entropy={self.spatial_entropy:.3f}"
        )


@dataclass
class QualityThresholds:
    """Thresholds for quality-gate decisions.

    Each ``*_floor`` is the minimum acceptable value for that metric.
    Set to 0.0 to disable.
    """

    brightness_floor: float = 0.08       # below → reroll (too dark to salvage)
    contrast_floor: float = 0.05         # below → reroll (completely flat)
    motion_floor: float = 0.005          # below → reroll (stasis)
    temporal_variance_floor: float = 0.002  # below → reroll (monotonous)
    spatial_entropy_floor: float = 0.005 # below → reroll (blank / solid)
    brightness_fixable: float = 0.15     # dim but salvageable → auto_levels


# Empirical normalisation ceiling for Laplacian variance.
# Typical 720p video: ~100–2000.  Clips above this read as 1.0.
_ENTROPY_CEILING = 2000.0


def probe_quality(
    src: Path,
    cfg: Config,
    n_samples: int = 10,
) -> QualityReport:
    """Collect quality metrics from a single decode pass.

    Samples *n_samples* evenly-spaced frames and computes all features
    inside one ``read_frames()`` loop — one ffmpeg subprocess, one decode.
    """
    info = probe(src, cfg)
    total_frames = int(info.duration * info.fps)
    sample_interval = max(1, total_frames // n_samples)

    brightness_samples: list[float] = []
    luma_flat: list[np.ndarray] = []
    entropy_samples: list[float] = []
    motion_diffs: list[float] = []
    prev_frame: np.ndarray | None = None

    for i, frame in enumerate(read_frames(src, cfg=cfg)):
        if i % sample_interval != 0:
            continue

        # Brightness (BT.601 luma)
        luma = (
            0.299 * frame[:, :, 0].astype(np.float32)
            + 0.587 * frame[:, :, 1].astype(np.float32)
            + 0.114 * frame[:, :, 2].astype(np.float32)
        )
        brightness_samples.append(float(np.mean(luma)) / 255.0)
        luma_flat.append(luma.ravel())

        # Spatial entropy — Laplacian variance (edges / detail density)
        gray = np.clip(luma, 0, 255).astype(np.uint8)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        entropy_samples.append(float(np.var(lap)))

        # Motion — frame-pair MAD
        if prev_frame is not None:
            diff = float(np.mean(np.abs(
                frame.astype(np.int16) - prev_frame.astype(np.int16)
            ))) / 255.0
            motion_diffs.append(diff)
        prev_frame = frame

    # Aggregate
    brightness = float(np.mean(brightness_samples)) if brightness_samples else 0.0

    if luma_flat:
        all_luma = np.concatenate(luma_flat)
        p5 = float(np.percentile(all_luma, 5)) / 255.0
        p95 = float(np.percentile(all_luma, 95)) / 255.0
        contrast = p95 - p5
    else:
        contrast = 0.0

    motion = float(np.mean(motion_diffs)) if motion_diffs else 0.0

    temporal_variance = (
        float(np.std(brightness_samples))
        if len(brightness_samples) > 1
        else 0.0
    )

    raw_entropy = float(np.mean(entropy_samples)) if entropy_samples else 0.0
    spatial_entropy = min(raw_entropy / _ENTROPY_CEILING, 1.0)

    return QualityReport(
        brightness=brightness,
        contrast=contrast,
        motion=motion,
        temporal_variance=temporal_variance,
        spatial_entropy=spatial_entropy,
    )


def should_reroll(
    report: QualityReport,
    thresholds: QualityThresholds | None = None,
) -> tuple[bool, str, bool]:
    """Rule-based quality gate.

    Returns ``(reroll, reason, fixable)``:
    - *reroll*:  True → discard & regenerate.
    - *reason*:  human-readable explanation.
    - *fixable*: True → apply auto_levels instead of rerolling.
    """
    t = thresholds or QualityThresholds()

    # Hard failures — reroll (most egregious first)
    if t.motion_floor > 0 and report.motion < t.motion_floor:
        return True, f"stasis (motion={report.motion:.4f})", False

    if t.brightness_floor > 0 and report.brightness < t.brightness_floor:
        return True, f"too dark to salvage (brightness={report.brightness:.3f})", False

    if t.contrast_floor > 0 and report.contrast < t.contrast_floor:
        return True, f"no contrast (contrast={report.contrast:.3f})", False

    if t.spatial_entropy_floor > 0 and report.spatial_entropy < t.spatial_entropy_floor:
        return True, f"blank/solid (entropy={report.spatial_entropy:.3f})", False

    if t.temporal_variance_floor > 0 and report.temporal_variance < t.temporal_variance_floor:
        return True, f"monotonous (temporal_var={report.temporal_variance:.3f})", False

    # Fixable — dim but not catastrophic
    if report.brightness < t.brightness_fixable:
        return False, f"dim (brightness={report.brightness:.3f})", True

    return False, "ok", False


@task(name="auto-levels")
def auto_levels(
    src: Path,
    dst: Path,
    *,
    target: float = 0.45,
    max_gamma: float = 3.0,
    threshold: float = 0.05,
    cfg: Optional[Config] = None,
) -> Path:
    """
    Probe average brightness and apply gamma correction toward a target.

    Measures the clip's average luma, computes a gamma that would push it
    toward `target` (0–1 scale), and applies via ffmpeg's eq filter.
    Skips the encode if no correction is needed (avg already near target).

    target:    desired average brightness in [0, 1] (default 0.45)
    max_gamma: clamp gamma to [1/max_gamma, max_gamma] (default 3.0)
    threshold: skip correction if |gamma - 1| < threshold (default 0.05)
    """
    c = cfg or Config()

    avg = _probe_brightness(src, c)

    # Compute gamma: eq filter applies out = in^(1/gamma)
    # We want avg^(1/gamma) = target, so gamma = log(avg) / log(target)
    if avg < 0.005:
        gamma = max_gamma
    elif avg > 0.995:
        gamma = 1.0 / max_gamma
    else:
        gamma = np.log(avg) / np.log(target)
        gamma = float(np.clip(gamma, 1.0 / max_gamma, max_gamma))

    if abs(gamma - 1.0) < threshold:
        shutil.copy2(src, dst)
        return dst

    subprocess.run([
        c.ffmpeg_bin, "-y", "-loglevel", c.ffmpeg_loglevel,
        "-i", str(src),
        "-vf", f"eq=gamma={gamma:.3f}",
        "-an",
        *c.encode_args(),
        str(dst),
    ], check=True)

    return dst
