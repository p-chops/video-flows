"""
Prefect tasks for cutting / extracting segments from source footage.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

from prefect import task
from ..cache import FILE_VALIDATED_INPUTS

from ..config import Config
from ..ffmpeg import probe, extract_segment, copy_segment


@task(name="detect-cuts", cache_policy=FILE_VALIDATED_INPUTS, persist_result=True)
def detect_cuts(src: Path, threshold: float = 27.0,
                adaptive_threshold: float = 5.0,
                method: str = "adaptive",
                luma_only: bool = True,
                min_scene_len: int = 120,
                cfg: Optional[Config] = None) -> list[float]:
    """
    Detect scene cuts in a video using PySceneDetect.
    Returns a list of timestamps (seconds) where cuts were detected.

    method:
        "adaptive"  — AdaptiveDetector (default). Compares each frame against
                       a rolling average, adapts to local contrast.
                       Uses `adaptive_threshold` (higher = less sensitive).
        "content"   — ContentDetector. Absolute frame-difference score.
                       Uses `threshold` (0-100, lower = more sensitive).

    threshold:           ContentDetector score threshold (default 27.0).
    adaptive_threshold:  AdaptiveDetector sensitivity (default 5.0).
    luma_only:           Ignore hue/saturation, use luminance only (default True).
                         Essential for B&W footage; works well for colour too.
    min_scene_len:       Minimum frames between cuts (default 120, ~5s at 24fps).
                         Prevents false-positive clusters from dissolves / flicker.

    For B&W footage, over-segmenting is preferable to missing cuts — extra
    splits are benign in most pipelines (e.g. deep-color applies a different
    shader per segment, so more segments = more variety).
    """
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector, AdaptiveDetector

    video = open_video(str(src))
    scene_manager = SceneManager()

    if method == "adaptive":
        scene_manager.add_detector(
            AdaptiveDetector(adaptive_threshold=adaptive_threshold,
                             luma_only=luma_only,
                             min_scene_len=min_scene_len))
    elif method == "content":
        scene_manager.add_detector(
            ContentDetector(threshold=threshold, luma_only=luma_only,
                            min_scene_len=min_scene_len))
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'adaptive' or 'content'.")

    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()

    # Each scene is (start_timecode, end_timecode). Cuts are at
    # the start of every scene after the first.
    return [scene[0].get_seconds() for scene in scene_list[1:]]


def sweep_thresholds(src: Path, cfg: Optional[Config] = None) -> None:
    """
    Test a range of detector configurations and print a comparison table.
    Not a Prefect task — meant for interactive threshold tuning.
    """
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector, AdaptiveDetector

    configs = [
        # (label, detector_factory)
        # ── Content + luma_only: sweep threshold and min_scene_len ──
        ("ct=35 luma msl=15",
         lambda: ContentDetector(threshold=35.0, luma_only=True,
                                 min_scene_len=15)),
        ("ct=40 luma msl=15",
         lambda: ContentDetector(threshold=40.0, luma_only=True,
                                 min_scene_len=15)),
        ("ct=50 luma msl=15",
         lambda: ContentDetector(threshold=50.0, luma_only=True,
                                 min_scene_len=15)),
        ("ct=60 luma msl=15",
         lambda: ContentDetector(threshold=60.0, luma_only=True,
                                 min_scene_len=15)),
        ("ct=70 luma msl=15",
         lambda: ContentDetector(threshold=70.0, luma_only=True,
                                 min_scene_len=15)),
        # Same thresholds but min 5s between cuts (~120 frames @ 24fps)
        ("ct=35 luma msl=120",
         lambda: ContentDetector(threshold=35.0, luma_only=True,
                                 min_scene_len=120)),
        ("ct=40 luma msl=120",
         lambda: ContentDetector(threshold=40.0, luma_only=True,
                                 min_scene_len=120)),
        ("ct=50 luma msl=120",
         lambda: ContentDetector(threshold=50.0, luma_only=True,
                                 min_scene_len=120)),
        ("ct=60 luma msl=120",
         lambda: ContentDetector(threshold=60.0, luma_only=True,
                                 min_scene_len=120)),
        # ── Adaptive + luma_only with longer min_scene_len ──
        ("ad=2.0 luma msl=120",
         lambda: AdaptiveDetector(adaptive_threshold=2.0, luma_only=True,
                                  min_scene_len=120)),
        ("ad=3.0 luma msl=120",
         lambda: AdaptiveDetector(adaptive_threshold=3.0, luma_only=True,
                                  min_scene_len=120)),
        ("ad=4.0 luma msl=120",
         lambda: AdaptiveDetector(adaptive_threshold=4.0, luma_only=True,
                                  min_scene_len=120)),
    ]

    c = cfg or Config()
    info = probe(src, c)
    print(f"Source: {src.name}  duration={info.duration:.1f}s  "
          f"{info.width}x{info.height} @ {info.fps:.2f}fps")
    print(f"\n{'Config':<25s}  {'Cuts':>5s}  {'Segments':>8s}  "
          f"{'Shortest':>8s}  {'Longest':>8s}  Cut timestamps")
    print("-" * 100)

    for label, detector_factory in configs:
        video = open_video(str(src))
        sm = SceneManager()
        sm.add_detector(detector_factory())
        sm.detect_scenes(video)
        scene_list = sm.get_scene_list()

        cuts = [s[0].get_seconds() for s in scene_list[1:]]
        # Build segment durations
        boundaries = [0.0] + sorted(cuts) + [info.duration]
        durations = [boundaries[i + 1] - boundaries[i]
                     for i in range(len(boundaries) - 1)]

        shortest = min(durations) if durations else 0
        longest = max(durations) if durations else 0
        cut_str = ", ".join(f"{t:.1f}" for t in cuts[:20])
        if len(cuts) > 20:
            cut_str += f" ... (+{len(cuts) - 20})"

        print(f"{label:<25s}  {len(cuts):5d}  {len(durations):8d}  "
              f"{shortest:7.1f}s  {longest:7.1f}s  {cut_str}")

    print()


@task(name="extract-segment")
def extract_segment_task(src: Path, dst: Path,
                         start: float, duration: float,
                         cfg: Optional[Config] = None) -> Path:
    """
    Extract a time segment from src → dst.
    Returns the output path.
    """
    info = probe(src, cfg)
    extract_segment(src, dst, start, duration, fps=info.fps, cfg=cfg)
    return dst


@task(name="random-segments")
def random_segments(src: Path, count: int,
                    min_dur: float = 5.0, max_dur: float = 20.0,
                    output_dir: Optional[Path] = None,
                    cfg: Optional[Config] = None) -> list[Path]:
    """
    Extract `count` random segments from a source video.

    Each segment has a random start point and a random duration
    in [min_dur, max_dur] with fractional offsets (so segments
    from different runs drift out of sync).

    Returns a list of output file paths.
    """
    c = cfg or Config()
    info = probe(src, c)
    out_dir = output_dir or c.work_dir / "segments"
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for i in range(count):
        dur = min_dur + random.random() * (max_dur - min_dur)
        # Add fractional ms so segments from different batches drift
        dur += random.randint(0, 999) / 1000.0

        max_start = max(0, info.duration - dur - 2.0)
        start = random.random() * max_start

        dst = out_dir / f"seg_{i:04d}.mp4"
        extract_segment(src, dst, start, dur, fps=info.fps, cfg=c)
        paths.append(dst)

    return paths


@task(name="segment-at-cuts")
def segment_at_cuts(src: Path, cuts: list[float],
                    min_segment: float = 2.0,
                    output_dir: Optional[Path] = None,
                    cfg: Optional[Config] = None) -> list[Path]:
    """
    Split a video into segments at detected cut points.
    Segments shorter than min_segment are merged with the next.
    Returns a list of output file paths.
    """
    c = cfg or Config()
    info = probe(src, c)
    out_dir = output_dir or c.work_dir / "segments"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build segment boundaries
    boundaries = [0.0] + sorted(cuts) + [info.duration]
    segments = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        dur = end - start
        if dur >= min_segment:
            segments.append((start, dur))

    paths = []
    for i, (start, dur) in enumerate(segments):
        dst = out_dir / f"cut_{i:04d}.mp4"
        copy_segment(src, dst, start, dur, cfg=c)
        paths.append(dst)

    return paths
