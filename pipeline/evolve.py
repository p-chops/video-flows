"""
Shader stack evolution via diversity-weighted random search.

Evaluates candidate shader stacks by rendering frames in-memory via GL
and scoring visual features. No file I/O during fitness evaluation.

Usage (standalone):
    from pipeline.evolve import evolve, EvolveConfig
    from scripts.generate_stacks import validate_shaders
    processors, generators, _ = validate_shaders(Path("packs/starter/shaders"))
    selected = evolve(processors, generators, EvolveConfig(seed=42))
"""

from __future__ import annotations

import hashlib
import random as _random_mod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ClassVar, Optional

import cv2
import numpy as np

from pipeline.gl import GLContext, FBOSlot
from pipeline.isf import ISFShader, parse_isf


# ─── Data structures ─────────────────────────────────────────────────────────

@dataclass
class Gene:
    """One shader in a stack, with concrete params and output spec."""
    shader_stem: str
    shader_path: Path
    params: dict[str, Any]          # concrete values for GPU eval
    param_spec: dict[str, Any]      # YAML-style spec for output


@dataclass
class Genome:
    """A candidate shader stack."""
    genes: list[Gene]
    fitness: float = 0.0
    features: dict[str, float] = field(default_factory=dict)
    uid: str = ""


@dataclass
class VisualFeatures:
    """Feature set for fitness evaluation — balanced spatial + temporal."""
    brightness: float           # mean luma [0, 1]
    contrast: float             # p95 − p5 luma [0, 1]
    spatial_entropy: float      # Laplacian variance, normalized
    color_coherence: float      # hue palette peakedness [0, 1]
    mid_frequency_ratio: float  # mid-band FFT energy (not noise, not flat)
    spatial_autocorrelation: float  # structural vs noise (smooth decay = structure)
    spectral_flatness: float    # geo_mean/arith_mean of spectrum (1=organic, 0=periodic)
    frame_variance: float       # variance across evaluated frames
    temporal_smoothness: float  # motion consistency (1 − CV of diffs)
    motion_magnitude: float     # mean frame-to-frame pixel change [0, 1]

    FEATURE_NAMES: ClassVar[list[str]] = [
        "brightness", "contrast", "spatial_entropy", "color_coherence",
        "mid_frequency_ratio", "spatial_autocorrelation", "spectral_flatness",
        "frame_variance", "temporal_smoothness", "motion_magnitude",
    ]

    def to_array(self) -> np.ndarray:
        return np.array([getattr(self, n) for n in self.FEATURE_NAMES],
                        dtype=np.float64)

    def to_dict(self) -> dict[str, float]:
        return {n: round(getattr(self, n), 6) for n in self.FEATURE_NAMES}


@dataclass
class EvolveConfig:
    n_candidates: int = 2000
    n_output: int = 15
    min_stack_size: int = 1
    max_stack_size: int = 5
    eval_frames: int = 5
    eval_width: int = 640
    eval_height: int = 360
    seed: int = 42
    diversity_weight: float = 1.0  # λ: higher = more spread
    min_fitness: float = 0.5       # floor: ignore low-quality candidates
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))


# ─── Fitness weights ──────────────────────────────────────────────────────────

DEFAULT_WEIGHTS: dict[str, float] = {
    "brightness": 0.0,          # penalty only
    "contrast": 1.5,
    "spatial_entropy": 2.0,
    "color_coherence": 1.0,
    "mid_frequency_ratio": 1.2,
    "spatial_autocorrelation": 1.2,
    "spectral_flatness": 0.0,   # multiplicative penalty, not additive
    "frame_variance": 1.5,
    "temporal_smoothness": 1.0,
    "motion_magnitude": 1.5,    # motion floor penalty handles the hard cutoff
}

_BRIGHTNESS_PENALTY_DARK = 0.08
_BRIGHTNESS_PENALTY_BRIGHT = 0.92


# ─── In-memory GL rendering ──────────────────────────────────────────────────

def generate_source_frames(
    gl: GLContext,
    generator: ISFShader,
    w: int, h: int,
    n_frames: int = 5,
    fps: float = 30.0,
    seed: int = 0,
    time_spread: float = 4.0,
) -> list[np.ndarray]:
    """Render N frames from a generator shader. All in-memory, no file I/O.

    Frames are spread across `time_spread` seconds to catch temporal
    divergence (e.g. brightness blowout from feedback accumulation).
    """
    try:
        prog = gl.compile(generator.glsl_source)
    except Exception:
        # Fallback to noise frames
        return _noise_frames(seed, w, h, n_frames)

    vao = gl.vao(prog)
    params = generator.default_params()
    fbo_a, fbo_b = gl.fbo_pair(w, h)

    # Spread eval frames across time_spread seconds
    frame_indices = [round(i * time_spread * fps / max(1, n_frames - 1))
                     for i in range(n_frames)] if n_frames > 1 else [0]

    frames = []
    for i in frame_indices:
        elapsed = i / fps
        fbo_a.fbo.use()

        if "u_time" in prog:
            prog["u_time"] = elapsed
        if "u_rendersize" in prog:
            prog["u_rendersize"] = (float(w), float(h))
        if "u_frameindex" in prog:
            prog["u_frameindex"] = i
        for pname, pval in params.items():
            if pname in prog:
                prog[pname] = pval

        vao.render()

        raw = fbo_a.fbo.read(components=3)
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
        frames.append(np.flipud(frame).copy())

    fbo_a.release()
    fbo_b.release()
    prog.release()

    # Degenerate check — if all frames are near-black, fall back to noise
    if all(f.mean() < 1.0 for f in frames):
        return _noise_frames(seed, w, h, n_frames, time_spread, fps)

    return frames, frame_indices


def _noise_frames(
    seed: int, w: int, h: int, n: int,
    time_spread: float = 4.0, fps: float = 30.0,
) -> tuple[list[np.ndarray], list[int]]:
    """Generate random noise frames as fallback source."""
    rng = np.random.RandomState(seed)
    frames = [rng.randint(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(n)]
    indices = [round(i * time_spread * fps / max(1, n - 1))
               for i in range(n)] if n > 1 else [0]
    return frames, indices


def apply_stack_to_frames(
    gl: GLContext,
    frames: list[np.ndarray],
    shaders: list[ISFShader],
    param_overrides: dict[str, dict[str, Any]],
    w: int, h: int,
    fps: float = 30.0,
    frame_indices: list[int] | None = None,
) -> list[np.ndarray]:
    """Apply a shader stack to in-memory frames. No file I/O.

    Mirrors _apply_shader_stack logic from shader.py but operates on
    numpy arrays instead of ffmpeg streams. If frame_indices is provided,
    uses those for elapsed time calculation (for time-spread evaluation).
    """
    overrides = param_overrides or {}

    # Compile each shader
    compiled = []
    for shader in shaders:
        prog = gl.compile(shader.glsl_source)
        shader_vao = gl.vao(prog)
        params = shader.default_params()
        params.update(overrides.get(shader.path.stem, {}))

        # Persistent buffers for multi-pass shaders
        buffers: dict[str, _PersistentBuffer] = {}
        temp_targets: dict[str, FBOSlot] = {}
        for p in shader.passes:
            if p.target and p.persistent:
                slots = gl.fbo_pair(w, h, float_tex=p.float_tex)
                for slot in slots:
                    slot.fbo.clear()
                buffers[p.target] = _PersistentBuffer(slots)
            elif p.target and not p.persistent:
                pair = gl.fbo_pair(w, h, float_tex=p.float_tex)
                pair[1].release()
                pair[0].fbo.clear()
                temp_targets[p.target] = pair[0]

        compiled.append((shader, prog, shader_vao, params, buffers, temp_targets))

    fbo_a, fbo_b = gl.fbo_pair(w, h)
    input_tex = gl.texture(w, h)

    indices = frame_indices if frame_indices is not None else list(range(len(frames)))
    output_frames = []
    for frame_idx, frame in zip(indices, frames):
        elapsed = frame_idx / fps

        input_tex.write(np.flipud(frame).tobytes())

        current_tex = input_tex
        write_fbo = fbo_a

        for shader, prog, shader_vao, params, buffers, temp_targets in compiled:
            if "u_time" in prog:
                prog["u_time"] = elapsed
            if "u_rendersize" in prog:
                prog["u_rendersize"] = (float(w), float(h))
            if "u_frameindex" in prog:
                prog["u_frameindex"] = frame_idx
            for pname, pval in params.items():
                if pname in prog:
                    prog[pname] = pval

            if shader.is_multipass and (buffers or temp_targets):
                n_passes = len(shader.passes)
                wrote_chain = False
                for pass_idx, pass_info in enumerate(shader.passes):
                    if pass_info.target and pass_info.target in buffers:
                        buffers[pass_info.target].write_slot.fbo.use()
                    elif pass_info.target and pass_info.target in temp_targets:
                        temp_targets[pass_info.target].fbo.use()
                    else:
                        write_fbo.fbo.use()
                        wrote_chain = True

                    if "u_passindex" in prog:
                        prog["u_passindex"] = pass_idx

                    current_tex.use(location=0)
                    if "inputImage" in prog:
                        prog["inputImage"] = 0

                    next_loc = 1
                    for buf_name, pbuf in buffers.items():
                        pbuf.read_texture.use(location=next_loc)
                        if buf_name in prog:
                            prog[buf_name] = next_loc
                        next_loc += 1
                    for buf_name, slot in temp_targets.items():
                        slot.texture.use(location=next_loc)
                        if buf_name in prog:
                            prog[buf_name] = next_loc
                        next_loc += 1

                    shader_vao.render()

                    if pass_info.target and pass_info.target in buffers:
                        buffers[pass_info.target].swap()

                if not wrote_chain:
                    last_pass = shader.passes[-1]
                    if last_pass.target and last_pass.target in buffers:
                        src_fbo = buffers[last_pass.target].slots[
                            buffers[last_pass.target].read_idx].fbo
                        gl.ctx.copy_framebuffer(write_fbo.fbo, src_fbo)
                    elif last_pass.target and last_pass.target in temp_targets:
                        src_fbo = temp_targets[last_pass.target].fbo
                        gl.ctx.copy_framebuffer(write_fbo.fbo, src_fbo)
            else:
                write_fbo.fbo.use()
                current_tex.use(location=0)
                if "inputImage" in prog:
                    prog["inputImage"] = 0
                shader_vao.render()

            current_tex = write_fbo.texture
            write_fbo = fbo_b if write_fbo is fbo_a else fbo_a

        result_fbo = fbo_b if write_fbo is fbo_a else fbo_a
        raw = result_fbo.fbo.read(components=3)
        out_frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
        output_frames.append(np.flipud(out_frame).copy())

    # Cleanup
    for _, prog, _, _, buffers, temp_targets in compiled:
        for pbuf in buffers.values():
            for slot in pbuf.slots:
                slot.release()
        for slot in temp_targets.values():
            slot.release()
        prog.release()
    fbo_a.release()
    fbo_b.release()
    input_tex.release()

    return output_frames


@dataclass
class _PersistentBuffer:
    """Ping-pong FBO pair for persistent buffer targets."""
    slots: list[FBOSlot]
    read_idx: int = 0

    @property
    def write_slot(self) -> FBOSlot:
        return self.slots[1 - self.read_idx]

    @property
    def read_texture(self):
        return self.slots[self.read_idx].texture

    def swap(self):
        self.read_idx = 1 - self.read_idx


# ─── Feature extraction ──────────────────────────────────────────────────────

def extract_features(frames: list[np.ndarray]) -> VisualFeatures:
    """Compute visual features from a list of numpy frames."""
    if not frames:
        return VisualFeatures(*(0.0,) * len(VisualFeatures.FEATURE_NAMES))

    brightnesses = []
    contrasts = []
    entropies = []
    mid_freq_ratios = []
    color_coherences = []
    autocorrelations = []
    spectral_flatnesses = []
    frame_diffs = []
    prev_gray = None

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        # Brightness
        brightnesses.append(gray.mean() / 255.0)

        # Contrast (p95 - p5)
        p5, p95 = np.percentile(gray, [5, 95])
        contrasts.append((p95 - p5) / 255.0)

        # Spatial entropy (Laplacian variance)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        entropies.append(min(lap.var() / 2000.0, 1.0))

        # Color coherence (palette peakedness — fewer dominant hues = better)
        hue_hist = cv2.calcHist([hsv], [0], None, [36], [0, 180]).flatten()
        hue_hist = hue_hist / (hue_hist.sum() + 1e-10)
        top3 = np.sort(hue_hist)[-3:].sum()
        color_coherences.append(float(top3))

        # Frame-to-frame diff (used for motion_magnitude + temporal_smoothness)
        if prev_gray is not None:
            diff_img = np.abs(gray.astype(float) - prev_gray.astype(float))
            frame_diffs.append(diff_img)
        prev_gray = gray

        # Mid-frequency ratio (3-band FFT: low / mid / high)
        gf = gray.astype(float)
        f = np.fft.fft2(gf)
        fshift = np.fft.fftshift(f)
        mag = np.abs(fshift)
        cy, cx = mag.shape[0] // 2, mag.shape[1] // 2
        max_r = min(cy, cx)
        r_low = max_r // 6
        r_mid = max_r // 2
        total = mag.sum() + 1e-10
        y, x = np.ogrid[:mag.shape[0], :mag.shape[1]]
        dist_sq = (y - cy)**2 + (x - cx)**2
        mid_mask = (dist_sq > r_low**2) & (dist_sq <= r_mid**2)
        mid_energy = mag[mid_mask].sum()
        mid_freq_ratios.append(float(mid_energy / total))

        # Spectral flatness — combination of Wiener entropy and peak-to-median
        # ratio. Wiener alone misses "soft periodic" patterns (halftone with
        # neighbor sampling). Peak-to-median catches any concentrated spectral
        # energy. Combined: both must be organic for full credit.
        mag_nz = mag[mag > 0].flatten()
        if len(mag_nz) > 0:
            # Wiener entropy: geometric_mean / arithmetic_mean
            log_mean = np.mean(np.log(mag_nz + 1e-10))
            geo_mean = np.exp(log_mean)
            arith_mean = np.mean(mag_nz)
            wiener = geo_mean / (arith_mean + 1e-10)
            # Peak-to-median: how much the strongest peaks exceed the median.
            # Exclude DC (center pixel). Subsample for speed.
            mag_no_dc = mag.copy()
            mag_no_dc[cy, cx] = 0
            mag_flat = mag_no_dc[mag_no_dc > 0].flatten()
            if len(mag_flat) > 10000:
                mag_flat = mag_flat[::len(mag_flat) // 10000]
            if len(mag_flat) > 0:
                median_mag = float(np.median(mag_flat))
                top_k = max(1, len(mag_flat) // 200)  # top 0.5%
                top_vals = np.partition(mag_flat, -top_k)[-top_k:]
                peak_ratio = float(np.mean(top_vals) / (median_mag + 1e-10))
                # Normalize: peak_ratio of 5 = organic, 50+ = periodic grid
                # Map to [0, 1]: 1.0 at ratio ≤ 5, 0.0 at ratio ≥ 40
                peak_score = float(np.clip(1.0 - (peak_ratio - 5.0) / 35.0, 0.0, 1.0))
            else:
                peak_score = 0.0
            # Combine: geometric mean so both must be high
            sf = float(min(1.0, (wiener * peak_score) ** 0.5))
            spectral_flatnesses.append(sf)
        else:
            spectral_flatnesses.append(0.0)

        # Spatial autocorrelation (structure vs noise)
        gf_norm = gf - gf.mean()
        var = (gf_norm * gf_norm).mean()
        if var > 1e-6:
            autocorrs = []
            for offset in (4, 8, 16):
                if offset < gf.shape[1] - 1:
                    corr_h = (gf_norm[:, :-offset] * gf_norm[:, offset:]).mean() / var
                    corr_v = (gf_norm[:-offset, :] * gf_norm[offset:, :]).mean() / var
                    autocorrs.append((corr_h + corr_v) / 2.0)
            if len(autocorrs) >= 2:
                mean_ac = sum(max(0, a) for a in autocorrs) / len(autocorrs)
                decay_ok = all(autocorrs[i] >= autocorrs[i+1] - 0.05
                               for i in range(len(autocorrs) - 1))
                ac_score = mean_ac * (1.0 if decay_ok else 0.5)
            else:
                ac_score = 0.0
        else:
            ac_score = 0.0
        autocorrelations.append(float(max(0.0, min(1.0, ac_score))))

    # Frame variance (temporal interest)
    frame_means = np.array(brightnesses)
    frame_var = float(frame_means.var()) if len(frame_means) > 1 else 0.0

    # Motion magnitude — simple mean pixel diff between frames.
    # The motion floor penalty in compute_fitness() handles the hard cutoff
    # for static output. No fancy filtering needed here.
    if frame_diffs:
        motion_mag = float(np.mean([d.mean() / 255.0 for d in frame_diffs]))
    else:
        motion_mag = 0.0

    # Temporal smoothness: 1 − coefficient of variation of per-frame mean diffs
    # Low CV = consistent motion (good), high CV = flickery (bad)
    if len(frame_diffs) >= 2:
        per_frame_means = np.array([d.mean() for d in frame_diffs])
        mean_diff = per_frame_means.mean()
        if mean_diff > 0.1:  # has meaningful motion
            cv = per_frame_means.std() / mean_diff
            temporal_smooth = float(max(0.0, min(1.0, 1.0 - cv)))
        else:
            temporal_smooth = 0.0  # static — no smoothness credit
    else:
        temporal_smooth = 0.0

    return VisualFeatures(
        brightness=float(np.mean(brightnesses)),
        contrast=float(np.mean(contrasts)),
        spatial_entropy=float(np.mean(entropies)),
        color_coherence=float(np.mean(color_coherences)),
        mid_frequency_ratio=float(np.mean(mid_freq_ratios)),
        spatial_autocorrelation=float(np.mean(autocorrelations)),
        spectral_flatness=float(np.mean(spectral_flatnesses)),
        frame_variance=frame_var,
        temporal_smoothness=temporal_smooth,
        motion_magnitude=motion_mag,
    )


def compute_fitness(
    features: VisualFeatures,
    weights: dict[str, float],
) -> float:
    """Weighted feature combination with brightness penalties."""
    score = 0.0
    for name in VisualFeatures.FEATURE_NAMES:
        w = weights.get(name, 0.0)
        score += w * getattr(features, name)

    # Penalize extreme brightness
    b = features.brightness
    if b < _BRIGHTNESS_PENALTY_DARK:
        score *= 0.1
    elif b > _BRIGHTNESS_PENALTY_BRIGHT:
        score *= 0.3

    # Penalize degenerate output (near-flat)
    if features.contrast < 0.02:
        score *= 0.05

    # Nonlinear entropy penalty: penalize both boring (< 0.1) and chaotic (> 0.85)
    e = features.spatial_entropy
    if e < 0.1:
        score *= 0.2 + 0.8 * (e / 0.1)   # ramp 0.2 → 1.0
    elif e > 0.85:
        score *= max(0.2, 1.0 - (e - 0.85) / 0.15)  # ramp 1.0 → 0.2

    # Spectral flatness as multiplicative penalty — periodic grids get crushed.
    # flatness < 0.3 → 0.1×, flatness 0.3–0.6 → ramp to 1.0×, above 0.6 → no penalty
    sf = features.spectral_flatness
    if sf < 0.3:
        score *= 0.1
    elif sf < 0.6:
        score *= 0.1 + 0.9 * ((sf - 0.3) / 0.3)

    # Motion floor — stacks that don't move get crushed regardless of spatial beauty
    # motion < 0.005 → 0.1×, motion 0.005–0.02 → ramp to 1.0×
    m = features.motion_magnitude
    if m < 0.005:
        score *= 0.1
    elif m < 0.02:
        score *= 0.1 + 0.9 * ((m - 0.005) / 0.015)

    return score


# ─── Genome construction ─────────────────────────────────────────────────────

def _sample_params(
    rng: _random_mod.Random,
    shader: ISFShader,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Generate concrete params and a param spec for a shader.

    Returns (concrete_params, param_spec).
    """
    from scripts.generate_stacks import _param_spec_for_input

    concrete: dict[str, Any] = {}
    spec: dict[str, Any] = {}

    for inp in shader.param_inputs:
        s = _param_spec_for_input(inp)
        if s is None:
            continue

        spec[inp.name] = s

        # Sample a concrete value from the spec
        if isinstance(s, dict) and "choice" in s:
            concrete[inp.name] = rng.choice(s["choice"])
        elif isinstance(s, list) and len(s) == 2:
            if isinstance(s[0], int) and isinstance(s[1], int):
                concrete[inp.name] = rng.randint(s[0], s[1])
            else:
                concrete[inp.name] = rng.uniform(s[0], s[1])
        else:
            concrete[inp.name] = s

    return concrete, spec


def _hex_uid(rng: _random_mod.Random) -> str:
    return hashlib.md5(str(rng.random()).encode()).hexdigest()[:8]


def random_genome(
    rng: _random_mod.Random,
    processors: list[ISFShader],
    min_size: int = 1,
    max_size: int = 5,
) -> Genome:
    """Create a random genome from available processor shaders."""
    size = rng.randint(min_size, min(max_size, len(processors)))
    chosen = rng.sample(processors, size)

    genes = []
    for shader in chosen:
        concrete, spec = _sample_params(rng, shader)
        genes.append(Gene(
            shader_stem=shader.path.stem,
            shader_path=shader.path,
            params=concrete,
            param_spec=spec,
        ))

    return Genome(genes=genes, uid=_hex_uid(rng))


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_genome(
    gl: GLContext,
    genome: Genome,
    source_frames: list[np.ndarray],
    w: int, h: int,
    weights: dict[str, float],
    frame_indices: list[int] | None = None,
) -> float:
    """Evaluate a genome: apply stack, extract features, compute fitness."""
    try:
        shaders = [parse_isf(g.shader_path) for g in genome.genes]
        param_overrides = {g.shader_stem: g.params for g in genome.genes}
        output_frames = apply_stack_to_frames(
            gl, source_frames, shaders, param_overrides, w, h,
            frame_indices=frame_indices,
        )
        features = extract_features(output_frames)
        genome.features = features.to_dict()
        genome.fitness = compute_fitness(features, weights)
    except Exception:
        genome.fitness = 0.0
        genome.features = {}
    return genome.fitness


# ─── Evolution: diversity-weighted random search ──────────────────────────────

def evolve(
    processors: list[ISFShader],
    generators: list[ISFShader],
    config: EvolveConfig,
    progress_callback: Optional[Callable] = None,
) -> list[Genome]:
    """Find diverse, high-quality stacks via random search + greedy selection.

    Phase 1: Generate and evaluate N random stacks.
    Phase 2: Greedily select K stacks that are both high-fitness
             and far apart in feature space.
    """
    rng = _random_mod.Random(config.seed)
    gl = GLContext()

    try:
        # Separate generators with/without inputImage
        true_generators = [g for g in generators
                           if not any(i.type == "image"
                                      for i in g.image_inputs)]

        # Generate source frames
        if true_generators:
            gen_shader = rng.choice(true_generators)
            source_frames, frame_indices = generate_source_frames(
                gl, gen_shader, config.eval_width, config.eval_height,
                n_frames=config.eval_frames, seed=config.seed,
            )
        else:
            source_frames, frame_indices = _noise_frames(
                config.seed, config.eval_width, config.eval_height,
                config.eval_frames,
            )

        # Qualifying round
        qualified = []
        reject_reasons = {}
        for shader in processors:
            params = shader.default_params()
            try:
                out_frames = apply_stack_to_frames(
                    gl, source_frames,
                    [shader], {shader.path.stem: params},
                    config.eval_width, config.eval_height,
                    frame_indices=frame_indices,
                )
            except Exception:
                reject_reasons[shader.path.stem] = "crash"
                continue

            diffs = []
            for src_f, out_f in zip(source_frames, out_frames):
                diffs.append(np.abs(src_f.astype(float) - out_f.astype(float)).mean() / 255.0)
            mean_diff = float(np.mean(diffs))
            out_brightness = float(np.mean([f.mean() for f in out_frames]) / 255.0)
            out_lumas = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in out_frames]
            out_contrast = float(np.mean([
                (np.percentile(g, 95) - np.percentile(g, 5)) / 255.0 for g in out_lumas
            ]))

            if mean_diff < 0.001:
                reject_reasons[shader.path.stem] = "no-op"
            elif out_brightness < 0.03 or out_brightness > 0.97:
                reject_reasons[shader.path.stem] = "degenerate brightness"
            elif out_contrast < 0.01:
                reject_reasons[shader.path.stem] = "degenerate contrast"
            else:
                qualified.append(shader)

        if progress_callback:
            progress_callback(-1, 0.0, [],
                              msg=f"qualified {len(qualified)}/{len(processors)} shaders "
                                  f"(rejected {len(processors) - len(qualified)}: "
                                  + ", ".join(f"{r}={sum(1 for v in reject_reasons.values() if v == r)}"
                                              for r in sorted(set(reject_reasons.values())))
                                  + ")")
        if len(qualified) < 3:
            qualified = processors
        processors = qualified

        # Phase 1: generate and evaluate random candidates
        candidates: list[Genome] = []
        n_source_rotations = max(1, config.n_candidates // 500)
        batch_size = config.n_candidates // n_source_rotations

        for batch in range(n_source_rotations):
            if batch > 0 and true_generators:
                gen_shader = rng.choice(true_generators)
                source_frames, frame_indices = generate_source_frames(
                    gl, gen_shader, config.eval_width, config.eval_height,
                    n_frames=config.eval_frames,
                    seed=config.seed + batch,
                )

            for i in range(batch_size):
                genome = random_genome(rng, processors,
                                       config.min_stack_size,
                                       config.max_stack_size)
                evaluate_genome(gl, genome, source_frames,
                                config.eval_width, config.eval_height,
                                config.weights, frame_indices=frame_indices)
                candidates.append(genome)

            if progress_callback:
                evaluated = len(candidates)
                best = max(candidates, key=lambda g: g.fitness)
                progress_callback(
                    batch + 1, best.fitness, [],
                    msg=f"evaluated {evaluated}/{config.n_candidates} candidates"
                )

        # Phase 2: greedy diversity-weighted selection
        # Filter to minimum fitness
        viable = [g for g in candidates if g.fitness >= config.min_fitness]
        if len(viable) < config.n_output:
            # Relax: take top n_output by fitness
            candidates.sort(key=lambda g: g.fitness, reverse=True)
            viable = candidates[:max(config.n_output * 2, 50)]

        if progress_callback:
            progress_callback(0, 0.0, [],
                              msg=f"{len(viable)} viable candidates "
                                  f"(fitness >= {config.min_fitness})")

        selected = _greedy_diverse_select(
            viable, config.n_output, config.diversity_weight)

        if progress_callback and selected:
            best = max(selected, key=lambda g: g.fitness)
            progress_callback(
                0, best.fitness, [],
                msg=f"selected {len(selected)} diverse stacks "
                    f"(fitness range {min(g.fitness for g in selected):.3f}"
                    f"–{max(g.fitness for g in selected):.3f})")

        return selected

    finally:
        gl.release()


def _greedy_diverse_select(
    candidates: list[Genome],
    n: int,
    diversity_weight: float,
) -> list[Genome]:
    """Greedily select n candidates maximizing fitness + diversity.

    Each pick maximizes: fitness + λ * combined_distance
    where combined_distance blends feature-space distance (Euclidean in
    normalized 10-feature space) with shader-set distance (1 − Jaccard
    similarity). This prevents the same dominant shaders from appearing
    in every selected stack.
    """
    if not candidates:
        return []

    # Build feature matrix and normalize each dimension to [0, 1]
    feature_names = VisualFeatures.FEATURE_NAMES
    features = np.array([
        [g.features.get(f, 0.0) for f in feature_names]
        for g in candidates
    ], dtype=np.float64)

    # Normalize columns to [0, 1]
    mins = features.min(axis=0)
    maxs = features.max(axis=0)
    ranges = maxs - mins
    ranges[ranges < 1e-10] = 1.0  # avoid div by zero
    normed = (features - mins) / ranges

    fitnesses = np.array([g.fitness for g in candidates], dtype=np.float64)
    # Normalize fitness to [0, 1] for fair weighting
    f_min, f_max = fitnesses.min(), fitnesses.max()
    if f_max - f_min > 1e-10:
        norm_fit = (fitnesses - f_min) / (f_max - f_min)
    else:
        norm_fit = np.ones_like(fitnesses)

    # Pre-compute shader stem sets for Jaccard distance
    shader_sets = [set(g.shader_stem for g in c.genes) for c in candidates]

    selected_indices: list[int] = []
    # Start with highest fitness
    selected_indices.append(int(np.argmax(fitnesses)))

    for _ in range(min(n - 1, len(candidates) - 1)):
        best_score = -1.0
        best_idx = -1

        for i in range(len(candidates)):
            if i in selected_indices:
                continue

            # Min feature-space distance to any already selected
            min_feat_dist = min(
                float(np.linalg.norm(normed[i] - normed[j]))
                for j in selected_indices
            )
            # Min shader-set distance (1 − Jaccard) to any already selected
            min_shader_dist = min(
                _jaccard_distance(shader_sets[i], shader_sets[j])
                for j in selected_indices
            )
            # Blend: equal weight to feature diversity and shader diversity
            combined_dist = 0.5 * min_feat_dist + 0.5 * min_shader_dist
            score = norm_fit[i] + diversity_weight * combined_dist
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx >= 0:
            selected_indices.append(best_idx)

    return [candidates[i] for i in selected_indices]


def _jaccard_distance(a: set, b: set) -> float:
    """1 − Jaccard similarity. Returns 1.0 for disjoint sets, 0.0 for identical."""
    if not a and not b:
        return 0.0
    return 1.0 - len(a & b) / len(a | b)


# ─── Output conversion ───────────────────────────────────────────────────────

def genomes_to_stacks_yaml(
    genomes: list[Genome],
    n_stacks: Optional[int] = None,
) -> dict[str, dict]:
    """Convert top genomes to stacks.yaml format.

    Deduplicates by shader chain — keeps the highest-fitness variant
    of each unique shader combination.
    """
    from scripts.generate_stacks import _random_slug

    rng = _random_mod.Random(42)
    used_slugs: set[str] = set()
    stacks: dict[str, dict] = {}
    seen_chains: set[tuple[str, ...]] = set()
    target = n_stacks or len(genomes)

    for genome in genomes:
        if len(stacks) >= target:
            break
        chain = tuple(g.shader_stem for g in genome.genes)
        if chain in seen_chains:
            continue
        seen_chains.add(chain)
        slug = _random_slug(rng, used_slugs)
        shader_names = [g.shader_stem for g in genome.genes]

        shader_params: dict[str, dict[str, Any]] = {}
        for gene in genome.genes:
            spec = _widen_params_to_spec(gene)
            if spec:
                shader_params[gene.shader_stem] = spec

        entry: dict[str, Any] = {"shaders": shader_names}
        if shader_params:
            entry["shader_params"] = shader_params
        stacks[slug] = entry

    return stacks


def _widen_params_to_spec(gene: Gene) -> dict[str, Any]:
    """Convert a gene's concrete params to a YAML spec with randomization ranges.

    Creates ±15% ranges around the evolved sweet spot, clamped to shader bounds.
    """
    try:
        shader = parse_isf(gene.shader_path)
    except Exception:
        return gene.param_spec

    inp_by_name = {inp.name: inp for inp in shader.param_inputs}
    spec: dict[str, Any] = {}

    for pname, val in gene.params.items():
        inp = inp_by_name.get(pname)
        if inp is None:
            continue

        if inp.type == "float" and inp.min is not None and inp.max is not None:
            full_range = float(inp.max) - float(inp.min)
            spread = full_range * 0.15
            lo = max(float(inp.min), float(val) - spread)
            hi = min(float(inp.max), float(val) + spread)
            if abs(hi - lo) < 0.001:
                spec[pname] = round(float(val), 4)
            else:
                spec[pname] = [round(lo, 4), round(hi, 4)]
        elif inp.type == "long":
            spec[pname] = int(val)
        elif inp.type == "bool":
            spec[pname] = {"choice": [0.0, 1.0]}

    return spec
