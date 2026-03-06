"""
Genetic algorithm for evolving shader stacks.

Evaluates candidate shader stacks by rendering frames in-memory via GL
and scoring visual features. No file I/O during fitness evaluation.

Usage (standalone):
    from pipeline.evolve import evolve, EvolutionConfig
    from scripts.generate_stacks import validate_shaders
    processors, generators, _ = validate_shaders(Path("packs/starter/shaders"))
    population = evolve(processors, generators, EvolutionConfig(seed=42))
"""

from __future__ import annotations

import hashlib
import random as _random_mod
from copy import deepcopy
from dataclasses import dataclass, field
from math import sqrt
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
    """Feature set for GA fitness evaluation — balanced spatial + temporal."""
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
class EvolutionConfig:
    population_size: int = 50
    generations: int = 20
    elite_count: int = 5
    tournament_k: int = 3
    crossover_rate: float = 0.7
    mutation_rate: float = 0.3
    min_stack_size: int = 1
    max_stack_size: int = 5
    eval_frames: int = 5
    eval_width: int = 640
    eval_height: int = 360
    seed: int = 42
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))


# ─── Fitness weights ──────────────────────────────────────────────────────────

DEFAULT_WEIGHTS: dict[str, float] = {
    "brightness": 0.0,          # penalty only
    "contrast": 1.2,
    "spatial_entropy": 1.5,
    "color_coherence": 1.0,
    "mid_frequency_ratio": 1.2,
    "spatial_autocorrelation": 1.2,
    "spectral_flatness": 0.0,   # multiplicative penalty, not additive
    "frame_variance": 2.0,
    "temporal_smoothness": 1.5,
    "motion_magnitude": 3.0,    # highest weight — motion is king
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
            diff = np.abs(gray.astype(float) - prev_gray.astype(float)).mean()
            frame_diffs.append(diff)
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

    # Motion magnitude: mean frame-to-frame pixel change, normalized to [0, 1]
    if frame_diffs:
        motion_mag = float(np.mean(frame_diffs) / 255.0)
    else:
        motion_mag = 0.0

    # Temporal smoothness: 1 − coefficient of variation of frame diffs
    # Low CV = consistent motion (good), high CV = flickery (bad)
    if len(frame_diffs) >= 2:
        diffs_arr = np.array(frame_diffs)
        mean_diff = diffs_arr.mean()
        if mean_diff > 0.1:  # has meaningful motion
            cv = diffs_arr.std() / mean_diff
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


def _shader_similarity(a: Genome, b: Genome) -> float:
    """Fraction of shared shader stems between two genomes (Jaccard)."""
    set_a = set(g.shader_stem for g in a.genes)
    set_b = set(g.shader_stem for g in b.genes)
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def apply_fitness_sharing(
    population: list[Genome],
    similarity_threshold: float = 0.6,
    sharing_strength: float = 0.3,
) -> None:
    """Light niche penalty to maintain population diversity.

    Genomes similar (Jaccard > threshold) to fitter ones get fitness
    reduced. Gentle enough to let fitness features drive selection —
    diversity should come from good features, not population penalties.
    """
    ranked = sorted(population, key=lambda g: g.fitness, reverse=True)
    for i, genome in enumerate(ranked):
        niche_count = 0
        for j in range(i):  # only compare to fitter genomes
            if _shader_similarity(genome, ranked[j]) > similarity_threshold:
                niche_count += 1
        if niche_count > 0:
            genome.fitness /= (1.0 + sharing_strength * niche_count)


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


# ─── Genetic operators ────────────────────────────────────────────────────────

def tournament_select(
    rng: _random_mod.Random,
    population: list[Genome],
    k: int = 3,
) -> Genome:
    """Pick k random individuals, return the fittest."""
    contestants = rng.sample(population, min(k, len(population)))
    return max(contestants, key=lambda g: g.fitness)


def crossover(
    rng: _random_mod.Random,
    parent_a: Genome,
    parent_b: Genome,
    max_size: int = 5,
) -> Genome:
    """Crossover two parents to produce a child."""
    method = rng.choice(["single_point", "param_blend", "graft"])

    if method == "single_point" and len(parent_a.genes) > 1 and len(parent_b.genes) > 1:
        cut_a = rng.randint(1, len(parent_a.genes) - 1)
        cut_b = rng.randint(1, len(parent_b.genes) - 1)
        child_genes = deepcopy(parent_a.genes[:cut_a]) + deepcopy(parent_b.genes[cut_b:])
        child_genes = child_genes[:max_size]

    elif method == "param_blend":
        # Keep parent_a's structure, blend params where shaders overlap
        child_genes = deepcopy(parent_a.genes)
        b_params = {g.shader_stem: g.params for g in parent_b.genes}
        for gene in child_genes:
            if gene.shader_stem in b_params:
                for k, v in b_params[gene.shader_stem].items():
                    if k in gene.params:
                        a_val = gene.params[k]
                        b_val = v
                        if isinstance(a_val, (int, float)) and isinstance(b_val, (int, float)):
                            t = rng.random()
                            blended = a_val * t + b_val * (1 - t)
                            gene.params[k] = int(round(blended)) if isinstance(a_val, int) else blended

    else:  # graft
        child_genes = deepcopy(parent_a.genes)
        if parent_b.genes:
            start = rng.randint(0, len(parent_b.genes) - 1)
            end = rng.randint(start + 1, len(parent_b.genes))
            graft_genes = deepcopy(parent_b.genes[start:end])
            insert_at = rng.randint(0, len(child_genes))
            child_genes = child_genes[:insert_at] + graft_genes + child_genes[insert_at:]
            child_genes = child_genes[:max_size]

    if not child_genes:
        child_genes = deepcopy(parent_a.genes[:1])

    return Genome(genes=child_genes, uid=_hex_uid(rng))


def mutate(
    rng: _random_mod.Random,
    genome: Genome,
    processors: list[ISFShader],
    mutation_rate: float = 0.3,
    min_size: int = 1,
    max_size: int = 5,
) -> Genome:
    """Apply mutations to a genome."""
    if rng.random() >= mutation_rate:
        return genome

    g = deepcopy(genome)
    g.uid = _hex_uid(rng)

    # Weighted mutation selection
    mutations = [
        ("perturb_params", 0.35),
        ("swap_positions", 0.25),
        ("replace_shader", 0.15),
        ("insert_shader", 0.10),
        ("delete_shader", 0.10),
        ("reverse_segment", 0.05),
    ]
    total = sum(w for _, w in mutations)
    roll = rng.random() * total
    cumulative = 0.0
    chosen = mutations[0][0]
    for name, weight in mutations:
        cumulative += weight
        if roll <= cumulative:
            chosen = name
            break

    if chosen == "perturb_params" and g.genes:
        gene = rng.choice(g.genes)
        if gene.params:
            param_name = rng.choice(list(gene.params.keys()))
            val = gene.params[param_name]
            if isinstance(val, float):
                gene.params[param_name] = val * (1.0 + rng.gauss(0, 0.2))
            elif isinstance(val, int):
                gene.params[param_name] = max(0, val + rng.choice([-1, 0, 1]))

    elif chosen == "swap_positions" and len(g.genes) >= 2:
        i, j = rng.sample(range(len(g.genes)), 2)
        g.genes[i], g.genes[j] = g.genes[j], g.genes[i]

    elif chosen == "replace_shader" and g.genes and processors:
        idx = rng.randint(0, len(g.genes) - 1)
        new_shader = rng.choice(processors)
        concrete, spec = _sample_params(rng, new_shader)
        g.genes[idx] = Gene(
            shader_stem=new_shader.path.stem,
            shader_path=new_shader.path,
            params=concrete,
            param_spec=spec,
        )

    elif chosen == "insert_shader" and len(g.genes) < max_size and processors:
        new_shader = rng.choice(processors)
        concrete, spec = _sample_params(rng, new_shader)
        pos = rng.randint(0, len(g.genes))
        g.genes.insert(pos, Gene(
            shader_stem=new_shader.path.stem,
            shader_path=new_shader.path,
            params=concrete,
            param_spec=spec,
        ))

    elif chosen == "delete_shader" and len(g.genes) > min_size:
        idx = rng.randint(0, len(g.genes) - 1)
        g.genes.pop(idx)

    elif chosen == "reverse_segment" and len(g.genes) >= 3:
        i = rng.randint(0, len(g.genes) - 2)
        j = rng.randint(i + 2, len(g.genes))
        g.genes[i:j] = g.genes[i:j][::-1]

    return g


# ─── Main evolution loop ─────────────────────────────────────────────────────

def evolve(
    processors: list[ISFShader],
    generators: list[ISFShader],
    config: EvolutionConfig,
    progress_callback: Optional[Callable] = None,
) -> list[Genome]:
    """Run the genetic algorithm. Returns final population sorted by fitness."""
    if not processors:
        return []

    rng = _random_mod.Random(config.seed)
    gl = GLContext()

    try:
        # Generate source frames from a random generator
        # Frames are spread across 4 seconds to catch temporal divergence
        if generators:
            gen_shader = rng.choice(generators)
            source_frames, frame_indices = generate_source_frames(
                gl, gen_shader, config.eval_width, config.eval_height,
                n_frames=config.eval_frames, seed=config.seed,
            )
        else:
            source_frames, frame_indices = _noise_frames(
                config.seed, config.eval_width, config.eval_height,
                config.eval_frames,
            )

        # Initialize population
        population = [
            random_genome(rng, processors, config.min_stack_size, config.max_stack_size)
            for _ in range(config.population_size)
        ]

        # Evaluate initial population
        for genome in population:
            evaluate_genome(gl, genome, source_frames,
                            config.eval_width, config.eval_height, config.weights,
                            frame_indices=frame_indices)
        apply_fitness_sharing(population)

        for gen in range(config.generations):
            population.sort(key=lambda g: g.fitness, reverse=True)

            # Elitism
            next_gen = [deepcopy(g) for g in population[:config.elite_count]]

            # Fill via selection + crossover + mutation
            while len(next_gen) < config.population_size:
                if rng.random() < config.crossover_rate:
                    a = tournament_select(rng, population, config.tournament_k)
                    b = tournament_select(rng, population, config.tournament_k)
                    child = crossover(rng, a, b, config.max_stack_size)
                else:
                    child = deepcopy(tournament_select(rng, population, config.tournament_k))

                child = mutate(rng, child, processors, config.mutation_rate,
                               config.min_stack_size, config.max_stack_size)
                next_gen.append(child)

            # Evaluate new individuals (skip elites)
            for genome in next_gen[config.elite_count:]:
                evaluate_genome(gl, genome, source_frames,
                                config.eval_width, config.eval_height, config.weights,
                                frame_indices=frame_indices)

            # Fitness sharing: penalize crowded niches
            apply_fitness_sharing(next_gen)

            # Rotate source frames every 5 generations
            if gen % 5 == 4 and generators:
                gen_shader = rng.choice(generators)
                source_frames, frame_indices = generate_source_frames(
                    gl, gen_shader, config.eval_width, config.eval_height,
                    n_frames=config.eval_frames, seed=config.seed + gen,
                )

            population = next_gen

            if progress_callback:
                best = max(population, key=lambda g: g.fitness)
                progress_callback(gen + 1, best.fitness, population)

        population.sort(key=lambda g: g.fitness, reverse=True)
        return population

    finally:
        gl.release()


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
