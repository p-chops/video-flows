"""
Prefect tasks for applying ISF shader stacks to video via moderngl.

Each frame is piped through ffmpeg → numpy → OpenGL texture → shader(s)
→ readback → numpy → ffmpeg.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from prefect import task
from ..cache import FILE_VALIDATED_INPUTS

from ..config import Config
from ..ffmpeg import probe, read_frames, FrameWriter
from ..gl import GLContext, FBOSlot
from ..isf import ISFShader, parse_isf, load_shader_dir


class _PersistentBuffer:
    """Ping-pong FBO pair for a persistent ISF buffer target."""
    __slots__ = ('slots', 'read_idx')

    def __init__(self, slots):
        self.slots = slots
        self.read_idx = 0

    @property
    def read_texture(self):
        return self.slots[self.read_idx].texture

    @property
    def write_slot(self):
        return self.slots[1 - self.read_idx]

    def swap(self):
        self.read_idx = 1 - self.read_idx


def _apply_shader_stack(
    src: Path,
    dst: Path,
    shaders: list[ISFShader],
    param_overrides: Optional[dict[str, dict[str, float]]] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """
    Core routine: read src frame-by-frame, apply each shader in sequence
    (ping-pong FBO), write result to dst.

    param_overrides: optional dict of {shader_stem: {param: value}}
    """
    c = cfg or Config()
    info = probe(src, c)
    w, h = info.width, info.height
    overrides = param_overrides or {}

    gl = GLContext()

    # Compile each shader and collect programs + VAOs + params + buffers
    compiled = []
    for shader in shaders:
        prog = gl.compile(shader.glsl_source)
        shader_vao = gl.vao(prog)
        params = shader.default_params()
        params.update(overrides.get(shader.path.stem, {}))

        # Create buffer FBOs for multi-pass shaders
        buffers: dict[str, _PersistentBuffer] = {}
        temp_targets: dict[str, FBOSlot] = {}
        for p in shader.passes:
            if p.target and p.persistent:
                slots = gl.fbo_pair(w, h, float_tex=p.float_tex)
                for slot in slots:
                    slot.fbo.clear()
                buffers[p.target] = _PersistentBuffer(slots)
            elif p.target and not p.persistent:
                # Non-persistent: single FBO for inter-pass communication
                pair = gl.fbo_pair(w, h, float_tex=p.float_tex)
                pair[1].release()  # only need one
                pair[0].fbo.clear()
                temp_targets[p.target] = pair[0]

        compiled.append((shader, prog, shader_vao, params, buffers, temp_targets))

    # Ping-pong FBOs for shader chaining
    fbo_a, fbo_b = gl.fbo_pair(w, h)
    input_tex = gl.texture(w, h)

    frame_idx = 0

    with FrameWriter(dst, w, h, fps=info.fps, cfg=c) as writer:
        for frame in read_frames(src, cfg=c):
            elapsed = frame_idx / info.fps

            # Upload source frame to input texture
            input_tex.write(np.flipud(frame).tobytes())

            # Chain shaders: input → shader[0] → shader[1] → ...
            current_tex = input_tex
            write_fbo = fbo_a

            for shader, prog, shader_vao, params, buffers, temp_targets in compiled:
                # Common uniforms
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
                    # Multi-pass: render through buffer FBOs
                    for pass_idx, pass_info in enumerate(shader.passes):
                        if pass_info.target and pass_info.target in buffers:
                            buffers[pass_info.target].write_slot.fbo.use()
                        elif pass_info.target and pass_info.target in temp_targets:
                            temp_targets[pass_info.target].fbo.use()
                        else:
                            write_fbo.fbo.use()

                        if "u_passindex" in prog:
                            prog["u_passindex"] = pass_idx

                        current_tex.use(location=0)
                        if "inputImage" in prog:
                            prog["inputImage"] = 0

                        # Bind persistent buffer textures
                        next_loc = 1
                        for buf_name, pbuf in buffers.items():
                            pbuf.read_texture.use(location=next_loc)
                            if buf_name in prog:
                                prog[buf_name] = next_loc
                            next_loc += 1

                        # Bind non-persistent target textures
                        for buf_name, slot in temp_targets.items():
                            slot.texture.use(location=next_loc)
                            if buf_name in prog:
                                prog[buf_name] = next_loc
                            next_loc += 1

                        shader_vao.render()

                        if pass_info.target and pass_info.target in buffers:
                            buffers[pass_info.target].swap()
                else:
                    # Single-pass
                    write_fbo.fbo.use()

                    current_tex.use(location=0)
                    if "inputImage" in prog:
                        prog["inputImage"] = 0

                    shader_vao.render()

                # Output of this stage becomes input of the next
                current_tex = write_fbo.texture
                write_fbo = fbo_b if write_fbo is fbo_a else fbo_a

            # Read back the result from whichever FBO was last written
            result_fbo = fbo_b if write_fbo is fbo_a else fbo_a
            raw = result_fbo.fbo.read(components=3)
            out_frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
            writer.write(np.flipud(out_frame))

            frame_idx += 1

    # Release buffer FBOs
    for _, _, _, _, buffers, temp_targets in compiled:
        for pbuf in buffers.values():
            for slot in pbuf.slots:
                slot.release()
        for slot in temp_targets.values():
            slot.release()

    gl.release()
    return dst


@task(name="apply-shader", cache_policy=FILE_VALIDATED_INPUTS, persist_result=True)
def apply_shader(src: Path, dst: Path, shader_path: Path,
                 params: Optional[dict[str, float]] = None,
                 cfg: Optional[Config] = None) -> Path:
    """
    Apply a single ISF shader to every frame of src → dst.
    """
    shader = parse_isf(shader_path)
    overrides = {shader.path.stem: params} if params else {}
    return _apply_shader_stack(src, dst, [shader],
                               param_overrides=overrides, cfg=cfg)


@task(name="apply-shader-stack", cache_policy=FILE_VALIDATED_INPUTS, persist_result=True)
def apply_shader_stack(src: Path, dst: Path,
                       shader_paths: list[Path],
                       param_overrides: Optional[dict[str, dict[str, float]]] = None,
                       cfg: Optional[Config] = None) -> Path:
    """
    Apply a sequence of ISF shaders to every frame (chained in order).
    """
    shaders = [parse_isf(p) for p in shader_paths]
    return _apply_shader_stack(src, dst, shaders,
                               param_overrides=param_overrides, cfg=cfg)


@task(name="apply-random-shader-stack", cache_policy=FILE_VALIDATED_INPUTS, persist_result=True)
def apply_random_shader_stack(
    src: Path, dst: Path,
    shader_dir: Path,
    min_shaders: int = 1, max_shaders: int = 4,
    seed: Optional[int] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """
    Pick a random subset of shaders from a directory and apply them
    with randomised parameters. Great for generative exploration.
    """
    import random as rng
    if seed is not None:
        rng.seed(seed)

    all_shaders = load_shader_dir(shader_dir)
    if not all_shaders:
        raise ValueError(f"No .fs shaders found in {shader_dir}")

    count = rng.randint(min_shaders, min(max_shaders, len(all_shaders)))
    chosen_names = rng.sample(list(all_shaders.keys()), count)
    shaders = [all_shaders[n] for n in chosen_names]

    # Randomise float params within their declared min/max
    overrides: dict[str, dict[str, float]] = {}
    for shader in shaders:
        sp = {}
        for inp in shader.param_inputs:
            if inp.type == "float" and inp.min is not None and inp.max is not None:
                sp[inp.name] = inp.min + rng.random() * (inp.max - inp.min)
        if sp:
            overrides[shader.path.stem] = sp

    return _apply_shader_stack(src, dst, shaders,
                               param_overrides=overrides, cfg=cfg)
