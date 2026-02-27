"""
OpenGL / moderngl utilities.

Manages the standalone GL context, FBO pairs for ping-pong rendering,
and the full-screen quad geometry. All GPU rendering goes through here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import moderngl
import numpy as np


# ─── Full-screen quad ────────────────────────────────────────────────────────

QUAD_VERTS = np.array([
    -1.0, -1.0,
     1.0, -1.0,
    -1.0,  1.0,
     1.0, -1.0,
     1.0,  1.0,
    -1.0,  1.0,
], dtype="f4")

VERT_SHADER = """
#version 330
in  vec2 in_vert;
out vec2 v_texcoord;
void main() {
    gl_Position = vec4(in_vert, 0.0, 1.0);
    v_texcoord  = in_vert * 0.5 + 0.5;
}
"""


# ─── FBO pair for ping-pong rendering ────────────────────────────────────────

@dataclass
class FBOSlot:
    fbo: moderngl.Framebuffer
    texture: moderngl.Texture

    def use(self):
        self.fbo.use()

    def read(self, components: int = 3) -> bytes:
        return self.fbo.read(components=components)

    def release(self):
        self.fbo.release()
        self.texture.release()


def make_fbo_pair(ctx: moderngl.Context, w: int, h: int,
                  float_tex: bool = False) -> list[FBOSlot]:
    """Create two FBO/texture pairs for ping-pong rendering."""
    def _make():
        if float_tex:
            tex = ctx.texture((w, h), 4, dtype="f4")
        else:
            tex = ctx.texture((w, h), 3)
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        fbo = ctx.framebuffer(color_attachments=[tex])
        return FBOSlot(fbo=fbo, texture=tex)
    return [_make(), _make()]


# ─── GL context wrapper ──────────────────────────────────────────────────────

class GLContext:
    """
    Manages a standalone moderngl context with a shared VBO and quad geometry.

    Usage:
        gl = GLContext()
        prog = gl.compile(frag_source)
        vao = gl.vao(prog)
        # ... render ...
        gl.release()
    """

    def __init__(self):
        self.ctx = moderngl.create_standalone_context()
        self._vbo = self.ctx.buffer(QUAD_VERTS.tobytes())

    def compile(self, fragment_shader: str,
                vertex_shader: str = VERT_SHADER) -> moderngl.Program:
        """Compile a vertex + fragment shader pair into a program."""
        return self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader,
        )

    def vao(self, program: moderngl.Program) -> moderngl.VertexArray:
        """Create a VAO binding the quad VBO to a compiled program."""
        return self.ctx.simple_vertex_array(program, self._vbo, "in_vert")

    def texture(self, w: int, h: int, data: Optional[bytes] = None,
                components: int = 3) -> moderngl.Texture:
        """Create a texture, optionally uploading raw pixel data."""
        tex = self.ctx.texture((w, h), components, data=data)
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        return tex

    def fbo_pair(self, w: int, h: int,
                 float_tex: bool = False) -> list[FBOSlot]:
        """Create a ping-pong FBO pair."""
        return make_fbo_pair(self.ctx, w, h, float_tex)

    def release(self):
        self._vbo.release()
        self.ctx.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
