"""
ISF (Interactive Shader Format) loader.

Parses ISF v2 .fs files:
  1. Extracts the JSON header (inputs, passes, metadata)
  2. Translates the GLSL body to standard #version 330
     (replaces ISF built-ins with uniforms)
  3. Returns a structured ISFShader object ready for compilation

Handles single-pass and multi-pass (persistent buffer) shaders.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


# ─── ISF data structures ─────────────────────────────────────────────────────

@dataclass
class ISFInput:
    name: str
    type: str            # "float", "bool", "event", "image", "long", etc.
    default: Any = None
    min: Any = None
    max: Any = None
    label: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> ISFInput:
        return cls(
            name=d["NAME"],
            type=d["TYPE"],
            default=d.get("DEFAULT"),
            min=d.get("MIN"),
            max=d.get("MAX"),
            label=d.get("LABEL", ""),
        )


@dataclass
class ISFPass:
    target: Optional[str] = None
    persistent: bool = False
    float_tex: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> ISFPass:
        return cls(
            target=d.get("TARGET"),
            persistent=d.get("PERSISTENT", False),
            float_tex=d.get("FLOAT", False),
        )


@dataclass
class ISFShader:
    """Parsed ISF shader ready for compilation."""

    path: Path
    description: str
    credit: str
    categories: list[str]

    inputs: list[ISFInput]
    passes: list[ISFPass]

    # The original ISF GLSL body (before translation)
    isf_body: str
    # The translated #version 330 fragment shader source
    glsl_source: str

    @property
    def is_multipass(self) -> bool:
        return len(self.passes) > 1

    @property
    def has_persistent_buffer(self) -> bool:
        return any(p.persistent for p in self.passes)

    @property
    def param_inputs(self) -> list[ISFInput]:
        """Inputs that are shader parameters (not images)."""
        return [i for i in self.inputs if i.type != "image"]

    @property
    def image_inputs(self) -> list[ISFInput]:
        """Inputs that are images/textures."""
        return [i for i in self.inputs if i.type == "image"]

    def default_params(self) -> dict[str, Any]:
        """Return a dict of parameter names → default values.

        Handles all ISF input types:
        - float → float
        - bool/event → float (0.0 or 1.0)
        - long → int
        - color → tuple of 4 floats (RGBA)
        - point2D → tuple of 2 floats
        """
        out: dict[str, Any] = {}
        for inp in self.param_inputs:
            if inp.type == "float" and inp.default is not None:
                out[inp.name] = float(inp.default)
            elif inp.type == "bool":
                out[inp.name] = 1.0 if inp.default else 0.0
            elif inp.type == "event":
                out[inp.name] = 0.0
            elif inp.type == "long" and inp.default is not None:
                out[inp.name] = int(inp.default)
            elif inp.type == "color" and inp.default is not None:
                vals = [float(v) for v in inp.default]
                # Pad to 4 components if short
                while len(vals) < 4:
                    vals.append(1.0)
                out[inp.name] = tuple(vals[:4])
            elif inp.type == "point2D" and inp.default is not None:
                vals = [float(v) for v in inp.default]
                while len(vals) < 2:
                    vals.append(0.0)
                out[inp.name] = tuple(vals[:2])
        return out


# ─── ISF → GLSL translation ─────────────────────────────────────────────────

def _translate_isf_to_glsl(body: str, shader: ISFShader) -> str:
    """
    Translate ISF GLSL to standard #version 330.

    Replacements:
        isf_FragNormCoord       → v_texcoord
        vv_FragNormCoord        → v_texcoord  (v002 variant)
        IMG_NORM_PIXEL(tex, uv) → texture(tex, uv)
        IMG_THIS_PIXEL(tex)     → texture(tex, v_texcoord)
        IMG_THIS_NORM_PIXEL(tex)→ texture(tex, v_texcoord)
        IMG_PIXEL(tex, px)      → texelFetch(tex, ivec2(px), 0)
        RENDERSIZE              → u_rendersize
        TIME                    → u_time
        PASSINDEX               → u_passindex
        FRAMEINDEX              → u_frameindex
        gl_FragColor            → fragColor
    """

    src = body

    # Bool/event inputs are uniform floats but ISF shaders use them as
    # booleans (if(name), name ? a : b, etc.).  Two-pass fix for #version 330:
    #
    # Pass 1: Convert "name == true" / "name != false" etc. to float
    #   comparisons ("name > 0.5").  ISF convention uses true/false keywords
    #   but our uniforms are float, so float == bool is a type error.
    #
    # Pass 2: Wrap remaining standalone uses with bool() so if(name) works.
    #   Negative lookbehind for '.' prevents corrupting vec swizzle (.r).
    #   Negative lookahead for comparison operators prevents double-
    #   converting "name > 0.5" → "bool(name) > 0.5".
    for inp in shader.inputs:
        if inp.type in ("bool", "event"):
            esc = re.escape(inp.name)
            # Pass 1: true/false comparisons → float comparisons
            src = re.sub(rf'(?<!\.)\b{esc}\b\s*==\s*true\b',
                         f'{inp.name} > 0.5', src)
            src = re.sub(rf'(?<!\.)\b{esc}\b\s*!=\s*true\b',
                         f'{inp.name} < 0.5', src)
            src = re.sub(rf'(?<!\.)\b{esc}\b\s*==\s*false\b',
                         f'{inp.name} < 0.5', src)
            src = re.sub(rf'(?<!\.)\b{esc}\b\s*!=\s*false\b',
                         f'{inp.name} > 0.5', src)
            # Pass 2: general bool wrapping (skip float comparisons)
            src = re.sub(
                rf'(?<!\.)\b{esc}\b(?!\s*[><=!])',
                f'bool({inp.name})',
                src,
            )

    # IMG_THIS_PIXEL(texname) → texture(texname, v_texcoord)
    # (must come before IMG_PIXEL to avoid partial match)
    src = re.sub(
        r'IMG_THIS_PIXEL\s*\(\s*(\w+)\s*\)',
        r'texture(\1, v_texcoord)',
        src,
    )

    # IMG_THIS_NORM_PIXEL(texname) → texture(texname, v_texcoord)
    src = re.sub(
        r'IMG_THIS_NORM_PIXEL\s*\(\s*(\w+)\s*\)',
        r'texture(\1, v_texcoord)',
        src,
    )

    # IMG_NORM_PIXEL(texname, uv) → texture(texname, uv)
    # Use nested-paren-aware pattern: (?:[^()]+|\([^)]*\))+ handles
    # one level of nesting like vec2(0.5) inside the argument.
    src = re.sub(
        r'IMG_NORM_PIXEL\s*\(\s*(\w+)\s*,\s*((?:[^()]+|\([^)]*\))+)\)',
        r'texture(\1, \2)',
        src,
    )

    # IMG_PIXEL(texname, px) → texelFetch(texname, ivec2(px), 0)
    src = re.sub(
        r'IMG_PIXEL\s*\(\s*(\w+)\s*,\s*((?:[^()]+|\([^)]*\))+)\)',
        r'texelFetch(\1, ivec2(\2), 0)',
        src,
    )

    # Simple token replacements
    src = src.replace("isf_FragNormCoord", "v_texcoord")
    src = src.replace("vv_FragNormCoord", "v_texcoord")
    src = src.replace("RENDERSIZE", "u_rendersize")
    src = src.replace("PASSINDEX", "u_passindex")
    src = src.replace("FRAMEINDEX", "u_frameindex")
    # TIME is a common word — only replace as a standalone token
    src = re.sub(r'\bTIME\b', 'u_time', src)
    # gl_FragColor → fragColor (output)
    src = src.replace("gl_FragColor", "fragColor")
    # texture2D → texture (deprecated in #version 330)
    src = re.sub(r'\btexture2D\b', 'texture', src)

    # ── Build the full #version 330 source ────────────────────────────────

    lines = ["#version 330"]

    # Vertex input
    lines.append("in vec2 v_texcoord;")
    lines.append("out vec4 fragColor;")
    lines.append("")

    # Built-in uniforms
    lines.append("uniform float u_time;")
    lines.append("uniform vec2  u_rendersize;")
    if shader.is_multipass:
        lines.append("uniform int   u_passindex;")
    lines.append("uniform int   u_frameindex;")
    lines.append("")

    # ISF input uniforms
    for inp in shader.inputs:
        if inp.type == "image":
            lines.append(f"uniform sampler2D {inp.name};")
        elif inp.type == "float":
            lines.append(f"uniform float {inp.name};")
        elif inp.type in ("bool", "event"):
            # ISF bools/events are floats in practice (0 or 1).
            # The body is patched below to wrap uses with bool() so
            # if(name) works in #version 330.
            lines.append(f"uniform float {inp.name};")
        elif inp.type == "long":
            lines.append(f"uniform int {inp.name};")
        elif inp.type == "point2D":
            lines.append(f"uniform vec2 {inp.name};")
        elif inp.type == "color":
            lines.append(f"uniform vec4 {inp.name};")

    # Persistent buffer samplers
    for p in shader.passes:
        if p.target:
            lines.append(f"uniform sampler2D {p.target};")

    lines.append("")

    # Shader body
    lines.append(src)

    return "\n".join(lines)


# ─── Parser ──────────────────────────────────────────────────────────────────

def parse_isf(path: Path) -> ISFShader:
    """
    Parse an ISF .fs file and return an ISFShader with the translated GLSL.
    """
    raw = path.read_text().lstrip()

    # Extract the JSON header: everything between /*{ and }*/
    match = re.match(r'/\*\s*(\{.+?\})\s*\*/', raw, re.DOTALL)
    if not match:
        raise ValueError(f"No ISF JSON header found in {path}")

    # Strip trailing commas before } or ] (common ISF quirk, invalid JSON)
    json_str = re.sub(r',\s*([}\]])', r'\1', match.group(1))
    header = json.loads(json_str)
    body = raw[match.end():]

    inputs = [ISFInput.from_dict(d) for d in header.get("INPUTS", [])]
    passes = [ISFPass.from_dict(d) for d in header.get("PASSES", [{}])]

    # Cross-reference PERSISTENT_BUFFERS top-level key with pass targets
    # (some shaders declare persistence here instead of per-pass PERSISTENT flag)
    persistent_names = set(header.get("PERSISTENT_BUFFERS", []))
    for p in passes:
        if p.target in persistent_names:
            p.persistent = True

    shader = ISFShader(
        path=path,
        description=header.get("DESCRIPTION", ""),
        credit=header.get("CREDIT", ""),
        categories=header.get("CATEGORIES", []),
        inputs=inputs,
        passes=passes,
        isf_body=body,
        glsl_source="",  # filled below
    )

    shader.glsl_source = _translate_isf_to_glsl(body, shader)
    return shader


def load_shader_dir(shader_dir: Path) -> dict[str, ISFShader]:
    """
    Load all .fs files from a directory.
    Returns a dict of stem name → ISFShader.
    """
    shaders = {}
    for fs_path in sorted(shader_dir.glob("*.fs")):
        try:
            shaders[fs_path.stem] = parse_isf(fs_path)
        except Exception as e:
            print(f"Warning: failed to parse {fs_path.name}: {e}")
    return shaders
