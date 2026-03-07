/*{
  "DESCRIPTION": "ULP Domain Warp Cascade — recursive FBM where each layer warps the next. Produces alien landscape formations that slowly morph between states. Higher structural complexity than single-pass FBM.",
  "CREDIT": "Undersea Lair Project / P-Chops",
  "ISFVSN": "2",
  "CATEGORIES": ["Generator", "Ambient", "ULP"],
  "INPUTS": [
    {
      "NAME": "warp_depth",
      "TYPE": "float",
      "DEFAULT": 0.5,
      "MIN": 0.0,
      "MAX": 1.0,
      "LABEL": "Warp Depth  [recursion intensity]"
    },
    {
      "NAME": "morph_speed",
      "TYPE": "float",
      "DEFAULT": 0.4,
      "MIN": 0.0,
      "MAX": 1.0,
      "LABEL": "Morph Speed  [evolution rate]"
    },
    {
      "NAME": "terrain_scale",
      "TYPE": "float",
      "DEFAULT": 0.5,
      "MIN": 0.0,
      "MAX": 1.0,
      "LABEL": "Terrain Scale  [feature size]"
    },
    {
      "NAME": "palette_mode",
      "TYPE": "float",
      "DEFAULT": 0.3,
      "MIN": 0.0,
      "MAX": 1.0,
      "LABEL": "Palette  [0=geological, 0.5=alien, 1=toxic]"
    },
    {
      "NAME": "ridge_enhance",
      "TYPE": "float",
      "DEFAULT": 0.3,
      "MIN": 0.0,
      "MAX": 1.0,
      "LABEL": "Ridge Enhance  [sharpens folded structures]"
    },
    {
      "NAME": "brightness",
      "TYPE": "float",
      "DEFAULT": 1.0,
      "MIN": 0.0,
      "MAX": 2.0,
      "LABEL": "Brightness"
    }
  ]
}*/

// ─── Noise ──────────────────────────────────────────────────────────────────

vec2 hash2(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)),
             dot(p, vec2(269.5, 183.3)));
    return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
}

float snoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(dot(hash2(i + vec2(0.0, 0.0)), f - vec2(0.0, 0.0)),
                   dot(hash2(i + vec2(1.0, 0.0)), f - vec2(1.0, 0.0)), u.x),
               mix(dot(hash2(i + vec2(0.0, 1.0)), f - vec2(0.0, 1.0)),
                   dot(hash2(i + vec2(1.0, 1.0)), f - vec2(1.0, 1.0)), u.x), u.y);
}

float fbm(vec2 p) {
    float v = 0.0;
    float amp = 0.5;
    float freq = 1.0;
    for (int i = 0; i < 6; i++) {
        v += amp * snoise(p * freq);
        freq *= 2.1;
        amp *= 0.48;
    }
    return v;
}

// Ridged FBM — absolute value creates sharp ridges
float ridgedFBM(vec2 p) {
    float v = 0.0;
    float amp = 0.5;
    float freq = 1.0;
    for (int i = 0; i < 6; i++) {
        float n = snoise(p * freq);
        n = 1.0 - abs(n); // Ridge
        n = n * n;         // Sharpen
        v += amp * n;
        freq *= 2.1;
        amp *= 0.48;
    }
    return v;
}

// ─── Main ───────────────────────────────────────────────────────────────────

void main() {
    vec2 uv = isf_FragNormCoord * 2.0 - 1.0;
    uv.x *= RENDERSIZE.x / RENDERSIZE.y;

    float scale = mix(1.5, 4.0, terrain_scale);
    float speed = mix(0.1, 0.45, morph_speed);
    float warpStr = mix(2.0, 6.0, warp_depth);

    vec2 pos = uv * scale;

    // Time evolution — use circular orbits to avoid net drift
    float e1 = TIME * speed;
    float e2 = TIME * speed * 0.73;
    float e3 = TIME * speed * 0.53;
    vec2 evo1 = vec2(sin(e1), cos(e1)) * 1.3;
    vec2 evo2 = vec2(sin(e2), cos(e2)) * 0.9;
    vec2 evo3 = vec2(sin(e3), cos(e3)) * 0.7;

    // ─── Four-layer domain warp cascade ─────────────────────────────────────
    // Each layer feeds into the next, creating recursive distortion

    // Layer 1: base displacement
    vec2 q = vec2(
        fbm(pos + vec2(0.0, 0.0) + evo1),
        fbm(pos + vec2(5.2, 1.3) + evo1)
    );

    // Layer 2: warp by layer 1
    vec2 r = vec2(
        fbm(pos + warpStr * q + vec2(1.7, 9.2) + evo2),
        fbm(pos + warpStr * q + vec2(8.3, 2.8) + evo2)
    );

    // Layer 3: warp by layer 2
    vec2 s = vec2(
        fbm(pos + warpStr * 0.7 * r + vec2(3.4, 7.1) + evo3),
        fbm(pos + warpStr * 0.7 * r + vec2(6.7, 4.9) + evo3)
    );

    // Final sample — blends smooth and ridged FBM
    vec2 finalPos = pos + warpStr * 0.5 * s + evo3 * 0.5;
    float fSmooth = fbm(finalPos);
    float fRidged = ridgedFBM(finalPos);
    float f = mix(fSmooth, fRidged, ridge_enhance);
    f = 0.5 + 0.5 * f;

    // Micro-detail layer — high frequency noise modulated by warp structure
    float micro = snoise(finalPos * 8.0 + evo1 * 0.3);
    f += micro * 0.08;  // subtle texture grain

    // Sharper contrast — wider gap between darks and brights
    f = smoothstep(0.08, 0.82, f);

    // Secondary value for color variation — use q/r divergence
    float colorVar = 0.5 + 0.5 * dot(normalize(q + 0.001), normalize(r + 0.001));
    // Third channel — warp fold lines create color boundaries
    float foldLine = abs(fSmooth - fRidged);
    foldLine = smoothstep(0.0, 0.3, foldLine);

    // ─── Palette ────────────────────────────────────────────────────────────

    vec3 col;
    if (palette_mode < 0.33) {
        // Geological — sandstone, slate, oxidized copper
        vec3 c0 = vec3(0.15, 0.1, 0.08);
        vec3 c1 = vec3(0.45, 0.3, 0.18);
        vec3 c2 = vec3(0.6, 0.55, 0.4);
        vec3 c3 = vec3(0.35, 0.5, 0.45);
        vec3 c4 = vec3(0.8, 0.75, 0.6);

        if (f < 0.25) col = mix(c0, c1, f / 0.25);
        else if (f < 0.5) col = mix(c1, c2, (f - 0.25) / 0.25);
        else if (f < 0.75) col = mix(c2, c3, (f - 0.5) / 0.25);
        else col = mix(c3, c4, (f - 0.75) / 0.25);
    } else if (palette_mode < 0.66) {
        // Alien — indigo, magenta, cyan, gold
        vec3 c0 = vec3(0.05, 0.02, 0.15);
        vec3 c1 = vec3(0.3, 0.05, 0.35);
        vec3 c2 = vec3(0.1, 0.3, 0.5);
        vec3 c3 = vec3(0.5, 0.35, 0.1);
        vec3 c4 = vec3(0.15, 0.6, 0.5);

        if (f < 0.25) col = mix(c0, c1, f / 0.25);
        else if (f < 0.5) col = mix(c1, c2, (f - 0.25) / 0.25);
        else if (f < 0.75) col = mix(c2, c3, (f - 0.5) / 0.25);
        else col = mix(c3, c4, (f - 0.75) / 0.25);
    } else {
        // Toxic — dark green, bile yellow, bruise purple, acid
        vec3 c0 = vec3(0.02, 0.08, 0.02);
        vec3 c1 = vec3(0.15, 0.3, 0.05);
        vec3 c2 = vec3(0.5, 0.5, 0.05);
        vec3 c3 = vec3(0.3, 0.1, 0.3);
        vec3 c4 = vec3(0.4, 0.7, 0.1);

        if (f < 0.25) col = mix(c0, c1, f / 0.25);
        else if (f < 0.5) col = mix(c1, c2, (f - 0.25) / 0.25);
        else if (f < 0.75) col = mix(c2, c3, (f - 0.5) / 0.25);
        else col = mix(c3, c4, (f - 0.75) / 0.25);
    }

    // Color variation from warp layers + fold lines for extra detail
    col = mix(col, col.gbr, colorVar * 0.3);
    // Fold lines brighten where smooth and ridged FBM diverge
    col += col * foldLine * 0.4;

    col *= brightness;

    // Vignette
    float vign = 1.0 - dot(uv * 0.35, uv * 0.35);
    col *= clamp(vign, 0.0, 1.0);

    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
