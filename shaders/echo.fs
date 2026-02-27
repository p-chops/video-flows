/*{
    "DESCRIPTION": "Multi-layer spatial ghost with per-layer colour tint. Two to four copies of the input are sampled at slowly drifting spatial offsets and composited with distance-weighted blending. Layer 0 is blue-shifted; the final layer is red-shifted, giving the characteristic warm-to-cool gradient of broadcast ghosting or CRT phosphor persistence.",
    "CREDIT": "P-Chops / Undersea Lair Project",
    "ISFVSN": "2",
    "INPUTS": [
        {
            "NAME": "inputImage",
            "TYPE": "image"
        },
        {
            "NAME": "intensity",
            "TYPE": "float",
            "DEFAULT": 0.50,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Intensity"
        },
        {
            "NAME": "drift",
            "TYPE": "float",
            "DEFAULT": 0.50,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Ghost Drift"
        },
        {
            "NAME": "num_ghosts",
            "TYPE": "float",
            "DEFAULT": 0.50,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Ghost Layers (2–4)"
        }
    ]
}*/

void main() {
    vec2  uv  = isf_FragNormCoord;
    vec3  src = IMG_NORM_PIXEL(inputImage, uv).rgb;

    int   n    = int(floor(2.0 + num_ghosts * 2.0));   // 2–4 layers
    vec3  acc  = vec3(0.0);
    float wsum = 0.0;

    for (int i = 0; i < 4; i++) {
        if (i >= n) break;
        float t = float(i) / max(float(n - 1), 1.0);   // 0 → 1

        // Each layer drifts in a slowly rotating direction (π/2 per layer).
        float angle  = TIME * 0.25 + t * 1.5708;
        vec2  offset = vec2(cos(angle), sin(angle) * 0.4) * drift * 0.06 * t;
        vec3  ghost  = IMG_NORM_PIXEL(inputImage, clamp(uv + offset, 0.0, 1.0)).rgb;

        // Tint: first layer blue-shifted, last layer red-shifted.
        ghost.r = clamp(ghost.r * (1.0 + t * 0.5),         0.0, 1.0);
        ghost.b = clamp(ghost.b * (1.0 + (1.0 - t) * 0.5), 0.0, 1.0);

        float w = 1.0 - t * 0.55;
        acc  += ghost * w;
        wsum += w;
    }

    vec3 composite = acc / wsum;
    gl_FragColor = vec4(mix(src, composite, intensity), 1.0);
}
