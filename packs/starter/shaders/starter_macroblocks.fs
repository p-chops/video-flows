/*{
    "DESCRIPTION": "Simulates H.264 macroblock compression artifacts: temporal prediction errors (stale blocks frozen on old content), DCT-style quantisation, and bidirectional block-edge ringing. Blocks hold their stale/fresh state for ~0.5s so motion clearly reveals frozen blocks lagging behind the live image.",
    "CREDIT": "Undersea Lair Project",
    "ISFVSN": "2",
    "CATEGORIES": ["Stylize"],
    "INPUTS": [
        {
            "NAME": "inputImage",
            "TYPE": "image"
        },
        {
            "NAME": "block_size",
            "TYPE": "float",
            "DEFAULT": 16.0,
            "MIN": 4.0,
            "MAX": 64.0
        },
        {
            "NAME": "stale_prob",
            "TYPE": "float",
            "DEFAULT": 0.3,
            "MIN": 0.0,
            "MAX": 0.95
        },
        {
            "NAME": "ringing",
            "TYPE": "float",
            "DEFAULT": 0.08,
            "MIN": 0.0,
            "MAX": 0.5
        },
        {
            "NAME": "quantize",
            "TYPE": "float",
            "DEFAULT": 16.0,
            "MIN": 2.0,
            "MAX": 64.0
        }
    ],
    "PASSES": [
        {
            "TARGET": "macroBuffer",
            "PERSISTENT": true
        }
    ]
}*/

// 2D → scalar hash (Inigo Quilez)
float hash21(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

void main() {
    vec2 uv  = isf_FragNormCoord;
    vec2 res = RENDERSIZE;
    vec2 px  = uv * res;
    float bs = max(block_size, 1.0);

    vec2 block = floor(px / bs);   // block index
    vec2 cell  = fract(px / bs);   // 0..1 within the block

    // Per-block, per-tick random — decisions change at ~2 ticks/sec so
    // blocks hold their stale/fresh state for ~0.5s worth of frames.
    // This lets stale blocks visibly fall behind moving content before
    // they snap back, rather than flickering every frame.
    float tick = floor(u_time * 2.0);
    float r    = hash21(block + tick * vec2(7.13, 3.71));
    bool is_stale = r < stale_prob;

    // Stale block: show content from macroBuffer (previous frame's composite).
    // Because macroBuffer holds last frame's output — which may itself have
    // contained stale blocks — staleness naturally accumulates across frames.
    // Fresh block: sample inputImage and apply DCT-style quantisation.
    vec4 color;
    if (is_stale) {
        color = IMG_NORM_PIXEL(macroBuffer, uv);
    } else {
        color = IMG_NORM_PIXEL(inputImage, uv);
        color.rgb = floor(color.rgb * quantize) / quantize;
    }

    // Block-edge ringing: damped sinusoidal overshoot at block boundaries,
    // scaled by the contrast between neighbouring blocks (ringing is
    // loudest where the DCT prediction error is largest).
    if (ringing > 0.001) {
        vec2  edgePx  = min(cell, 1.0 - cell) * bs;  // px distance to nearest edge
        float nearPx  = min(edgePx.x, edgePx.y);

        // Bidirectional decaying oscillation from edge (overshoot + undershoot).
        // Full 2π cycle so ring alternates bright then dark, not just bright.
        float ring = sin(nearPx * 6.28318) * exp(-nearPx * 0.7);

        // Contrast between this block and its right neighbour
        vec2 ctrUV = (block + 0.5) * bs / res;
        vec2 nbrUV = clamp((block + vec2(1.5, 0.5)) * bs / res, 0.0, 1.0);
        float contrast = length(
            IMG_NORM_PIXEL(inputImage, ctrUV).rgb -
            IMG_NORM_PIXEL(inputImage, nbrUV).rgb
        );

        color.rgb = clamp(color.rgb + ring * contrast * ringing, 0.0, 1.0);
    }

    gl_FragColor = vec4(color.rgb, 1.0);
}