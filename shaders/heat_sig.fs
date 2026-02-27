/*{
    "DESCRIPTION": "Thermal camera look — constrained warm palette only. Black to deep purple to red to orange to hot white. No rainbow, just heat.",
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
            "DEFAULT": 0.75,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Intensity"
        },
        {
            "NAME": "exposure",
            "TYPE": "float",
            "DEFAULT": 0.50,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Heat Exposure"
        },
        {
            "NAME": "hotspot",
            "TYPE": "float",
            "DEFAULT": 0.30,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Hotspot Bloom"
        }
    ]
}*/

void main() {
    vec2 uv = isf_FragNormCoord;
    vec3 src = IMG_NORM_PIXEL(inputImage, uv).rgb;
    float luma = dot(src, vec3(0.299, 0.587, 0.114));

    // Shift the luma curve with exposure
    float t = clamp(luma * mix(0.7, 1.6, exposure), 0.0, 1.0);

    // Warm-only ramp: muted violet → crimson → orange → hot white
    // Lifted shadow floor — no near-black zones
    vec3 col;
    if (t < 0.2) {
        // Muted violet
        col = mix(vec3(0.15, 0.06, 0.2), vec3(0.3, 0.08, 0.3), t * 5.0);
    } else if (t < 0.4) {
        // Violet to crimson
        col = mix(vec3(0.3, 0.08, 0.3), vec3(0.75, 0.15, 0.12), (t - 0.2) * 5.0);
    } else if (t < 0.65) {
        // Crimson to orange
        col = mix(vec3(0.75, 0.15, 0.12), vec3(1.0, 0.55, 0.08), (t - 0.4) * 4.0);
    } else if (t < 0.85) {
        // Orange to hot yellow
        col = mix(vec3(1.0, 0.55, 0.08), vec3(1.0, 0.9, 0.5), (t - 0.65) * 5.0);
    } else {
        // Hot yellow to white
        col = mix(vec3(1.0, 0.9, 0.5), vec3(1.0, 1.0, 0.95), (t - 0.85) * 6.67);
    }

    // Hotspot bloom — bright areas bleed outward
    if (hotspot > 0.01) {
        float blurLuma = 0.0;
        float px = 1.0 / RENDERSIZE.x;
        float py = 1.0 / RENDERSIZE.y;
        // Simple 5-tap cross sample for glow
        blurLuma += dot(IMG_NORM_PIXEL(inputImage, uv + vec2(px * 3.0, 0.0)).rgb, vec3(0.299, 0.587, 0.114));
        blurLuma += dot(IMG_NORM_PIXEL(inputImage, uv - vec2(px * 3.0, 0.0)).rgb, vec3(0.299, 0.587, 0.114));
        blurLuma += dot(IMG_NORM_PIXEL(inputImage, uv + vec2(0.0, py * 3.0)).rgb, vec3(0.299, 0.587, 0.114));
        blurLuma += dot(IMG_NORM_PIXEL(inputImage, uv - vec2(0.0, py * 3.0)).rgb, vec3(0.299, 0.587, 0.114));
        blurLuma *= 0.25;
        float bloom = max(blurLuma - 0.5, 0.0) * 2.0;
        col += vec3(bloom * 0.3, bloom * 0.15, 0.0) * hotspot;
    }

    gl_FragColor = vec4(mix(src, col, intensity), 1.0);
}
