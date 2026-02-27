/*{
    "DESCRIPTION": "Neon accent color that bleeds into the image from bright regions. One dominant hue pools and spreads like a neon sign casting light on a dark scene. Color concentrates in highlights and feathers into midtones.",
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
            "DEFAULT": 0.70,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Intensity"
        },
        {
            "NAME": "hue",
            "TYPE": "float",
            "DEFAULT": 0.55,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Neon Hue"
        },
        {
            "NAME": "spread",
            "TYPE": "float",
            "DEFAULT": 0.50,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Spread"
        },
        {
            "NAME": "saturation",
            "TYPE": "float",
            "DEFAULT": 0.70,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Saturation"
        }
    ]
}*/

vec3 hue2rgb(float h) {
    // Convert hue (0–1) to RGB — constrained to one hue
    float r = abs(h * 6.0 - 3.0) - 1.0;
    float g = 2.0 - abs(h * 6.0 - 2.0);
    float b = 2.0 - abs(h * 6.0 - 4.0);
    return clamp(vec3(r, g, b), 0.0, 1.0);
}

void main() {
    vec2 uv = isf_FragNormCoord;
    vec3 src = IMG_NORM_PIXEL(inputImage, uv).rgb;
    float luma = dot(src, vec3(0.299, 0.587, 0.114));

    // The neon color
    vec3 neon = hue2rgb(hue);
    // Boost saturation — push toward vivid
    neon = mix(vec3(dot(neon, vec3(0.33))), neon, 1.0 + saturation * 0.5);
    neon = clamp(neon, 0.0, 1.0);

    // Bleed mask — highlights catch the neon, threshold softened by spread
    float threshold = mix(0.7, 0.15, spread);
    float mask = smoothstep(threshold, threshold + 0.25, luma);

    // Slight bleed into darker regions via a softer secondary mask
    float softMask = smoothstep(threshold - 0.2, threshold + 0.4, luma) * 0.3;
    mask = max(mask, softMask);

    // Neon tint: multiply source brightness by neon color, boosted
    vec3 tinted = luma * neon * (1.4 + saturation * 0.5);

    // Shadows keep source brightness, neon blends in via mask
    vec3 col = mix(src, tinted, mask);

    gl_FragColor = vec4(mix(src, col, intensity), 1.0);
}
