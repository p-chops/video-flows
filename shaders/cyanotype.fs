/*{
    "DESCRIPTION": "Cyanotype / blueprint process. Deep Prussian blue shadows, paper-white highlights with optional warm cream tint. Classic photographic alternative process look.",
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
            "DEFAULT": 0.80,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Intensity"
        },
        {
            "NAME": "depth",
            "TYPE": "float",
            "DEFAULT": 0.60,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Blue Depth"
        },
        {
            "NAME": "warmth",
            "TYPE": "float",
            "DEFAULT": 0.20,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Highlight Warmth"
        }
    ]
}*/

void main() {
    vec2 uv = isf_FragNormCoord;
    vec3 src = IMG_NORM_PIXEL(inputImage, uv).rgb;
    float luma = dot(src, vec3(0.299, 0.587, 0.114));

    // Prussian blue for shadows — lifted floor so stacking doesn't crush to black
    vec3 shadow = vec3(0.1, 0.15, 0.35) * (0.6 + depth * 0.4);
    // Mid-blue — bright enough to read through stacks
    vec3 mid = vec3(0.2, 0.38, 0.65) * (0.6 + depth * 0.4);
    // Highlight — white with optional warm cream
    vec3 highlight = mix(vec3(0.92, 0.94, 0.96),
                         vec3(0.96, 0.92, 0.84), warmth);

    // Three-zone blend — wider midtone range
    vec3 col;
    if (luma < 0.3) {
        col = mix(shadow, mid, luma / 0.3);
    } else if (luma < 0.65) {
        col = mix(mid, highlight, (luma - 0.3) / 0.35);
    } else {
        col = mix(highlight, vec3(1.0), (luma - 0.65) / 0.35);
    }

    gl_FragColor = vec4(mix(src, col, intensity), 1.0);
}
