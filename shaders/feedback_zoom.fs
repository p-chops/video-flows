/*{
    "DESCRIPTION": "Simulated video feedback loop: the image is recursively zoomed and rotated, layering ghostly echoes that spiral inward or outward. Mimics the effect of pointing a camera at its own monitor. Uses multiple zoom-rotate iterations per frame for depth without temporal buffers.",
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
            "NAME": "zoom_rate",
            "TYPE": "float",
            "DEFAULT": 0.50,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Zoom Rate"
        },
        {
            "NAME": "twist",
            "TYPE": "float",
            "DEFAULT": 0.30,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Twist"
        },
        {
            "NAME": "layers",
            "TYPE": "float",
            "DEFAULT": 0.50,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Echo Layers"
        }
    ]
}*/

void main() {
    vec2 uv = isf_FragNormCoord;
    vec3 src = IMG_NORM_PIXEL(inputImage, uv).rgb;

    // Number of recursive echo layers (2–8)
    int nLayers = int(mix(2.0, 8.0, layers));

    // Per-layer zoom and rotation increments
    float zStep = mix(0.02, 0.12, zoom_rate);
    float rStep = mix(0.01, 0.15, twist);

    // Animate the centre of recursion slowly
    vec2 centre = vec2(
        0.5 + 0.03 * sin(TIME * 0.37),
        0.5 + 0.03 * cos(TIME * 0.29)
    );

    vec3 accum = vec3(0.0);
    float totalW = 0.0;

    for (int i = 1; i <= 8; i++) {
        if (i > nLayers) break;

        float fi = float(i);

        // Progressive zoom: each layer zooms in/out more
        float z = 1.0 + zStep * fi * sin(TIME * 0.2 + fi * 0.7);

        // Progressive rotation
        float angle = rStep * fi * sin(TIME * 0.15 + fi * 1.3);
        float ca = cos(angle), sa = sin(angle);

        // Transform UV relative to animated centre
        vec2 p = uv - centre;
        p /= z;
        p = vec2(p.x * ca - p.y * sa, p.x * sa + p.y * ca);
        p += centre;

        // Wrap coordinates
        p = fract(p);

        // Deeper layers fade out
        float w = 1.0 / (1.0 + fi * 0.6);
        accum += IMG_NORM_PIXEL(inputImage, p).rgb * w;
        totalW += w;
    }

    accum /= totalW;

    // Colour shift between layers — offset red and blue slightly
    vec2 p2 = uv - centre;
    float z2 = 1.0 + zStep * 0.5;
    p2 /= z2;
    p2 += centre;
    p2 = fract(p2);
    accum.r = mix(accum.r, IMG_NORM_PIXEL(inputImage, p2).r, 0.15);

    vec3 col = mix(src, accum, intensity);
    gl_FragColor = vec4(col, 1.0);
}
