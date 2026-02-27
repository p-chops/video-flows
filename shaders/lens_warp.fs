/*{
    "DESCRIPTION": "Barrel and pincushion lens distortion — simulates wide-angle or telephoto lens curvature. Positive distortion bulges outward (barrel/fisheye), negative pinches inward (pincushion). Animated breathing optional.",
    "CREDIT": "P-Chops / Undersea Lair Project",
    "ISFVSN": "2",
    "CATEGORIES": ["Distort"],
    "INPUTS": [
        {
            "NAME": "inputImage",
            "TYPE": "image"
        },
        {
            "NAME": "intensity",
            "TYPE": "float",
            "LABEL": "Intensity",
            "DEFAULT": 0.65,
            "MIN": 0.0,
            "MAX": 1.0
        },
        {
            "NAME": "distortion",
            "TYPE": "float",
            "LABEL": "Distortion",
            "DEFAULT": 0.8,
            "MIN": -2.0,
            "MAX": 2.0
        },
        {
            "NAME": "zoom",
            "TYPE": "float",
            "LABEL": "Zoom Compensation",
            "DEFAULT": 0.5,
            "MIN": 0.0,
            "MAX": 1.0
        },
        {
            "NAME": "breathe",
            "TYPE": "float",
            "LABEL": "Breathe Animation",
            "DEFAULT": 0.3,
            "MIN": 0.0,
            "MAX": 1.0
        }
    ]
}*/

void main() {
    vec2 uv = isf_FragNormCoord;
    float aspect = RENDERSIZE.x / RENDERSIZE.y;

    // Centre and correct for aspect
    vec2 centered = uv - 0.5;
    centered.x *= aspect;

    float r = length(centered);
    float r2 = r * r;

    // Animate distortion amount
    float d = distortion * intensity;
    d += breathe * 0.4 * sin(TIME * 0.35) * intensity;

    // Brown-Conrady radial distortion model
    // r' = r * (1 + k1*r^2 + k2*r^4)
    float k1 = d;
    float k2 = d * 0.3;
    float distort_factor = 1.0 + k1 * r2 + k2 * r2 * r2;

    vec2 distorted = centered * distort_factor;

    // Zoom compensation — pull back so edges stay in frame
    float z = 1.0 + zoom * abs(d) * 0.5;
    distorted /= z;

    // Undo aspect and re-centre
    distorted.x /= aspect;
    distorted += 0.5;

    // Clamp with edge fade
    vec2 clamped = clamp(distorted, 0.0, 1.0);
    vec4 col = IMG_NORM_PIXEL(inputImage, clamped);

    // Darken pixels that went out of bounds (vignette at extremes)
    float oob = step(0.0, distorted.x) * step(distorted.x, 1.0)
              * step(0.0, distorted.y) * step(distorted.y, 1.0);
    col.rgb *= mix(0.3, 1.0, oob);

    gl_FragColor = col;
}
