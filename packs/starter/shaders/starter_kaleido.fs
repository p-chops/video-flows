/*{
    "DESCRIPTION": "Kaleidoscope — polar mirror symmetry with rotation",
    "CREDIT": "Starter Pack",
    "ISFVSN": "2",
    "CATEGORIES": ["Stylize", "Geometry"],
    "INPUTS": [
        {
            "NAME": "inputImage",
            "TYPE": "image"
        },
        {
            "NAME": "segments",
            "TYPE": "float",
            "DEFAULT": 6.0,
            "MIN": 2.0,
            "MAX": 16.0
        },
        {
            "NAME": "spin",
            "TYPE": "float",
            "DEFAULT": 0.2,
            "MIN": 0.0,
            "MAX": 1.0
        },
        {
            "NAME": "zoom",
            "TYPE": "float",
            "DEFAULT": 1.0,
            "MIN": 0.5,
            "MAX": 3.0
        }
    ]
}*/

#define PI 3.14159265359
#define TAU 6.28318530718

void main() {
    vec2 uv = isf_FragNormCoord;
    vec2 center = uv - 0.5;

    // Aspect correction
    float aspect = RENDERSIZE.x / RENDERSIZE.y;
    center.x *= aspect;

    // Polar coordinates
    float r = length(center) * zoom;
    float a = atan(center.y, center.x) + TIME * spin;

    // Mirror-fold into segment wedge
    float seg = TAU / floor(segments);
    a = mod(a, seg);
    if (a > seg * 0.5) a = seg - a; // mirror within wedge

    // Back to cartesian — sample from source
    vec2 sample_uv = vec2(cos(a), sin(a)) * r;
    sample_uv.x /= aspect;
    sample_uv += 0.5;

    gl_FragColor = IMG_NORM_PIXEL(inputImage, sample_uv);
}
