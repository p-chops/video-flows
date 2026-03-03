/*{
    "DESCRIPTION": "Classic plasma — layered sine waves with cosine palette",
    "CREDIT": "Starter Pack",
    "ISFVSN": "2",
    "CATEGORIES": ["Generator", "Brain Wipe"],
    "INPUTS": [
        {
            "NAME": "scale",
            "TYPE": "float",
            "DEFAULT": 3.0,
            "MIN": 1.0,
            "MAX": 10.0
        },
        {
            "NAME": "speed",
            "TYPE": "float",
            "DEFAULT": 0.4,
            "MIN": 0.05,
            "MAX": 1.5
        },
        {
            "NAME": "palette_shift",
            "TYPE": "float",
            "DEFAULT": 0.0,
            "MIN": 0.0,
            "MAX": 1.0
        },
        {
            "NAME": "warp",
            "TYPE": "float",
            "DEFAULT": 0.5,
            "MIN": 0.0,
            "MAX": 2.0
        },
        {
            "NAME": "brightness",
            "TYPE": "float",
            "DEFAULT": 0.8,
            "MIN": 0.3,
            "MAX": 1.5
        }
    ]
}*/

#define PI 3.14159265359
#define TAU 6.28318530718

// Cosine palette — IQ's classic formulation
vec3 palette(float t, vec3 a, vec3 b, vec3 c, vec3 d) {
    return a + b * cos(TAU * (c * t + d));
}

void main() {
    vec2 uv = isf_FragNormCoord;
    float aspect = RENDERSIZE.x / RENDERSIZE.y;
    vec2 p = (uv - 0.5) * vec2(aspect, 1.0) * scale;
    float t = TIME * speed;

    // Four overlapping plasma layers
    float v = 0.0;
    v += sin(p.x + t);
    v += sin(p.y + t * 0.7);
    v += sin((p.x + p.y + t * 0.5) * 0.7);
    float cx = p.x + warp * sin(t * 0.3);
    float cy = p.y + warp * cos(t * 0.4);
    v += sin(sqrt(cx * cx + cy * cy + 1.0) + t);
    v *= 0.25; // normalize to roughly [-1, 1]

    // Domain warp — second pass displaces by first pass
    vec2 p2 = p + vec2(sin(v * PI + t * 0.2), cos(v * PI + t * 0.3)) * warp;
    float v2 = 0.0;
    v2 += sin(p2.x * 1.3 + t * 0.6);
    v2 += sin(p2.y * 1.1 - t * 0.5);
    v2 += sin(length(p2) * 0.8 + t);
    v2 *= 0.333;

    float combined = (v + v2) * 0.5 + 0.5; // map to [0, 1]

    // Cosine palette with shift
    vec3 color = palette(
        combined + palette_shift,
        vec3(0.5, 0.5, 0.5),
        vec3(0.5, 0.5, 0.5),
        vec3(1.0, 1.0, 1.0),
        vec3(0.0, 0.33, 0.67)
    );

    color *= brightness;
    gl_FragColor = vec4(color, 1.0);
}
