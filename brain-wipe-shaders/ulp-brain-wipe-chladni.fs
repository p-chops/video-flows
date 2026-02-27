/*{
    "DESCRIPTION": "Chladni resonance figures — animated standing wave nodal patterns from vibrating plate mathematics. Mode morphing reveals the shapes sand makes at different frequencies.",
    "CREDIT": "ULP Brain Wipe Series",
    "ISFVSN": "2",
    "CATEGORIES": ["Brain Wipe", "ULP"],
    "INPUTS": [
        {
            "NAME": "mode_m",
            "TYPE": "float",
            "DEFAULT": 3.0,
            "MIN": 1.0,
            "MAX": 12.0,
            "LABEL": "Mode M"
        },
        {
            "NAME": "mode_n",
            "TYPE": "float",
            "DEFAULT": 4.0,
            "MIN": 1.0,
            "MAX": 12.0,
            "LABEL": "Mode N"
        },
        {
            "NAME": "morph_speed",
            "TYPE": "float",
            "DEFAULT": 0.08,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Morph Speed"
        },
        {
            "NAME": "line_width",
            "TYPE": "float",
            "DEFAULT": 0.07,
            "MIN": 0.005,
            "MAX": 0.4,
            "LABEL": "Line Width"
        },
        {
            "NAME": "line_softness",
            "TYPE": "float",
            "DEFAULT": 0.5,
            "MIN": 0.01,
            "MAX": 1.0,
            "LABEL": "Line Softness"
        },
        {
            "NAME": "scale",
            "TYPE": "float",
            "DEFAULT": 1.0,
            "MIN": 0.25,
            "MAX": 4.0,
            "LABEL": "Scale"
        },
        {
            "NAME": "slow_rotation",
            "TYPE": "float",
            "DEFAULT": 0.0,
            "MIN": -1.0,
            "MAX": 1.0,
            "LABEL": "Slow Rotation"
        },
        {
            "NAME": "color_hue",
            "TYPE": "float",
            "DEFAULT": 0.0,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Line Hue"
        },
        {
            "NAME": "saturation",
            "TYPE": "float",
            "DEFAULT": 0.0,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Saturation"
        },
        {
            "NAME": "bg_brightness",
            "TYPE": "float",
            "DEFAULT": 0.0,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "BG Brightness"
        },
        {
            "NAME": "invert",
            "TYPE": "bool",
            "DEFAULT": false,
            "LABEL": "Invert"
        }
    ]
}*/

#define PI 3.14159265359

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
    // Centered UV in [-1, 1], aspect-corrected, then scaled
    vec2 uv = (isf_FragNormCoord * 2.0 - 1.0) / scale;
    uv.x *= RENDERSIZE.x / RENDERSIZE.y;

    // Slow rotation
    float angle = TIME * slow_rotation * 0.15;
    float ca = cos(angle), sa = sin(angle);
    uv = vec2(ca * uv.x - sa * uv.y, sa * uv.x + ca * uv.y);

    // Animate mode numbers smoothly around their set values
    float t = TIME * morph_speed;
    float m = mode_m + sin(t * 0.71) * 0.45;
    float n = mode_n + cos(t * 0.53) * 0.45;

    // Chladni formula for a free square plate
    // Nodal lines where f(x,y) = 0 — the "sand settles here" locations
    float x = uv.x * PI;
    float y = uv.y * PI;

    float f_diff = cos(m * x) * cos(n * y) - cos(n * x) * cos(m * y);
    float f_sum  = cos(m * x) * cos(n * y) + cos(n * x) * cos(m * y);

    // Slowly blend between the two mode symmetries (odd/even)
    float blend = sin(t * 0.27) * 0.5 + 0.5;
    float f = mix(f_diff, f_sum, blend);

    // Draw as glowing nodal lines (where f ≈ 0)
    float edge = 1.0 - smoothstep(line_width * line_softness, line_width, abs(f));

    if (invert > 0.5) edge = 1.0 - edge;

    float brightness = mix(bg_brightness, 1.0, edge);
    vec3 col = hsv2rgb(vec3(fract(color_hue + edge * 0.05), saturation, brightness));

    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
