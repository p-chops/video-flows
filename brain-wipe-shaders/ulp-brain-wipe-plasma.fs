/*{
    "DESCRIPTION": "Plasma field — overlapping sinusoidal color pools, silky and endlessly morphing. Classic demoscene effect tuned for brain wipe use.",
    "CREDIT": "ULP Brain Wipe Series",
    "ISFVSN": "2",
    "CATEGORIES": ["Brain Wipe", "ULP"],
    "INPUTS": [
        {
            "NAME": "speed",
            "TYPE": "float",
            "DEFAULT": 0.4,
            "MIN": 0.0,
            "MAX": 3.0,
            "LABEL": "Speed"
        },
        {
            "NAME": "scale",
            "TYPE": "float",
            "DEFAULT": 3.0,
            "MIN": 0.5,
            "MAX": 12.0,
            "LABEL": "Scale"
        },
        {
            "NAME": "complexity",
            "TYPE": "float",
            "DEFAULT": 0.5,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Complexity"
        },
        {
            "NAME": "color_hue",
            "TYPE": "float",
            "DEFAULT": 0.0,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Hue"
        },
        {
            "NAME": "color_range",
            "TYPE": "float",
            "DEFAULT": 0.25,
            "MIN": 0.0,
            "MAX": 0.5,
            "LABEL": "Hue Range"
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
            "NAME": "brightness",
            "TYPE": "float",
            "DEFAULT": 1.0,
            "MIN": 0.1,
            "MAX": 2.0,
            "LABEL": "Brightness"
        },
        {
            "NAME": "contrast",
            "TYPE": "float",
            "DEFAULT": 1.2,
            "MIN": 0.5,
            "MAX": 4.0,
            "LABEL": "Contrast"
        },
        {
            "NAME": "invert",
            "TYPE": "bool",
            "DEFAULT": false,
            "LABEL": "Invert"
        }
    ]
}*/

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
    vec2 uv = isf_FragNormCoord;
    uv.x *= RENDERSIZE.x / RENDERSIZE.y;
    uv *= scale;

    float t = TIME * speed;

    // Core plasma: 4 sine layers at different angles and frequencies
    float v = 0.0;
    v += sin(uv.x + t);
    v += sin(uv.y * 0.9 + t * 0.8);
    v += sin((uv.x * 0.6 + uv.y * 0.8) + t * 0.7);

    // Circular ripple from drifting center
    float cx = uv.x + 0.5 * sin(t * 0.31);
    float cy = uv.y + 0.5 * cos(t * 0.41);
    v += sin(sqrt(cx * cx + cy * cy + 0.1) * 2.5 - t * 0.5);

    // Complexity layer — higher harmonics
    v += complexity * (
        sin(uv.x * 1.7 - t * 0.6) * sin(uv.y * 1.3 + t * 0.5) +
        sin(uv.x * 2.3 + uv.y * 1.1 + t * 0.9) * 0.5
    );

    // Normalize to 0..1
    float layers = 4.0 + complexity * 2.5;
    v = v / layers + 0.5;
    v = clamp(v, 0.0, 1.0);

    // Contrast
    v = pow(v, contrast);

    if (invert > 0.5) v = 1.0 - v;

    // Color via HSV: narrow hue range for moody, wide for psychedelic
    float hue = color_hue + v * color_range;
    vec3 col = hsv2rgb(vec3(fract(hue), saturation, v * brightness));

    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
