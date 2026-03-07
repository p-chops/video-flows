/*{
    "DESCRIPTION": "Temporal feedback loop with zoom and rotation. Composites the current frame over a transformed version of the previous output. Dark regions of the current frame become transparent, revealing spiraling echoes of past frames.",
    "CREDIT": "Undersea Lair Project",
    "ISFVSN": "2",
    "CATEGORIES": ["Stylize"],
    "INPUTS": [
        {
            "NAME": "inputImage",
            "TYPE": "image"
        },
        {
            "NAME": "feedback_mix",
            "TYPE": "float",
            "DEFAULT": 0.85,
            "MIN": 0.0,
            "MAX": 0.98
        },
        {
            "NAME": "zoom_amount",
            "TYPE": "float",
            "DEFAULT": 0.03,
            "MIN": -0.1,
            "MAX": 0.1
        },
        {
            "NAME": "rotate_amount",
            "TYPE": "float",
            "DEFAULT": 0.01,
            "MIN": -0.05,
            "MAX": 0.05
        },
        {
            "NAME": "dark_threshold",
            "TYPE": "float",
            "DEFAULT": 0.2,
            "MIN": 0.0,
            "MAX": 0.6
        },
        {
            "NAME": "color_drift",
            "TYPE": "float",
            "DEFAULT": 0.005,
            "MIN": 0.0,
            "MAX": 0.03
        }
    ],
    "PASSES": [
        {
            "TARGET": "feedbackBuffer",
            "PERSISTENT": true
        }
    ]
}*/

float luminance(vec3 c) {
    return dot(c, vec3(0.299, 0.587, 0.114));
}

vec3 rgb2hsv(vec3 c) {
    vec4 K = vec4(0.0, -1.0/3.0, 2.0/3.0, -1.0);
    vec4 p = c.g < c.b ? vec4(c.bg, K.wz) : vec4(c.gb, K.xy);
    vec4 q = c.r < p.x ? vec4(p.xyw, c.r) : vec4(c.r, p.yzx);
    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0*d + e)), d / (q.x + e), q.x);
}

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
    vec2 uv = isf_FragNormCoord;
    vec4 src = IMG_NORM_PIXEL(inputImage, uv);

    // Transform UV for feedback: zoom + rotate around center
    vec2 center = vec2(0.5);
    vec2 p = uv - center;

    // Zoom: positive = zoom in (feedback shrinks), negative = zoom out
    float z = 1.0 + zoom_amount;
    p /= z;

    // Rotate
    float ca = cos(rotate_amount);
    float sa = sin(rotate_amount);
    p = vec2(p.x * ca - p.y * sa, p.x * sa + p.y * ca);

    p += center;

    // Sample previous frame (transformed)
    vec3 prev = vec3(0.0);
    if (p.x >= 0.0 && p.x <= 1.0 && p.y >= 0.0 && p.y <= 1.0) {
        prev = IMG_NORM_PIXEL(feedbackBuffer, p).rgb;
    }

    // Subtle hue drift on feedback to create color trails
    if (color_drift > 0.0) {
        vec3 hsv = rgb2hsv(prev);
        hsv.x = fract(hsv.x + color_drift);
        prev = hsv2rgb(hsv);
    }

    // Fade the feedback
    prev *= feedback_mix;

    // Current frame transparency based on darkness
    float luma = luminance(src.rgb);
    float opacity = smoothstep(dark_threshold * 0.5, dark_threshold, luma);

    // Composite: current frame over faded+transformed feedback
    vec3 result = mix(prev, src.rgb, opacity);

    gl_FragColor = vec4(result, 1.0);
}
