/*{
    "DESCRIPTION": "Animated interlocking sine waves that produce a woven textile pattern. Scrolls diagonally with color cycling.",
    "CREDIT": "Undersea Lair Project",
    "ISFVSN": "2",
    "CATEGORIES": ["Generator", "Brain Wipe"],
    "INPUTS": [
        {
            "NAME": "scale",
            "TYPE": "float",
            "DEFAULT": 6.0,
            "MIN": 2.0,
            "MAX": 15.0
        },
        {
            "NAME": "speed",
            "TYPE": "float",
            "DEFAULT": 0.3,
            "MIN": 0.05,
            "MAX": 1.0
        },
        {
            "NAME": "color_cycle",
            "TYPE": "float",
            "DEFAULT": 0.5,
            "MIN": 0.0,
            "MAX": 1.0
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

void main() {
    vec2 uv = isf_FragNormCoord;
    float t = TIME * speed;

    vec2 p = uv * scale;

    // Two crossing wave sets
    float wave1 = sin(p.x + sin(p.y * 0.7 + t) * 1.5 + t * 0.8);
    float wave2 = sin(p.y + sin(p.x * 0.7 - t * 0.6) * 1.5 - t * 0.5);

    // Interference pattern
    float pattern = wave1 * wave2;

    // Add diagonal scroll
    float scroll = sin(p.x + p.y + t * 1.2) * 0.3;
    pattern += scroll;

    // Normalize
    pattern = pattern * 0.35 + 0.5;

    // Color from pattern value
    float hue = pattern * 0.4 + color_cycle + t * 0.05;
    float sat = 0.6 + pattern * 0.3;
    float val = 0.4 + pattern * 0.5;

    // HSV to RGB
    vec3 c = vec3(hue, sat, val);
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 px = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    vec3 col = c.z * mix(K.xxx, clamp(px - K.xxx, 0.0, 1.0), c.y);

    col *= brightness;

    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
