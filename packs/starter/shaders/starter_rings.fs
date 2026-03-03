/*{
    "DESCRIPTION": "Concentric rings with slow radial pulse and rotation. Simple geometric generator that produces continuously moving circular patterns.",
    "CREDIT": "Undersea Lair Project",
    "ISFVSN": "2",
    "CATEGORIES": ["Generator", "Brain Wipe"],
    "INPUTS": [
        {
            "NAME": "ring_count",
            "TYPE": "float",
            "DEFAULT": 8.0,
            "MIN": 3.0,
            "MAX": 20.0
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
            "NAME": "brightness",
            "TYPE": "float",
            "DEFAULT": 0.8,
            "MIN": 0.3,
            "MAX": 1.5
        }
    ]
}*/

vec3 palette(float t, float shift) {
    vec3 a = vec3(0.5);
    vec3 b = vec3(0.5);
    vec3 c = vec3(1.0, 1.0, 0.5);
    vec3 d = vec3(0.8, 0.9, 0.3) + shift;
    return a + b * cos(6.28318 * (c * t + d));
}

void main() {
    vec2 uv = isf_FragNormCoord;
    float aspect = RENDERSIZE.x / RENDERSIZE.y;
    vec2 p = (uv - 0.5) * vec2(aspect, 1.0);

    float t = TIME * speed;

    // Polar coords
    float r = length(p);
    float a = atan(p.y, p.x);

    // Concentric rings with radial pulse
    float pulse = sin(t * 0.7) * 0.3;
    float rings = sin((r + pulse) * ring_count * 3.14159 - t * 2.0);

    // Slight angular modulation for movement
    rings += 0.3 * sin(a * 3.0 + t * 1.5);

    // Remap to 0-1 range
    float f = rings * 0.5 + 0.5;

    // Add radial gradient for depth
    f = mix(f, f * (1.0 - r * 0.6), 0.5);

    vec3 col = palette(f, palette_shift) * brightness;

    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
