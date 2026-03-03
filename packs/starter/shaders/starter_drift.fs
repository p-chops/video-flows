/*{
    "DESCRIPTION": "Animated FBM noise flow with cosine palette. Domain-warped fractal noise produces continuously evolving abstract patterns.",
    "CREDIT": "Undersea Lair Project",
    "ISFVSN": "2",
    "CATEGORIES": ["Generator", "Brain Wipe"],
    "INPUTS": [
        {
            "NAME": "flow_speed",
            "TYPE": "float",
            "DEFAULT": 0.3,
            "MIN": 0.05,
            "MAX": 1.0
        },
        {
            "NAME": "complexity",
            "TYPE": "float",
            "DEFAULT": 0.5,
            "MIN": 0.1,
            "MAX": 1.0
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

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);

    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));

    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

float fbm(vec2 p, int octaves) {
    float value = 0.0;
    float amplitude = 0.5;
    // Per-octave rotation to break axis alignment
    mat2 rot = mat2(0.8, 0.6, -0.6, 0.8);
    for (int i = 0; i < 6; i++) {
        if (i >= octaves) break;
        value += amplitude * noise(p);
        p = rot * p * 2.0;
        amplitude *= 0.5;
    }
    return value;
}

vec3 palette(float t, float shift) {
    vec3 a = vec3(0.5);
    vec3 b = vec3(0.5);
    vec3 c = vec3(1.0);
    vec3 d = vec3(0.0, 0.33, 0.67) + shift;
    return a + b * cos(6.28318 * (c * t + d));
}

void main() {
    vec2 uv = isf_FragNormCoord;
    float t = TIME * flow_speed;

    // Scale UV to get interesting detail
    vec2 p = uv * 3.0;

    int octaves = int(3.0 + complexity * 3.0);

    // Domain warp: one FBM offsets another
    vec2 q = vec2(
        fbm(p + vec2(1.7, 9.2) + t * 0.3, octaves),
        fbm(p + vec2(8.3, 2.8) + t * 0.2, octaves)
    );

    float f = fbm(p + 4.0 * q, octaves);

    // Add secondary warp for richer structure
    vec2 r = vec2(
        fbm(p + 4.0 * q + vec2(1.0, 3.7) + t * 0.15, octaves),
        fbm(p + 4.0 * q + vec2(6.2, 1.3) + t * 0.12, octaves)
    );
    f = fbm(p + 4.0 * r, octaves);

    // Map to color via cosine palette
    vec3 col = palette(f, palette_shift);

    // Scale brightness to target dynamic range (mean 60-140 / 255)
    col *= brightness;

    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
