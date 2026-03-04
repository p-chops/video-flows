/*{
    "DESCRIPTION": "Infinite tunnel — raymarched concentric geometry with color banding",
    "CREDIT": "Starter Pack",
    "ISFVSN": "2",
    "CATEGORIES": ["Generator", "Brain Wipe"],
    "INPUTS": [
        {
            "NAME": "speed",
            "TYPE": "float",
            "DEFAULT": 0.6,
            "MIN": 0.05,
            "MAX": 2.0
        },
        {
            "NAME": "twist",
            "TYPE": "float",
            "DEFAULT": 0.3,
            "MIN": 0.0,
            "MAX": 2.0
        },
        {
            "NAME": "shape",
            "TYPE": "float",
            "DEFAULT": 0.5,
            "MIN": 0.0,
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
            "DEFAULT": 0.85,
            "MIN": 0.3,
            "MAX": 1.5
        }
    ]
}*/

#define PI 3.14159265359
#define TAU 6.28318530718

vec3 palette(float t, vec3 a, vec3 b, vec3 c, vec3 d) {
    return a + b * cos(TAU * (c * t + d));
}

void main() {
    vec2 uv = isf_FragNormCoord;
    float aspect = RENDERSIZE.x / RENDERSIZE.y;
    vec2 p = (uv - 0.5) * vec2(aspect, 1.0);
    float t = TIME * speed;

    // Polar coordinates
    float r = length(p);
    float a = atan(p.y, p.x);

    // Infinite tunnel: 1/r maps radius to depth
    float depth = 0.5 / (r + 0.001);

    // Twist increases with depth
    a += depth * twist + t * 0.2;

    // Shape blending: round (angle only) → polygonal (folded angle)
    float sides = 6.0;
    float seg = TAU / sides;
    float folded = mod(a, seg) - seg * 0.5;
    float polygon_r = cos(seg * 0.5) / cos(folded);
    // Blend between circular and polygonal tunnel cross-section
    float shaped_depth = 0.5 / (mix(r, r / polygon_r, shape) + 0.001);

    // Tile pattern along the tunnel
    float ring = fract(shaped_depth - t); // scrolling depth bands
    float stripe = fract(a / seg);        // angular sectors

    // Color from depth banding
    float band = floor(shaped_depth - t);
    float color_idx = fract(band * 0.127 + palette_shift);

    vec3 color = palette(
        color_idx,
        vec3(0.5, 0.5, 0.5),
        vec3(0.5, 0.5, 0.5),
        vec3(1.0, 0.7, 0.4),
        vec3(0.0, 0.15, 0.20)
    );

    // Edge darkening between rings and sectors
    float ring_edge = smoothstep(0.0, 0.08, ring) * smoothstep(1.0, 0.92, ring);
    float sector_edge = smoothstep(0.0, 0.06, stripe) * smoothstep(1.0, 0.94, stripe);
    float edge = ring_edge * sector_edge;

    // Depth fog — fade distant geometry
    float fog = 1.0 - smoothstep(2.0, 15.0, depth);

    color *= edge * fog * brightness;
    gl_FragColor = vec4(color, 1.0);
}
