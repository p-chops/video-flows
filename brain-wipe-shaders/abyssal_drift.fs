/*{
  "DESCRIPTION": "Domain-warped FBM drifting vertically bottom-to-top. Submarine window view during descent. Abyssal palette: black-blue through bioluminescent teal.",
  "CREDIT": "P-Chops / Undersea Lair Project",
  "ISFVSN": "2",
  "CATEGORIES": ["Generator", "Ambient"],
  "INPUTS": [
    { "NAME": "descentSpeed",  "TYPE": "float", "DEFAULT": 0.3,  "MIN": 0.0, "MAX": 2.0 },
    { "NAME": "warpEvolution", "TYPE": "float", "DEFAULT": 0.5,  "MIN": 0.0, "MAX": 4.0 },
    { "NAME": "warpStrength",  "TYPE": "float", "DEFAULT": 4.0,  "MIN": 0.0, "MAX": 8.0 }
  ]
}
*/

vec2 hash2(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)),
             dot(p, vec2(269.5, 183.3)));
    return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
}

float snoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(dot(hash2(i + vec2(0.0, 0.0)), f - vec2(0.0, 0.0)),
                   dot(hash2(i + vec2(1.0, 0.0)), f - vec2(1.0, 0.0)), u.x),
               mix(dot(hash2(i + vec2(0.0, 1.0)), f - vec2(0.0, 1.0)),
                   dot(hash2(i + vec2(1.0, 1.0)), f - vec2(1.0, 1.0)), u.x), u.y);
}

float fbm(vec2 p) {
    float v    = 0.0;
    float amp  = 0.5;
    float freq = 1.0;
    for (int i = 0; i < 5; i++) {
        v    += amp * snoise(p * freq);
        freq *= 2.1;
        amp  *= 0.48;
    }
    return v;
}

void main() {
    // ISF normalized coords → centered, aspect-corrected
    vec2 uv = isf_FragNormCoord * 2.0 - 1.0;
    uv.x *= RENDERSIZE.x / RENDERSIZE.y;

    // Shift the entire noise landscape downward over time so the pattern
    // scrolls upward on screen — objects outside the window rising as we descend.
    vec2 p = uv - vec2(0.0, TIME * descentSpeed);

    // Circular warp evolution offsets — each layer rotates slowly through noise space
    // at different rates so they evolve independently. Circular motion means zero net
    // directional drift; only the warp shape changes, not where it's going.
    float e1 = TIME * warpEvolution;
    float e2 = TIME * warpEvolution * 0.73; // irrational ratio → layers stay out of sync
    vec2 evo_q = vec2(sin(e1), cos(e1)) * 1.2;
    vec2 evo_r = vec2(sin(e2), cos(e2)) * 0.9;

    // Two-layer domain warp — scroll drives vertical movement, evo drives turbulence shape
    vec2 q = vec2(fbm(p + vec2(0.0, 0.0) + evo_q),
                  fbm(p + vec2(5.2, 1.3)  + evo_q));

    vec2 r = vec2(fbm(p + warpStrength * q + vec2(1.7, 9.2) + evo_r),
                  fbm(p + warpStrength * q + vec2(8.3, 2.8) + evo_r));

    float f = fbm(p + warpStrength * r + evo_r * 0.5);
    f = 0.5 + 0.5 * f; // remap to [0, 1]

    // Deep abyssal palette: black-blue → deep teal → bioluminescent cold glow
    vec3 col = mix(vec3(0.01, 0.02, 0.08),
                   vec3(0.02, 0.15, 0.25),
                   clamp(f * 1.5, 0.0, 1.0));
    col = mix(col, vec3(0.08, 0.38, 0.42), clamp(f * f * 2.0,      0.0, 1.0));
    col = mix(col, vec3(0.15, 0.55, 0.5),  clamp(pow(f, 4.0) * 3.0, 0.0, 1.0));

    // Vignette
    float vign = 1.0 - dot(uv * 0.5, uv * 0.5);
    col *= clamp(vign, 0.0, 1.0);

    gl_FragColor = vec4(col, 1.0);
}
