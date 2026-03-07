/*{
  "DESCRIPTION": "ULP Infernal Drift — domain-warped FBM rising through fire. Counterpart to abyssal_drift. Staring into an infinite furnace: turbulent flame structures, floating embers, heat shimmer. Lair frame: the boiler room beneath the lair.",
  "CREDIT": "Undersea Lair Project / P-Chops",
  "ISFVSN": "2",
  "CATEGORIES": ["Generator", "Ambient", "ULP"],
  "INPUTS": [
    {
      "NAME": "burn_rate",
      "TYPE": "float",
      "DEFAULT": 0.4,
      "MIN": 0.0,
      "MAX": 1.0,
      "LABEL": "Burn Rate  [flame speed]"
    },
    {
      "NAME": "turbulence",
      "TYPE": "float",
      "DEFAULT": 0.5,
      "MIN": 0.0,
      "MAX": 1.0,
      "LABEL": "Turbulence  [chaos intensity]"
    },
    {
      "NAME": "heat",
      "TYPE": "float",
      "DEFAULT": 0.45,
      "MIN": 0.0,
      "MAX": 1.0,
      "LABEL": "Heat  [brightness / white-hot intensity]"
    },
    {
      "NAME": "ember_density",
      "TYPE": "float",
      "DEFAULT": 0.4,
      "MIN": 0.0,
      "MAX": 1.0,
      "LABEL": "Ember Density  [floating sparks]"
    },
    {
      "NAME": "flame_scale",
      "TYPE": "float",
      "DEFAULT": 0.5,
      "MIN": 0.0,
      "MAX": 1.0,
      "LABEL": "Flame Scale  [structure size]"
    },
    {
      "NAME": "color_shift",
      "TYPE": "float",
      "DEFAULT": 0.0,
      "MIN": 0.0,
      "MAX": 1.0,
      "LABEL": "Color Shift  [0=natural fire, 1=blue chemical flame]"
    },
    {
      "NAME": "burst",
      "TYPE": "float",
      "DEFAULT": 0.0,
      "MIN": 0.0,
      "MAX": 1.0,
      "LABEL": "Burst  [map → audio peak]"
    }
  ]
}*/

// ─── Noise ──────────────────────────────────────────────────────────────────

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
    float v = 0.0;
    float amp = 0.5;
    float freq = 1.0;
    for (int i = 0; i < 6; i++) {
        v += amp * snoise(p * freq);
        freq *= 2.1;
        amp *= 0.48;
    }
    return v;
}

// ─── Main ───────────────────────────────────────────────────────────────────

void main() {
    vec2 uv = isf_FragNormCoord * 2.0 - 1.0;
    uv.x *= RENDERSIZE.x / RENDERSIZE.y;

    float scale = mix(1.5, 4.0, flame_scale);
    float speed = mix(0.15, 0.55, burn_rate);
    float warpStr = mix(2.5, 7.0, turbulence);

    // Fire rises — scroll upward (positive Y over time)
    vec2 pos = uv * scale + vec2(0.0, TIME * speed);

    // Domain warp evolution — circular orbits avoid net drift
    float e1 = TIME * mix(0.3, 0.9, turbulence);
    float e2 = TIME * mix(0.3, 0.9, turbulence) * 0.73;
    vec2 evo_q = vec2(sin(e1), cos(e1)) * 1.5;
    vec2 evo_r = vec2(sin(e2), cos(e2)) * 1.1;

    // Two-layer domain warp — more aggressive than abyssal
    vec2 q = vec2(fbm(pos + vec2(0.0, 0.0) + evo_q),
                  fbm(pos + vec2(5.2, 1.3) + evo_q));

    vec2 r = vec2(fbm(pos + warpStr * q + vec2(1.7, 9.2) + evo_r),
                  fbm(pos + warpStr * q + vec2(8.3, 2.8) + evo_r));

    float f = fbm(pos + warpStr * r + evo_r * 0.5);
    f = 0.5 + 0.5 * f;

    // Heat controls the brightness curve — low heat = mostly dark embers,
    // high heat = more white-hot regions
    f = pow(f, mix(1.6, 0.4, heat));

    // Contrast stretch — push darks darker, lights lighter
    f = smoothstep(0.03, 0.82, f);

    // ─── Fire Palette ───────────────────────────────────────────────────────
    // Natural fire: black → deep red → orange → amber → white-hot
    vec3 c0 = vec3(0.02, 0.0, 0.0);
    vec3 c1 = vec3(0.45, 0.03, 0.0);
    vec3 c2 = vec3(0.85, 0.25, 0.01);
    vec3 c3 = vec3(1.0, 0.55, 0.05);
    vec3 c4 = vec3(1.0, 0.9, 0.6);

    // Blue chemical flame: black → deep indigo → violet → cyan → pale blue
    vec3 b0 = vec3(0.0, 0.0, 0.02);
    vec3 b1 = vec3(0.05, 0.0, 0.4);
    vec3 b2 = vec3(0.15, 0.08, 0.75);
    vec3 b3 = vec3(0.25, 0.45, 1.0);
    vec3 b4 = vec3(0.7, 0.9, 1.0);

    // Interpolate palettes
    vec3 p0 = mix(c0, b0, color_shift);
    vec3 p1 = mix(c1, b1, color_shift);
    vec3 p2 = mix(c2, b2, color_shift);
    vec3 p3 = mix(c3, b3, color_shift);
    vec3 p4 = mix(c4, b4, color_shift);

    // 4-stop gradient ramp
    vec3 col;
    if (f < 0.25) {
        col = mix(p0, p1, f / 0.25);
    } else if (f < 0.5) {
        col = mix(p1, p2, (f - 0.25) / 0.25);
    } else if (f < 0.75) {
        col = mix(p2, p3, (f - 0.5) / 0.25);
    } else {
        col = mix(p3, p4, (f - 0.75) / 0.25);
    }

    // ─── Embers ─────────────────────────────────────────────────────────────
    // Rising sparks — like marine snow but upward and hot
    vec2 emberUV = uv * 30.0 + vec2(TIME * 0.03, TIME * 0.1);
    vec2 emberCell = floor(emberUV);
    vec2 emberFrac = fract(emberUV);

    // Two independent hash rolls combined for smooth probability
    float roll1 = fract(sin(dot(emberCell, vec2(127.1, 311.7))) * 43758.5453);
    float roll2 = fract(sin(dot(emberCell, vec2(269.5, 183.3))) * 43758.5453);
    float emberRoll = fract(roll1 + roll2);

    float eDensityThresh = 1.0 - ember_density * 0.12;
    if (emberRoll > eDensityThresh) {
        vec2 emberPos = vec2(
            fract(sin(dot(emberCell * 13.3, vec2(127.1, 311.7))) * 43758.5453),
            fract(sin(dot(emberCell * 13.3, vec2(269.5, 183.3))) * 43758.5453)
        ) * 0.7 + 0.15;
        float emberDist = length(emberFrac - emberPos);
        float emberDot = exp(-emberDist * emberDist / 0.0015) * 0.1;

        // Flicker
        float flicker = 0.4 + 0.6 * sin(TIME * 3.5 + roll1 * 100.0);
        emberDot *= flicker;

        // Ember color: orange-yellow for natural, blue-white for shifted
        vec3 emberCol = mix(
            mix(vec3(1.0, 0.4, 0.02), vec3(1.0, 0.75, 0.15), flicker),
            mix(vec3(0.3, 0.35, 1.0), vec3(0.6, 0.8, 1.0), flicker),
            color_shift
        );
        col += emberCol * emberDot;
    }

    // ─── Heat Shimmer ───────────────────────────────────────────────────────
    float shimmer = snoise(uv * 10.0 + vec2(0.0, TIME * 2.5)) * 0.03 * turbulence;
    col *= 1.0 + shimmer;

    // ─── Vignette ───────────────────────────────────────────────────────────
    float vign = 1.0 - dot(uv * 0.4, uv * 0.4);
    col *= clamp(vign, 0.0, 1.0);

    // ─── Burst ──────────────────────────────────────────────────────────────
    vec3 burstCol = mix(vec3(0.2, 0.08, 0.0), vec3(0.05, 0.08, 0.2), color_shift);
    col += burstCol * burst;

    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
