/*{
  "DESCRIPTION": "ULP Curl Flow — color advected through a curl noise velocity field. Produces flowing paint-like streams, filaments, and turbulence. Material moves and stretches rather than just undulating.",
  "CREDIT": "Undersea Lair Project / P-Chops",
  "ISFVSN": "2",
  "CATEGORIES": ["Generator", "Ambient", "ULP"],
  "INPUTS": [
    {
      "NAME": "flow_speed",
      "TYPE": "float",
      "DEFAULT": 0.4,
      "MIN": 0.0,
      "MAX": 1.0,
      "LABEL": "Flow Speed  [advection rate]"
    },
    {
      "NAME": "turbulence",
      "TYPE": "float",
      "DEFAULT": 0.5,
      "MIN": 0.0,
      "MAX": 1.0,
      "LABEL": "Turbulence  [flow complexity]"
    },
    {
      "NAME": "trail_length",
      "TYPE": "float",
      "DEFAULT": 0.5,
      "MIN": 0.0,
      "MAX": 1.0,
      "LABEL": "Trail Length  [stream persistence]"
    },
    {
      "NAME": "color_richness",
      "TYPE": "float",
      "DEFAULT": 0.5,
      "MIN": 0.0,
      "MAX": 1.0,
      "LABEL": "Color Richness  [palette variation]"
    },
    {
      "NAME": "vortex_scale",
      "TYPE": "float",
      "DEFAULT": 0.4,
      "MIN": 0.0,
      "MAX": 1.0,
      "LABEL": "Vortex Scale  [curl structure size]"
    },
    {
      "NAME": "brightness",
      "TYPE": "float",
      "DEFAULT": 1.0,
      "MIN": 0.0,
      "MAX": 2.0,
      "LABEL": "Brightness"
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

float fbm(vec2 p, int octaves) {
    float v = 0.0;
    float amp = 0.5;
    float freq = 1.0;
    for (int i = 0; i < 6; i++) {
        if (i >= octaves) break;
        v += amp * snoise(p * freq);
        freq *= 2.1;
        amp *= 0.48;
    }
    return v;
}

// ─── Curl noise — divergence-free 2D velocity from noise potential ──────────

vec2 curlNoise(vec2 p, float t, int octaves) {
    float eps = 0.01;
    // Potential field with time evolution
    vec2 tp = p + vec2(t * 0.1, t * 0.07);
    float n  = fbm(tp, octaves);
    float nx = fbm(tp + vec2(eps, 0.0), octaves);
    float ny = fbm(tp + vec2(0.0, eps), octaves);
    // Curl: rotate gradient 90 degrees
    return vec2(ny - n, -(nx - n)) / eps;
}

// ─── Palette ────────────────────────────────────────────────────────────────

vec3 flowPalette(float t, float richness) {
    // Base: medium purple-blue (bright enough for good dynamic range)
    vec3 a = vec3(0.35, 0.25, 0.45);
    // Amplitude scales with richness
    vec3 b = mix(vec3(0.3), vec3(0.5, 0.45, 0.5), richness);
    vec3 c = mix(vec3(1.0), vec3(1.0, 1.5, 2.0), richness);
    vec3 d = vec3(0.0, 0.15, 0.3);
    return a + b * cos(6.2831 * (c * t + d));
}

// ─── Main ───────────────────────────────────────────────────────────────────

void main() {
    vec2 uv = isf_FragNormCoord * 2.0 - 1.0;
    uv.x *= RENDERSIZE.x / RENDERSIZE.y;

    float speed = mix(0.2, 1.0, flow_speed);
    float t = TIME * speed;
    float scale = mix(1.5, 4.0, vortex_scale);
    int octaves = int(mix(3.0, 6.0, turbulence));

    // Advect the sample point backward through the flow field
    // Multiple steps for longer trails
    int steps = int(mix(3.0, 12.0, trail_length));
    float dt = 0.04;

    vec2 pos = uv * scale;
    float accum = 0.0;
    vec3 colAccum = vec3(0.0);

    for (int i = 0; i < 12; i++) {
        if (i >= steps) break;

        // Sample curl noise at current position
        vec2 vel = curlNoise(pos, t - float(i) * dt * 2.0, octaves);
        vel *= mix(0.5, 2.0, turbulence);

        // Advect backward
        pos -= vel * dt;

        // Color from position — different spatial frequencies
        float h1 = fbm(pos * 0.5 + vec2(t * 0.05, 0.0), 3);
        float h2 = fbm(pos * 0.3 + vec2(0.0, t * 0.03), 3);
        float hue = 0.5 + 0.5 * h1 + 0.3 * h2;

        vec3 c = flowPalette(hue, color_richness);

        // Weight earlier advection steps less
        float w = 1.0 / float(i + 1);
        colAccum += c * w;
        accum += w;
    }

    vec3 col = colAccum / accum;

    // Flow-aligned detail streaks at multiple scales
    vec2 flowDir = curlNoise(uv * scale, t, octaves);
    vec2 flowNorm = normalize(flowDir + 0.001);
    float streak1 = snoise(uv * scale * 6.0 + flowNorm * t * 0.5);
    float streak2 = snoise(uv * scale * 15.0 + flowNorm * t * 0.3);
    // Sharp streaks via abs — creates filament-like detail
    float filament = 1.0 - abs(streak1);
    filament *= filament;  // sharpen
    col = mix(col, col * 1.6, filament * 0.4);
    // Fine grain — adds texture complexity
    col *= 0.85 + 0.3 * (0.5 + 0.5 * streak2);

    // Velocity magnitude drives local brightness — fast regions glow
    float speed2 = length(flowDir);
    col *= 0.7 + 0.6 * smoothstep(0.0, 3.0, speed2);

    col *= brightness;

    // Vignette
    float vign = 1.0 - dot(uv * 0.35, uv * 0.35);
    col *= clamp(vign, 0.0, 1.0);

    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
