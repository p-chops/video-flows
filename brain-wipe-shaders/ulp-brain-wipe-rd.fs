/*{
  "DESCRIPTION": "ULP Brain Wipe — Gray-Scott reaction-diffusion. Self-organizing chemical pattern formation as a model of consciousness being overwritten. Two-pass: persistent RD simulation + colorization. Brain wipe frame, not lair frame.",
  "CREDIT": "Undersea Lair Project / P-Chops",
  "ISFVSN": "2",
  "CATEGORIES": ["Reaction-Diffusion", "ULP", "Brain Wipe"],
  "INPUTS": [
    {
      "NAME": "feed",
      "TYPE": "float",
      "DEFAULT": 0.037,
      "MIN": 0.01,
      "MAX": 0.095,
      "LABEL": "Feed Rate  [↑ = more structure]"
    },
    {
      "NAME": "kill",
      "TYPE": "float",
      "DEFAULT": 0.060,
      "MIN": 0.040,
      "MAX": 0.075,
      "LABEL": "Kill Rate  [↑ = more decay]"
    },
    {
      "NAME": "diff_a",
      "TYPE": "float",
      "DEFAULT": 0.82,
      "MIN": 0.1,
      "MAX": 1.5,
      "LABEL": "Diffusion A  [substrate]"
    },
    {
      "NAME": "diff_b",
      "TYPE": "float",
      "DEFAULT": 0.41,
      "MIN": 0.05,
      "MAX": 0.8,
      "LABEL": "Diffusion B  [replicant]"
    },
    {
      "NAME": "color_speed",
      "TYPE": "float",
      "DEFAULT": 0.12,
      "MIN": 0.0,
      "MAX": 1.0,
      "LABEL": "Color Cycle Speed"
    },
    {
      "NAME": "color_shift",
      "TYPE": "float",
      "DEFAULT": 0.0,
      "MIN": 0.0,
      "MAX": 1.0,
      "LABEL": "Color Shift  [palette offset]"
    },
    {
      "NAME": "inject_density",
      "TYPE": "float",
      "DEFAULT": 1.0,
      "MIN": 0.0,
      "MAX": 1.0,
      "LABEL": "Inject Density  [0=none, 1=full]"
    },
    {
      "NAME": "sim_scale",
      "TYPE": "float",
      "DEFAULT": 0.003,
      "MIN": 0.001,
      "MAX": 0.012,
      "LABEL": "Sim Scale  [pixel step — tune if patterns wrong size]"
    },
    {
      "NAME": "burst",
      "TYPE": "float",
      "DEFAULT": 0.0,
      "MIN": 0.0,
      "MAX": 1.0,
      "LABEL": "Burst  [map → audio peak]"
    },
    {
      "NAME": "reset",
      "TYPE": "event",
      "LABEL": "Reset"
    }
  ],
  "PASSES": [
    {
      "TARGET":     "rdBuffer",
      "PERSISTENT": true,
      "FLOAT":      true
    },
    {}
  ]
}*/

// ─── Noise / Hash Utilities ────────────────────────────────────────────────────

float hash1(float n) {
    return fract(sin(n) * 43758.5453123);
}

float hash2(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

// ─── Brain Wipe Color Palette ─────────────────────────────────────────────────
// Maps B concentration [0,1] to color. Frame: consciousness being overwritten.
// Low B  → void / dark neural background
// Mid B  → deep purple / violet invasion front
// High B → electric cyan / hot white saturation
//
// The cycling hue is a second axis — the overwrite has a carrier frequency.
// It should feel invasive, not relaxing.

vec3 brainWipePalette(float b, float cycle) {
    // Palette stops — deliberately unnatural, neurological, not bioluminescent
    vec3 c0 = vec3(0.01, 0.0,  0.03);   // near-black, trace violet
    vec3 c1 = vec3(0.45, 0.0,  0.85);   // deep electric purple
    vec3 c2 = vec3(0.0,  0.75, 1.0);    // hard cyan
    vec3 c3 = vec3(1.0,  1.0,  1.0);    // overload white

    // Smooth 4-stop gradient along B
    vec3 col = mix(c0, c1, smoothstep(0.0,  0.25, b));
    col       = mix(col, c2, smoothstep(0.2,  0.55, b));
    col       = mix(col, c3, smoothstep(0.65, 1.0,  b));

    // Cycle: slowly rotate the hue of mid-range values over time.
    // Applied as an additive red/green drift — keeps it from feeling static.
    float drift = sin(cycle * 6.28318) * 0.5 + 0.5;
    float mid   = smoothstep(0.1, 0.6, b) * (1.0 - smoothstep(0.6, 1.0, b));
    col.r += mid * drift * 0.25;
    col.b -= mid * drift * 0.15;

    return clamp(col, 0.0, 1.0);
}

// ─── Main ─────────────────────────────────────────────────────────────────────

void main() {
    vec2 uv = isf_FragNormCoord;
    // sim_scale replaces 1.0/RENDERSIZE — avoids RENDERSIZE host-compatibility issues.
    // Default 0.003 works for most resolutions; nudge up if patterns are too fine.
    vec2 px = vec2(sim_scale);

    // ═══════════════════════════════════════════════════════════════════════════
    // PASS 0 — Gray-Scott Reaction-Diffusion Update
    // rdBuffer stores: .r = A (substrate), .g = B (replicant)
    // A and B remain in [0, 1].
    // ═══════════════════════════════════════════════════════════════════════════
    if (PASSINDEX == 0) {

        vec4 center = IMG_NORM_PIXEL(rdBuffer, uv);
        float A = center.r;
        float B = center.g;

        // ─── Initialization / Reset ───────────────────────────────────────────
        // Persistent buffer starts at 0. Detect uninitialised state (A+B ≈ 0)
        // or a manual reset event. Seed A=1 everywhere, scatter B randomly.
        // After seeding, the RD dynamics take over within a few hundred frames.

        bool uninitialized = (A + B < 0.01);

        if (reset > 0.5 || uninitialized) {
            // Standard RD initialization: seed every pixel with uniform random B in [0, 0.25].
            // Sparse single-pixel seeds diffuse away before reaching critical mass.
            // Full-field noise lets the dynamics self-organize patterns across the whole canvas.
            float t     = fract(TIME * 0.001 + 1.0);
            float seedB = hash2(uv * 997.0 + vec2(t * 137.5, t * 251.3)) * 0.25;
            gl_FragColor = vec4(1.0, seedB, 0.0, 1.0);
            return;
        }


        // ─── Laplacian — 9-point weighted stencil ─────────────────────────────
        // Weights: cardinal 0.2, diagonal 0.05, center -1.0
        // More isotropic than the 4-point stencil; reduces grid artifacts.

        vec4 n  = IMG_NORM_PIXEL(rdBuffer, uv + vec2( 0.0,   px.y));
        vec4 s  = IMG_NORM_PIXEL(rdBuffer, uv + vec2( 0.0,  -px.y));
        vec4 e  = IMG_NORM_PIXEL(rdBuffer, uv + vec2( px.x,  0.0));
        vec4 w  = IMG_NORM_PIXEL(rdBuffer, uv + vec2(-px.x,  0.0));
        vec4 ne = IMG_NORM_PIXEL(rdBuffer, uv + vec2( px.x,  px.y));
        vec4 nw = IMG_NORM_PIXEL(rdBuffer, uv + vec2(-px.x,  px.y));
        vec4 se = IMG_NORM_PIXEL(rdBuffer, uv + vec2( px.x, -px.y));
        vec4 sw = IMG_NORM_PIXEL(rdBuffer, uv + vec2(-px.x, -px.y));

        float lapA = 0.2  * (n.r + s.r + e.r + w.r)
                   + 0.05 * (ne.r + nw.r + se.r + sw.r)
                   - center.r;

        float lapB = 0.2  * (n.g + s.g + e.g + w.g)
                   + 0.05 * (ne.g + nw.g + se.g + sw.g)
                   - center.g;


        // ─── Gray-Scott Update ────────────────────────────────────────────────
        // dA/dt = Da·∇²A  -  A·B²  +  f·(1-A)
        // dB/dt = Db·∇²B  +  A·B²  -  (f+k)·B
        // dt = 1.0 (baked into coefficient scale)

        float ABB  = A * B * B;

        float newA = A + (diff_a * lapA - ABB + feed * (1.0 - A));
        float newB = B + (diff_b * lapB + ABB - (kill + feed) * B);

        newA = clamp(newA, 0.0, 1.0);
        newB = clamp(newB, 0.0, 1.0);


        // ─── Continuous Injection ─────────────────────────────────────────────
        // Low-level B seeding keeps the system alive and prevents total decay.
        // Rate is tied to inject_density; burst (audio peak) spikes it hard.
        //
        // Uses two independent hash evaluations combined to produce a smooth
        // probability distribution. A single hash has a fixed maximum value
        // across all patch/frame combinations, creating a hard cutoff below
        // which ZERO injections occur. Combining two hashes breaks this ceiling
        // and allows very low inject_density values to produce rare events.

        float frameT = floor(TIME * 60.0);  // quantized to ~frame rate

        // Patch-based injection: group pixels into patches so injected B
        // has critical mass to survive diffusion and grow into the RD field.
        // Patch scale ~1/25 of UV range = ~12-20 sim pixels wide at default sim_scale.
        vec2  injectPatch = floor(uv * 25.0);
        float roll1 = hash2(injectPatch * 7.3 + hash1(frameT) * 97.3);
        float roll2 = hash2(injectPatch * 31.7 + vec2(hash1(frameT * 1.618 + 5.7)));
        float injectRoll = fract(roll1 + roll2);
        // inject_density is 0–1 user-facing; remap to internal 0–0.002 range
        float effectiveDensity = inject_density * 0.002 + burst * 0.08;

        if (injectRoll > (1.0 - effectiveDensity)) {
            newB = min(newB + 0.35, 1.0);
        }

        // Burst pulse: larger patches, wider scatter
        if (burst > 0.05) {
            vec2  burstPatch = floor(uv * 12.0);
            float burstRoll  = hash2(burstPatch * 13.1 + hash1(frameT * 3.7) * 31.1);
            newB = min(newB + burst * 0.55 * step(0.80, burstRoll), 1.0);
        }


        gl_FragColor = vec4(newA, newB, 0.0, 1.0);


    // ═══════════════════════════════════════════════════════════════════════════
    // PASS 1 — Colorization
    // Reads the settled RD state and maps it to brain wipe visuals.
    // ═══════════════════════════════════════════════════════════════════════════
    } else {

        vec4  state = IMG_NORM_PIXEL(rdBuffer, uv);
        float A     = state.r;
        float B     = state.g;

        // Reaction front: where A is depleted and B is actively consuming.
        // This is the most visually interesting zone — the invasion wavefront.
        float front = A * B * B;  // proportional to reaction rate

        // Time-based palette cycle
        float cycle = fract(color_shift + TIME * color_speed * 0.05);

        // Color from B concentration
        vec3 col = brainWipePalette(B, cycle);

        // Brighten the reaction front to emphasize wavefronts
        col += vec3(0.6, 0.9, 1.0) * front * 2.5;

        // Subtle structure in the "healthy" zones (high A, low B):
        // very faint purple tint so it's not pure black — feels like neural substrate
        col += vec3(0.03, 0.0, 0.06) * A * (1.0 - B);

        // Clamp — front boost can push past 1
        col = clamp(col, 0.0, 1.0);

        gl_FragColor = vec4(col, 1.0);
    }
}
