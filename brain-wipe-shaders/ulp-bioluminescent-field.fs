/*{
  "DESCRIPTION": "ULP Bioluminescent Field — drifting deep-sea organisms pulsing with cold light. Multi-layer hash-grid particle system with independent glow cycles, depth parallax, and propagating bloom events. Lair frame: looking out into the abyss.",
  "CREDIT": "Undersea Lair Project / P-Chops",
  "ISFVSN": "2",
  "CATEGORIES": ["Generator", "Particles", "ULP"],
  "INPUTS": [
    {
      "NAME": "density",
      "TYPE": "float",
      "DEFAULT": 0.75,
      "MIN": 0.0,
      "MAX": 1.0,
      "LABEL": "Density  [organism count]"
    },
    {
      "NAME": "drift_speed",
      "TYPE": "float",
      "DEFAULT": 0.35,
      "MIN": 0.0,
      "MAX": 1.0,
      "LABEL": "Drift Speed  [current strength]"
    },
    {
      "NAME": "pulse_rate",
      "TYPE": "float",
      "DEFAULT": 0.4,
      "MIN": 0.0,
      "MAX": 1.0,
      "LABEL": "Pulse Rate  [glow cycle speed]"
    },
    {
      "NAME": "glow_size",
      "TYPE": "float",
      "DEFAULT": 0.8,
      "MIN": 0.0,
      "MAX": 1.0,
      "LABEL": "Glow Size  [organism radius]"
    },
    {
      "NAME": "depth_layers",
      "TYPE": "float",
      "DEFAULT": 0.5,
      "MIN": 0.0,
      "MAX": 1.0,
      "LABEL": "Depth Layers  [parallax spread]"
    },
    {
      "NAME": "color_temp",
      "TYPE": "float",
      "DEFAULT": 0.3,
      "MIN": 0.0,
      "MAX": 1.0,
      "LABEL": "Color Temp  [0=cyan/green, 1=violet/magenta]"
    },
    {
      "NAME": "wander",
      "TYPE": "float",
      "DEFAULT": 0.4,
      "MIN": 0.0,
      "MAX": 1.0,
      "LABEL": "Wander  [Brownian drift intensity]"
    },
    {
      "NAME": "bloom_spread",
      "TYPE": "float",
      "DEFAULT": 0.4,
      "MIN": 0.0,
      "MAX": 1.0,
      "LABEL": "Bloom Spread  [how far burst propagates]"
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

// ─── Hash Utilities ──────────────────────────────────────────────────────────

float hash1(float n) {
    return fract(sin(n) * 43758.5453123);
}

float hash2(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

vec2 hash2v(vec2 p) {
    return vec2(
        fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453),
        fract(sin(dot(p, vec2(269.5, 183.3))) * 43758.5453)
    );
}


// ─── Bioluminescent Palette ──────────────────────────────────────────────────
// Two poles: cyan/green (dinoflagellate) and violet/magenta (ctenophore).
// color_temp blends between them. Glow is always brightest at center,
// fading to a colored halo, then to near-black.

vec3 bioColor(float intensity, float variant, float temp) {
    // Cyan-green pole
    vec3 cyan_core  = vec3(0.4, 1.0, 0.95);
    vec3 cyan_halo  = vec3(0.0, 0.25, 0.3);

    // Violet-magenta pole
    vec3 viol_core  = vec3(0.7, 0.3, 1.0);
    vec3 viol_halo  = vec3(0.15, 0.0, 0.25);

    // Per-organism hue variation (subtle shift within the pole)
    float hueShift = variant * 0.3;

    vec3 core = mix(cyan_core, viol_core, temp);
    vec3 halo = mix(cyan_halo, viol_halo, temp);

    // Shift green↔blue within cyan pole, red↔blue within violet pole
    core.gb += vec2(hueShift, -hueShift) * (1.0 - temp) * 0.2;
    core.rb += vec2(hueShift, -hueShift) * temp * 0.2;

    // Intensity maps core→halo→black
    vec3 col = mix(halo, core, smoothstep(0.0, 0.8, intensity));
    col *= intensity;

    return col;
}


// ─── Single Particle Layer ───────────────────────────────────────────────────
// Draws one depth layer of particles. Each layer has its own grid scale,
// drift speed, and glow size, creating parallax depth.

vec3 particleLayer(
    vec2 uv,
    float layerIndex,
    float layerScale,     // grid cell size (larger = fewer, bigger particles)
    float layerSpeed,     // drift multiplier
    float layerGlow,      // glow radius multiplier
    float layerBright,    // brightness multiplier
    float time,
    float dens,
    float pRate,
    float cTemp,
    float wanderAmt,      // Brownian drift intensity
    float bSpread,
    float bst
) {
    vec3 col = vec3(0.0);

    // Scale UV to grid
    vec2 scaledUV = uv / layerScale;

    // Slow drift: dominant direction is upward (organisms rising / us sinking)
    // with a gentle lateral current. Each layer drifts at its own rate.
    float driftT = time * layerSpeed;
    vec2 drift = vec2(
        sin(time * 0.07 + layerIndex * 1.3) * 0.3,  // gentle lateral sway
        -1.0                                           // upward drift
    ) * driftT;

    vec2 driftedUV = scaledUV + drift;

    // Grid cell
    vec2 cellID = floor(driftedUV);
    vec2 cellUV = fract(driftedUV);

    // Check this cell and all 8 neighbors (particles near cell edges
    // need to glow into adjacent cells)
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            vec2 neighbor = vec2(float(dx), float(dy));
            vec2 nID = cellID + neighbor;

            // Per-cell random: does this cell have a particle?
            float cellRoll = hash2(nID * 17.3 + layerIndex * 91.1);
            if (cellRoll > dens) continue;

            // Particle position within cell (randomized, not centered)
            vec2 particlePos = hash2v(nID * 31.7 + layerIndex * 53.3) * 0.7 + 0.15;

            // Brownian wander within cell — wanderAmt scales the amplitude.
            // At 0: organisms are stationary within their cells.
            // At 1: large drunken paths, organisms nearly escape their cells.
            float wanderT = time * mix(0.08, 0.35, wanderAmt) * layerSpeed;
            float wanderR = mix(0.03, 0.35, wanderAmt);
            particlePos += vec2(
                sin(wanderT * hash1(nID.x * 7.1 + nID.y * 13.3 + layerIndex) * 3.0 + nID.x) * wanderR,
                cos(wanderT * hash1(nID.y * 11.7 + nID.x * 5.3 + layerIndex) * 2.7 + nID.y) * wanderR
            );

            // Distance from fragment to particle center
            vec2 toParticle = (neighbor + particlePos) - cellUV;
            float dist = length(toParticle);

            // Glow radius (varies per particle)
            float baseRadius = 0.15 + hash1(nID.x * 41.3 + nID.y * 67.1 + layerIndex) * 0.2;
            float radius = baseRadius * layerGlow;

            if (dist > radius * 3.0) continue;  // early out

            // ─── Glow Pulse ──────────────────────────────────────────────
            // Each organism has its own phase and frequency.
            // The pulse is not a simple sine — it's a sharp brightening
            // followed by a slow fade, like a real bioluminescent flash.

            float pulsePhase = hash1(nID.x * 23.1 + nID.y * 37.7 + layerIndex * 7.0);
            float pulseFreq  = mix(0.3, 1.2, hash1(nID.y * 19.3 + nID.x * 29.1 + layerIndex)) * pRate;
            float pulseT     = fract(time * pulseFreq + pulsePhase);

            // Sharp attack, slow decay: pow(1 - t, 3) gives fast rise at t≈0, slow tail
            float pulse = pow(1.0 - pulseT, 3.0);

            // Some organisms are "always on" at low level, others go fully dark
            float restLevel = hash1(nID.x * 53.7 + nID.y * 71.3 + layerIndex) * 0.15;
            float glowIntensity = restLevel + pulse * (1.0 - restLevel);

            // ─── Burst Bloom ─────────────────────────────────────────────
            // Audio peak triggers a propagating flash. The bloom radiates
            // outward from screen center; particles closer to center fire
            // first. bloom_spread controls how far it reaches.

            if (bst > 0.02) {
                // World-space position of this particle
                vec2 worldPos = (nID + particlePos) * layerScale - drift * layerScale;
                // Normalized distance from screen center
                float distFromCenter = length(worldPos - vec2(0.5));

                // Propagation wave: expands outward over ~0.5 seconds
                float waveFront = fract(time * 2.0) * bSpread * 2.0;
                float waveHit = 1.0 - smoothstep(0.0, 0.15, abs(distFromCenter - waveFront));

                // Bloom adds to glow
                glowIntensity = min(glowIntensity + bst * waveHit * 0.9, 1.0);
            }

            // ─── Radial Glow Shape ───────────────────────────────────────
            // Soft exponential falloff — not a hard circle.
            // Core is bright and small; halo extends further and dimmer.

            float core = exp(-dist * dist / (radius * radius * 0.15)) * 0.8;
            float halo = exp(-dist * dist / (radius * radius * 1.5))  * 0.6;
            float shape = core + halo;

            // Per-particle color variation
            float colorVariant = hash1(nID.x * 61.3 + nID.y * 83.7 + layerIndex);

            vec3 particleCol = bioColor(glowIntensity * shape, colorVariant, cTemp);
            col += particleCol * layerBright;
        }
    }

    return col;
}


// ─── Main ────────────────────────────────────────────────────────────────────

void main() {
    vec2 uv = isf_FragNormCoord;

    // Aspect ratio correction
    float aspect = RENDERSIZE.x / RENDERSIZE.y;
    vec2 auv = vec2(uv.x * aspect, uv.y);

    // Map parameters to useful ranges
    float dens     = mix(0.2, 0.85, density);
    float dSpeed   = mix(0.02, 0.15, drift_speed);
    float pRate    = mix(0.15, 0.8, pulse_rate);
    float gSize    = mix(0.5, 2.0, glow_size);
    float bSpread  = mix(0.2, 1.5, bloom_spread);
    float wandAmt  = wander;

    // Number of depth layers: 3 to 5 based on depth_layers param
    int numLayers = int(mix(3.0, 5.0, depth_layers));

    vec3 col = vec3(0.0);

    // ─── Background ──────────────────────────────────────────────────────────
    // Deep abyss: murky underwater ambient with visible depth
    vec3 bgTop    = vec3(0.13, 0.21, 0.30);
    vec3 bgBottom = vec3(0.18, 0.28, 0.38);
    col = mix(bgBottom, bgTop, uv.y);

    // Ambient scatter — light diffusing through murky water
    float scatter = sin(auv.x * 3.1 + TIME * 0.05) * sin(auv.y * 2.7 - TIME * 0.03)
                  + sin(auv.x * 1.3 - auv.y * 2.1 + TIME * 0.07) * 0.5;
    scatter = scatter * 0.5 + 0.5;  // remap to 0-1
    col += mix(vec3(0.05, 0.12, 0.14), vec3(0.12, 0.07, 0.16), color_temp) * scatter * 0.30;

    // ─── Particle Layers ─────────────────────────────────────────────────────
    // Back layers: small, dim, slow, dense (distant organisms)
    // Front layers: large, bright, fast, sparse (close organisms)

    for (int i = 0; i < 5; i++) {
        if (i >= numLayers) break;
        float fi = float(i);
        float t  = fi / float(numLayers - 1);  // 0 = back, 1 = front

        // Exponential scaling: front layers are much larger than back
        float layerScale  = mix(0.06, 0.22, t * t);
        float layerSpeed  = mix(0.6, 1.4, t) * dSpeed;
        float layerGlow   = mix(0.5, 1.5, t) * gSize;
        float layerBright = mix(0.7, 1.2, t * t);

        // Back layers are denser (more tiny distant organisms)
        float layerDens = mix(dens, dens * 0.5, t);

        col += particleLayer(
            auv, fi,
            layerScale, layerSpeed, layerGlow, layerBright,
            TIME, layerDens, pRate, color_temp, wandAmt, bSpread, burst
        );
    }

    // ─── Marine Snow ─────────────────────────────────────────────────────────
    // Very faint, very small, very numerous drifting particles.
    // Not bioluminescent — just debris catching distant light.
    // Adds depth and atmosphere between the glowing organisms.

    vec2 snowUV = auv * 40.0 + vec2(TIME * 0.02, -TIME * 0.08);
    vec2 snowCell = floor(snowUV);
    vec2 snowFrac = fract(snowUV);

    float snowRoll = hash2(snowCell * 7.1);
    if (snowRoll > 0.92) {
        vec2 snowPos = hash2v(snowCell * 13.3) * 0.8 + 0.1;
        float snowDist = length(snowFrac - snowPos);
        float snowDot = exp(-snowDist * snowDist / 0.003) * 0.08;
        // Faint flicker
        snowDot *= 0.5 + 0.5 * sin(TIME * 2.0 + snowRoll * 100.0);
        col += vec3(snowDot * 0.6, snowDot * 0.7, snowDot * 0.8);
    }

    // ─── Depth Fog ───────────────────────────────────────────────────────────
    // Very subtle blue-black fog toward edges — frames the view as through glass.

    float fog = smoothstep(0.3, 0.95, length(uv - 0.5) * 1.6);
    col = mix(col, vec3(0.005, 0.01, 0.025), fog * 0.6);

    // ─── Burst Global Flash ──────────────────────────────────────────────────
    // Audio peak adds a very subtle ambient brightening across the whole field
    col += vec3(0.01, 0.03, 0.04) * burst * 0.5;

    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
