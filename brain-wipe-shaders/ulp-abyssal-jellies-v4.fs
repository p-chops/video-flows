/*{
  "DESCRIPTION": "ULP Abyssal Jellies — translucent jellyfish drifting through the deep. Fixed pool of organisms at varying depths: some loom large in the foreground, others are tiny distant specks. Pulsing bells with trailing tendrils. Lair frame: looking out into the abyss.",
  "CREDIT": "Undersea Lair Project / P-Chops",
  "ISFVSN": "2",
  "CATEGORIES": ["Generator", "Creatures", "ULP"],
  "INPUTS": [
    {
      "NAME": "swim_rate",
      "TYPE": "float",
      "DEFAULT": 0.45,
      "MIN": 0.0,
      "MAX": 1.0,
      "LABEL": "Swim Rate  [pulse frequency]"
    },
    {
      "NAME": "drift_speed",
      "TYPE": "float",
      "DEFAULT": 0.35,
      "MIN": 0.0,
      "MAX": 1.0,
      "LABEL": "Drift Speed  [ambient current]"
    },
    {
      "NAME": "wander",
      "TYPE": "float",
      "DEFAULT": 0.45,
      "MIN": 0.0,
      "MAX": 1.0,
      "LABEL": "Wander  [lateral drift intensity]"
    },
    {
      "NAME": "size_range",
      "TYPE": "float",
      "DEFAULT": 0.6,
      "MIN": 0.0,
      "MAX": 1.0,
      "LABEL": "Size Range  [0=all similar, 1=huge+tiny mix]"
    },
    {
      "NAME": "tendril_length",
      "TYPE": "float",
      "DEFAULT": 0.55,
      "MIN": 0.0,
      "MAX": 1.0,
      "LABEL": "Tendril Length"
    },
    {
      "NAME": "translucency",
      "TYPE": "float",
      "DEFAULT": 0.6,
      "MIN": 0.0,
      "MAX": 1.0,
      "LABEL": "Translucency  [see-through amount]"
    },
    {
      "NAME": "color_temp",
      "TYPE": "float",
      "DEFAULT": 0.25,
      "MIN": 0.0,
      "MAX": 1.0,
      "LABEL": "Color Temp  [0=cyan/green, 1=violet/magenta]"
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

vec2 hash2v(float n) {
    return vec2(
        fract(sin(n * 127.1) * 43758.5453),
        fract(sin(n * 269.5) * 43758.5453)
    );
}


// ─── Jellyfish Color ─────────────────────────────────────────────────────────

vec3 jellyColor(float intensity, float variant, float temp, float rim) {
    vec3 cyan_body = vec3(0.05, 0.35, 0.4);
    vec3 cyan_glow = vec3(0.2, 0.9, 0.85);
    vec3 viol_body = vec3(0.2, 0.05, 0.35);
    vec3 viol_glow = vec3(0.6, 0.2, 0.95);

    vec3 body = mix(cyan_body, viol_body, temp);
    vec3 glow = mix(cyan_glow, viol_glow, temp);

    body.gb += vec2(variant, -variant) * 0.08;
    glow.gb += vec2(variant, -variant) * 0.12;

    vec3 col = mix(body, glow, rim * 0.7 + intensity * 0.3);
    col *= intensity;
    return col;
}


// ─── Single Jellyfish ────────────────────────────────────────────────────────
// Everything in screen UV space. bellR is the bell radius on screen.

vec4 drawJellyfish(
    vec2 fragUV,
    vec2 center,
    float bellR,
    float tLenMult,
    float time,
    float seed,
    float swimFreq,
    float cTemp,
    float bst
) {
    vec2 p = fragUV - center;

    // Early out — skip if clearly outside the organism's bounding area
    float maxReach = bellR * (1.5 + 5.5 * tLenMult);
    if (abs(p.x) > bellR * 1.5 || p.y > bellR * 1.3 || -p.y > maxReach) {
        return vec4(0.0);
    }

    // ─── Swim Cycle ──────────────────────────────────────────────────────
    float phase = fract(time * swimFreq + seed);
    float contraction;
    if (phase < 0.3) {
        contraction = smoothstep(0.0, 0.15, phase) * smoothstep(0.3, 0.18, phase);
    } else {
        contraction = 0.0;
    }
    contraction = max(contraction, bst * 0.7);

    // ─── Bell ────────────────────────────────────────────────────────────
    float aspectY = mix(1.1, 0.55, contraction);
    float aspectX = mix(1.0, 1.25, contraction);

    vec2 bp = vec2(p.x / aspectX, p.y / aspectY);
    bp.y -= bellR * 0.15;
    float dist = length(bp);

    // Membrane
    float edgeW = max(bellR * 0.06, 0.003);  // minimum edge width for tiny jellies
    float outer = smoothstep(bellR, bellR - edgeW, dist);
    float inner = smoothstep(bellR * 0.65, bellR * 0.72, dist);
    float membrane = outer * inner;

    // Dome fill
    float cap = outer * smoothstep(-bellR * 0.1, bellR * 0.3, bp.y);
    float density = max(membrane, cap * 0.35);

    // Opening cutoff
    float opening = smoothstep(-bellR * 0.2, bellR * 0.08, p.y / aspectY);
    density *= opening;

    // Rim + lip glow
    float rim = membrane;
    float lipDist = abs(p.y / aspectY + bellR * 0.05);
    float lipRing = smoothstep(bellR * 0.15, 0.0, lipDist) * outer;
    rim = max(rim, lipRing);

    // ─── Tendrils ────────────────────────────────────────────────────────
    float tendrilLen = bellR * mix(1.5, 5.0, tLenMult);
    float tendrilW = max(bellR * 0.03, 0.002);  // minimum width for tiny jellies

    int numTendrils = 3 + int(hash1(seed * 31.7) * 3.0);
    float totalTendril = 0.0;

    if (p.y < bellR * 0.1 && -p.y < tendrilLen * 1.5) {
        for (int i = 0; i < 5; i++) {
            if (i >= numTendrils) break;
            float fi = float(i);
            float frac = (fi + 0.5) / float(numTendrils);

            float attachX = (frac - 0.5) * bellR * mix(1.6, 2.1, contraction);

            float tSeed = hash1(seed * 17.3 + fi * 41.7);
            float thisLen = tendrilLen * (0.6 + tSeed * 0.8);

            float ty = -p.y;
            if (ty > thisLen || ty < 0.0) continue;
            float t = ty / thisLen;

            float sw = bellR * 0.8;
            float sway1 = sin(t * 4.5 + time * 1.1 + seed * 10.0 + fi) * 0.15 * sw;
            float sway2 = sin(t * 8.0 - time * 0.7 + seed * 7.0 + fi) * 0.07 * sw;
            float pulse = sin(t * 6.0 - contraction * 8.0 + seed + fi) * 0.12 * sw * contraction;

            float centerX = attachX + (sway1 + sway2 + pulse) * (0.3 + t * 2.0);
            float dx = abs(p.x - centerX);

            float w = tendrilW * mix(1.0, 0.15, t * t);
            float tI = smoothstep(w, w * 0.1, dx) * (1.0 - t * t);

            float glow = exp(-dx * dx / (w * w * 30.0)) * 0.3 * (1.0 - t);
            tI += glow;

            totalTendril += tI;
        }
    }
    totalTendril = clamp(totalTendril, 0.0, 1.0);

    // ─── Combine ─────────────────────────────────────────────────────────
    float variant = hash1(seed * 61.3);
    vec3 bellCol = jellyColor(density, variant, cTemp, rim);
    vec3 tendrilCol = jellyColor(totalTendril * 0.7, variant, cTemp, totalTendril * 0.5);

    float internalPulse = contraction * 0.4;
    bellCol += jellyColor(1.0, variant, cTemp, 0.5) * internalPulse * density;

    vec3 col = bellCol + tendrilCol;
    float alpha = clamp(density + totalTendril * 0.6, 0.0, 1.0);

    return vec4(col, alpha);
}


// ─── Main ────────────────────────────────────────────────────────────────────

void main() {
    vec2 uv = isf_FragNormCoord;
    float aspect = RENDERSIZE.x / RENDERSIZE.y;
    vec2 auv = vec2(uv.x * aspect, uv.y);

    float sRate    = mix(0.25, 1.0, swim_rate);
    float dSpeed   = mix(0.01, 0.06, drift_speed);
    float wand     = wander;
    float tLenMult = tendril_length;
    float transAlpha = mix(0.5, 1.0, 1.0 - translucency);

    // ─── Background ──────────────────────────────────────────────────────
    vec3 col = vec3(0.0);
    vec3 bgTop    = vec3(0.0,  0.005, 0.02);
    vec3 bgBottom = vec3(0.008, 0.015, 0.035);
    col = mix(bgBottom, bgTop, uv.y);

    // Faint light shaft from above
    float shaft = exp(-pow((uv.x - 0.5) * 3.0, 2.0)) * 0.015;
    shaft *= smoothstep(0.3, 1.0, uv.y);
    col += vec3(shaft * 0.3, shaft * 0.5, shaft * 0.6);

    // ─── Jellyfish Pool ──────────────────────────────────────────────────
    // 16 fixed organisms. No grid — each one has a directly computed position.
    // Their "depth" (index-based) determines size, speed, and brightness.
    //
    // Index 0-7: mid-distance (moderate size, visible)
    // Index 8-12: mid-close
    // Index 13-14: close (large, bright)
    // Index 15: hero — very close, largest on screen

    for (int i = 0; i < 16; i++) {
        float fi = float(i);
        float seed = fi * 7.31 + 1.0;

        // Depth: 0=far, 1=near. Non-linear so most are mid/far.
        float depth = fi / 15.0;
        depth = depth * depth;

        // Bell radius: smallest are still clearly visible, largest fill the frame.
        // Raised minimum from 0.008 to 0.03 so nothing is a tiny speck.
        float minR = 0.03;
        float maxR = mix(0.08, 0.4, size_range);
        float bellR = mix(minR, maxR, depth);

        // Brightness: raised floor so even distant organisms read clearly
        float alpha = mix(0.35, 1.0, depth) * transAlpha;

        // Base position: wrapping Lissajous paths so organisms cycle
        // through the frame on long periods. Each has unique frequencies.
        float freqX = 0.013 + hash1(seed * 3.1) * 0.02;
        float freqY = 0.009 + hash1(seed * 5.7) * 0.015;
        float phaseX = hash1(seed * 11.3) * 6.28;
        float phaseY = hash1(seed * 13.7) * 6.28;

        // Range: organisms wander across a region wider than the screen
        // so they enter and exit naturally
        float rangeX = aspect * 0.8 + bellR * 2.0;
        float rangeY = 0.8 + bellR * 2.0;

        vec2 basePos = vec2(
            aspect * 0.5 + sin(TIME * freqX + phaseX) * rangeX,
            0.5 + sin(TIME * freqY + phaseY) * rangeY
        );

        // Upward drift (we're descending, organisms rise past us)
        // Near organisms drift faster (parallax)
        float driftY = TIME * dSpeed * mix(0.3, 1.5, depth);
        basePos.y = fract(basePos.y + driftY * 0.3) * (1.0 + bellR * 4.0) - bellR * 2.0;

        // Wander: additional Brownian-ish drift
        float wandT = TIME * mix(0.03, 0.12, wand);
        basePos.x += sin(wandT * hash1(seed * 17.1) * 3.0 + seed * 40.0) * wand * 0.15;
        basePos.y += cos(wandT * hash1(seed * 19.3) * 2.5 + seed * 30.0) * wand * 0.05;

        // Swim frequency: near ones pulse slower (larger = more majestic)
        float swimFreq = mix(0.5, 0.2, depth) * sRate;

        // Swim propulsion: upward kick on each stroke
        float swimPhase = fract(TIME * swimFreq + hash1(seed * 23.1));
        float propulsion;
        if (swimPhase < 0.3) {
            propulsion = smoothstep(0.0, 0.2, swimPhase) * bellR * 0.5;
        } else {
            propulsion = bellR * 0.5 * (1.0 - smoothstep(0.3, 1.0, swimPhase));
        }
        basePos.y += propulsion;

        // Draw
        vec4 jelly = drawJellyfish(
            auv, basePos,
            bellR, tLenMult,
            TIME, seed, swimFreq,
            color_temp, burst
        );

        col += jelly.rgb * jelly.a * alpha;
    }

    // ─── Marine Snow ─────────────────────────────────────────────────────
    vec2 snowUV = auv * 35.0 + vec2(TIME * 0.015, -TIME * 0.06);
    vec2 snowCell = floor(snowUV);
    vec2 snowFrac = fract(snowUV);
    float snowRoll = fract(sin(dot(snowCell, vec2(127.1, 311.7))) * 43758.5453);
    if (snowRoll > 0.92) {
        vec2 snowPos = vec2(
            fract(sin(dot(snowCell * 13.3, vec2(127.1, 311.7))) * 43758.5453),
            fract(sin(dot(snowCell * 13.3, vec2(269.5, 183.3))) * 43758.5453)
        ) * 0.8 + 0.1;
        float snowDist = length(snowFrac - snowPos);
        float snowDot = exp(-snowDist * snowDist / 0.002) * 0.06;
        snowDot *= 0.5 + 0.5 * sin(TIME * 1.5 + snowRoll * 80.0);
        col += vec3(snowDot * 0.5, snowDot * 0.6, snowDot * 0.7);
    }

    // ─── Depth Fog ───────────────────────────────────────────────────────
    float fog = smoothstep(0.25, 0.9, length(uv - 0.5) * 1.5);
    col = mix(col, vec3(0.003, 0.008, 0.02), fog * 0.55);

    // ─── Burst ───────────────────────────────────────────────────────────
    col += vec3(0.008, 0.02, 0.03) * burst * 0.4;

    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
