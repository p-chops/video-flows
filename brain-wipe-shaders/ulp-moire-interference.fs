/*{
  "DESCRIPTION": "ULP Moiré Interference — overlapping rotating line/circle grids produce complex emergent beating patterns. Dense, hypnotic, constantly evolving from simple mathematical rules.",
  "CREDIT": "Undersea Lair Project / P-Chops",
  "ISFVSN": "2",
  "CATEGORIES": ["Generator", "Ambient", "ULP"],
  "INPUTS": [
    {
      "NAME": "pattern_type",
      "TYPE": "float",
      "DEFAULT": 0.3,
      "MIN": 0.0,
      "MAX": 1.0,
      "LABEL": "Pattern  [0=lines, 0.5=circles, 1=mixed]"
    },
    {
      "NAME": "density",
      "TYPE": "float",
      "DEFAULT": 0.5,
      "MIN": 0.0,
      "MAX": 1.0,
      "LABEL": "Density  [line/circle frequency]"
    },
    {
      "NAME": "rotation_speed",
      "TYPE": "float",
      "DEFAULT": 0.4,
      "MIN": 0.0,
      "MAX": 1.0,
      "LABEL": "Rotation Speed  [grid rotation rate]"
    },
    {
      "NAME": "n_layers",
      "TYPE": "float",
      "DEFAULT": 0.5,
      "MIN": 0.0,
      "MAX": 1.0,
      "LABEL": "Layers  [number of overlapping grids]"
    },
    {
      "NAME": "color_shift",
      "TYPE": "float",
      "DEFAULT": 0.3,
      "MIN": 0.0,
      "MAX": 1.0,
      "LABEL": "Color Shift  [0=mono, 0.5=subtle, 1=chromatic]"
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

// ─── Grid patterns ──────────────────────────────────────────────────────────

float lineGrid(vec2 p, float freq) {
    return 0.5 + 0.5 * sin(p.x * freq);
}

float circleGrid(vec2 p, float freq) {
    return 0.5 + 0.5 * sin(length(p) * freq);
}

mat2 rot2(float a) {
    float c = cos(a), s = sin(a);
    return mat2(c, -s, s, c);
}

// ─── Main ───────────────────────────────────────────────────────────────────

void main() {
    vec2 uv = isf_FragNormCoord * 2.0 - 1.0;
    uv.x *= RENDERSIZE.x / RENDERSIZE.y;

    float freq = mix(15.0, 60.0, density);
    float speed = mix(0.1, 0.6, rotation_speed);
    int layers = int(mix(2.0, 5.0, n_layers));

    float sumR = 1.0;
    float sumG = 1.0;
    float sumB = 1.0;

    for (int i = 0; i < 5; i++) {
        if (i >= layers) break;

        float fi = float(i);

        // Each layer gets a different rotation rate and phase
        float angle = TIME * speed * (0.3 + fi * 0.25) + fi * 1.2566; // fi * 2pi/5
        float layerFreq = freq * (0.8 + fi * 0.15);

        // Slight offset per layer for asymmetry
        vec2 offset = vec2(
            sin(TIME * 0.13 + fi * 2.0) * 0.3,
            cos(TIME * 0.11 + fi * 1.7) * 0.3
        );

        vec2 p = rot2(angle) * (uv + offset);

        float v;
        // Mix between line and circle patterns
        float lineV = lineGrid(p, layerFreq);
        float circV = circleGrid(p, layerFreq);

        if (pattern_type < 0.33) {
            v = lineV;
        } else if (pattern_type < 0.66) {
            v = circV;
        } else {
            // Mixed: alternate layers
            v = (i % 2 == 0) ? lineV : circV;
        }

        // Chromatic separation — slight frequency offset per channel
        float chromaOffset = color_shift * 3.0;
        vec2 pR = rot2(angle + chromaOffset * 0.003) * (uv + offset);
        vec2 pB = rot2(angle - chromaOffset * 0.003) * (uv + offset);

        float vR, vB;
        if (pattern_type < 0.33) {
            vR = lineGrid(pR, layerFreq * (1.0 + chromaOffset * 0.01));
            vB = lineGrid(pB, layerFreq * (1.0 - chromaOffset * 0.01));
        } else if (pattern_type < 0.66) {
            vR = circleGrid(pR, layerFreq * (1.0 + chromaOffset * 0.01));
            vB = circleGrid(pB, layerFreq * (1.0 - chromaOffset * 0.01));
        } else {
            if (i % 2 == 0) {
                vR = lineGrid(pR, layerFreq * (1.0 + chromaOffset * 0.01));
                vB = lineGrid(pB, layerFreq * (1.0 - chromaOffset * 0.01));
            } else {
                vR = circleGrid(pR, layerFreq * (1.0 + chromaOffset * 0.01));
                vB = circleGrid(pB, layerFreq * (1.0 - chromaOffset * 0.01));
            }
        }

        // Multiplicative interference — true moiré beating
        sumR *= vR;
        sumG *= v;
        sumB *= vB;
    }

    // Multiplicative products are dark — pow(0.5^n, 1/n) ≈ 0.5
    // Use 1/layers to undo the compounding
    float rescale = 1.0 / float(layers);
    sumR = pow(max(sumR, 0.0), rescale);
    sumG = pow(max(sumG, 0.0), rescale);
    sumB = pow(max(sumB, 0.0), rescale);

    vec3 col = vec3(sumR, sumG, sumB);

    // Sharpen the interference fringes
    col = smoothstep(0.05, 0.85, col);
    col *= brightness;

    // Subtle vignette
    float vign = 1.0 - dot(uv * 0.3, uv * 0.3);
    col *= clamp(vign, 0.0, 1.0);

    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
