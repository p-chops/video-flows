/*{
  "DESCRIPTION": "ULP Voronoi Flow — animated Voronoi cells with drifting seed points. Organic cellular topology that constantly reshapes: cells split, merge, and flow. Generates both hard edge networks and smooth interior gradients.",
  "CREDIT": "Undersea Lair Project / P-Chops",
  "ISFVSN": "2",
  "CATEGORIES": ["Generator", "Ambient", "ULP"],
  "INPUTS": [
    {
      "NAME": "cell_scale",
      "TYPE": "float",
      "DEFAULT": 0.5,
      "MIN": 0.0,
      "MAX": 1.0,
      "LABEL": "Cell Scale  [size of cells]"
    },
    {
      "NAME": "flow_speed",
      "TYPE": "float",
      "DEFAULT": 0.4,
      "MIN": 0.0,
      "MAX": 1.0,
      "LABEL": "Flow Speed  [seed point drift rate]"
    },
    {
      "NAME": "edge_glow",
      "TYPE": "float",
      "DEFAULT": 0.6,
      "MIN": 0.0,
      "MAX": 1.0,
      "LABEL": "Edge Glow  [brightness of cell boundaries]"
    },
    {
      "NAME": "color_mode",
      "TYPE": "float",
      "DEFAULT": 0.3,
      "MIN": 0.0,
      "MAX": 1.0,
      "LABEL": "Color Mode  [0=cool ocean, 0.5=warm mineral, 1=neon]"
    },
    {
      "NAME": "warp_amount",
      "TYPE": "float",
      "DEFAULT": 0.3,
      "MIN": 0.0,
      "MAX": 1.0,
      "LABEL": "Warp  [domain distortion of cell grid]"
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
    return fract(sin(p) * 43758.5453123);
}

float snoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    float a = dot(hash2(i) * 2.0 - 1.0, f);
    float b = dot(hash2(i + vec2(1.0, 0.0)) * 2.0 - 1.0, f - vec2(1.0, 0.0));
    float c = dot(hash2(i + vec2(0.0, 1.0)) * 2.0 - 1.0, f - vec2(0.0, 1.0));
    float d = dot(hash2(i + vec2(1.0, 1.0)) * 2.0 - 1.0, f - vec2(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// ─── Voronoi ────────────────────────────────────────────────────────────────

// Returns vec3(min_dist, second_dist, cell_id)
vec3 voronoi(vec2 p, float t) {
    vec2 n = floor(p);
    vec2 f = fract(p);

    float md = 8.0;   // min distance
    float md2 = 8.0;  // second min distance
    float id = 0.0;

    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            vec2 g = vec2(float(i), float(j));
            vec2 o = hash2(n + g);

            // Animate seed points — circular orbits + drift
            float phase = dot(n + g, vec2(7.3, 13.7));
            o = 0.5 + 0.45 * sin(t * 0.8 + phase + o * 6.2831);

            vec2 r = g + o - f;
            float d = dot(r, r);

            if (d < md) {
                md2 = md;
                md = d;
                id = dot(n + g, vec2(127.1, 311.7));
            } else if (d < md2) {
                md2 = d;
            }
        }
    }

    return vec3(sqrt(md), sqrt(md2), id);
}

// ─── Palette ────────────────────────────────────────────────────────────────

vec3 palette(float t, float mode) {
    // Cool ocean
    vec3 a0 = vec3(0.15, 0.3, 0.5);
    vec3 b0 = vec3(0.3, 0.45, 0.5);
    vec3 c0 = vec3(0.3, 0.7, 0.8);

    // Warm mineral
    vec3 a1 = vec3(0.45, 0.2, 0.1);
    vec3 b1 = vec3(0.5, 0.35, 0.2);
    vec3 c1 = vec3(0.9, 0.6, 0.3);

    // Neon
    vec3 a2 = vec3(0.2, 0.1, 0.35);
    vec3 b2 = vec3(0.5, 0.3, 0.6);
    vec3 c2 = vec3(0.2, 0.9, 0.6);

    vec3 a, b, c;
    if (mode < 0.5) {
        float m = mode / 0.5;
        a = mix(a0, a1, m);
        b = mix(b0, b1, m);
        c = mix(c0, c1, m);
    } else {
        float m = (mode - 0.5) / 0.5;
        a = mix(a1, a2, m);
        b = mix(b1, b2, m);
        c = mix(c1, c2, m);
    }

    return a + b * cos(6.2831 * (c * t + vec3(0.0, 0.1, 0.2)));
}

// ─── Main ───────────────────────────────────────────────────────────────────

void main() {
    vec2 uv = isf_FragNormCoord * 2.0 - 1.0;
    uv.x *= RENDERSIZE.x / RENDERSIZE.y;

    float scale = mix(3.0, 8.0, cell_scale);
    float speed = mix(0.3, 1.5, flow_speed);
    float t = TIME * speed;

    // Domain warp for organic distortion
    vec2 warpUV = uv * scale;
    if (warp_amount > 0.01) {
        float w = warp_amount * 1.5;
        warpUV += w * vec2(
            snoise(uv * 2.0 + vec2(t * 0.2, 0.0)),
            snoise(uv * 2.0 + vec2(0.0, t * 0.15))
        );
    }

    // Primary voronoi — large cells
    vec3 v = voronoi(warpUV, t);
    float d1 = v.x;
    float d2 = v.y;
    float cellId = v.z;

    // Secondary voronoi — smaller cells for multi-scale detail
    vec3 v2 = voronoi(warpUV * 2.5 + vec2(3.7, 1.2), t * 0.7);
    float d1b = v2.x;
    float cellId2 = v2.z;

    // Edge detection — both scales
    float edge = 1.0 - smoothstep(0.0, 0.12, d2 - d1);
    edge = pow(edge, mix(2.0, 0.5, edge_glow));
    float edge2 = 1.0 - smoothstep(0.0, 0.08, v2.y - d1b);

    // Cell interior color — wide hue variation between neighbors
    float hue = fract(cellId * 0.0173 + t * 0.05);
    vec3 cellCol = palette(hue, color_mode);

    // Interior: bright centers, dimmer edges — strong gradient but brighter overall
    float interior = smoothstep(0.0, 0.5, 1.0 - d1);
    cellCol *= 0.5 + 0.8 * interior;

    // Secondary cell detail modulates brightness
    float hue2 = fract(cellId2 * 0.031 + t * 0.03);
    float detail = smoothstep(0.0, 0.3, 1.0 - d1b);
    cellCol *= 0.8 + 0.3 * detail;

    // Edge color — bright, contrasting hue
    vec3 edgeCol = palette(hue + 0.35, color_mode) * 1.5;

    vec3 col = mix(cellCol, edgeCol, edge * edge_glow);
    // Fine edge network from secondary scale
    col += palette(hue2 + 0.5, color_mode) * edge2 * edge_glow * 0.4;

    // NO ambient fog — let darks be dark for contrast
    col *= brightness;

    // Subtle vignette
    float vign = 1.0 - dot(uv * 0.35, uv * 0.35);
    col *= clamp(vign, 0.0, 1.0);

    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
