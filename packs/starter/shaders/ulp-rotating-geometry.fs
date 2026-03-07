/*{
  "DESCRIPTION": "ULP Rotating Geometry — concentric rotating polygons and spirals at variable speeds. Hard-edged, rhythmic, geometric. Fast aggressive motion that contrasts with the slow-drift generators.",
  "CREDIT": "Undersea Lair Project / P-Chops",
  "ISFVSN": "2",
  "CATEGORIES": ["Generator", "Ambient", "ULP"],
  "INPUTS": [
    {
      "NAME": "n_rings",
      "TYPE": "float",
      "DEFAULT": 0.5,
      "MIN": 0.0,
      "MAX": 1.0,
      "LABEL": "Rings  [number of concentric shapes]"
    },
    {
      "NAME": "spin_speed",
      "TYPE": "float",
      "DEFAULT": 0.5,
      "MIN": 0.0,
      "MAX": 1.0,
      "LABEL": "Spin Speed  [rotation rate]"
    },
    {
      "NAME": "shape_complexity",
      "TYPE": "float",
      "DEFAULT": 0.5,
      "MIN": 0.0,
      "MAX": 1.0,
      "LABEL": "Shape  [3=triangle → 12=dodecagon]"
    },
    {
      "NAME": "spiral_twist",
      "TYPE": "float",
      "DEFAULT": 0.3,
      "MIN": 0.0,
      "MAX": 1.0,
      "LABEL": "Spiral Twist  [radial rotation offset]"
    },
    {
      "NAME": "color_mode",
      "TYPE": "float",
      "DEFAULT": 0.4,
      "MIN": 0.0,
      "MAX": 1.0,
      "LABEL": "Color  [0=mono, 0.5=warm, 1=prismatic]"
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

// ─── Polygon SDF ────────────────────────────────────────────────────────────

// Signed distance to a regular polygon with N sides
float polygonSDF(vec2 p, int n, float radius) {
    float an = 3.14159265 / float(n);
    float he = radius * cos(an);

    // Symmetry reduction
    float a = atan(p.y, p.x);
    a = mod(a, 2.0 * an) - an;
    vec2 q = vec2(cos(a), abs(sin(a))) * length(p);

    return q.x - he;
}

// ─── Main ───────────────────────────────────────────────────────────────────

void main() {
    vec2 uv = isf_FragNormCoord * 2.0 - 1.0;
    uv.x *= RENDERSIZE.x / RENDERSIZE.y;

    int rings = int(mix(4.0, 16.0, n_rings));
    float speed = mix(0.3, 2.0, spin_speed);
    int sides = int(mix(3.0, 12.0, shape_complexity));
    float twist = mix(0.0, 3.14159, spiral_twist);

    vec3 col = vec3(0.0);

    for (int i = 0; i < 16; i++) {
        if (i >= rings) break;

        float fi = float(i);
        float fRings = float(rings);

        // Each ring at a different radius
        float radius = 0.15 + fi * (0.85 / fRings);

        // Alternating rotation directions, speed varies per ring
        float dir = (i % 2 == 0) ? 1.0 : -1.0;
        float ringSpeed = speed * (0.5 + fi * 0.15) * dir;

        // Spiral twist — rotate more at larger radii
        float spiralAngle = twist * fi / fRings;

        float angle = TIME * ringSpeed + spiralAngle;
        float ca = cos(angle), sa = sin(angle);
        vec2 rotUV = mat2(ca, -sa, sa, ca) * uv;

        // Polygon SDF at this ring's radius
        float d = polygonSDF(rotUV, sides + (i % 3), radius);

        // Ring band — narrow outline
        float ringWidth = mix(0.015, 0.04, fi / fRings);
        float ring = 1.0 - smoothstep(0.0, ringWidth, abs(d));

        // Wider filled regions — every ring has fill, alternating opacity
        float filled = 1.0 - smoothstep(-0.06, 0.0, d);
        filled *= (i % 2 == 0) ? 0.5 : 0.25;

        // Color per ring
        float hue = fi / fRings;
        vec3 ringCol;

        if (color_mode < 0.33) {
            // Mono — white/gray
            float v = 0.5 + 0.5 * sin(hue * 6.2831 + TIME * 0.5);
            ringCol = vec3(v);
        } else if (color_mode < 0.66) {
            // Warm — amber, orange, red spectrum
            ringCol = vec3(
                0.5 + 0.5 * sin(hue * 6.2831 + 0.0),
                0.3 + 0.3 * sin(hue * 6.2831 + 1.0),
                0.1 + 0.1 * sin(hue * 6.2831 + 2.0)
            );
        } else {
            // Prismatic — full rainbow
            ringCol = vec3(
                0.5 + 0.5 * sin(hue * 6.2831 + 0.0 + TIME * 0.3),
                0.5 + 0.5 * sin(hue * 6.2831 + 2.094 + TIME * 0.3),
                0.5 + 0.5 * sin(hue * 6.2831 + 4.189 + TIME * 0.3)
            );
        }

        col += ringCol * (ring + filled);
    }

    // Dense radial interference — many concentric waves
    float radialDist = length(uv);
    float wave1 = 0.5 + 0.5 * sin(radialDist * 30.0 - TIME * speed * 3.0);
    float wave2 = 0.5 + 0.5 * sin(radialDist * 45.0 + TIME * speed * 2.0);
    float radialPattern = wave1 * wave2;
    radialPattern *= smoothstep(1.5, 0.1, radialDist);
    col += vec3(0.35, 0.2, 0.45) * radialPattern;

    // Angular sectors — dense rotating spoke patterns
    float a = atan(uv.y, uv.x);
    float spokes1 = 0.5 + 0.5 * sin(a * float(sides) * 2.0 + TIME * speed * 1.5);
    float spokes2 = 0.5 + 0.5 * sin(a * float(sides) * 3.0 - TIME * speed * 0.8);
    float spokePattern = spokes1 * spokes2 * smoothstep(1.2, 0.2, radialDist);
    col += vec3(0.3, 0.15, 0.35) * spokePattern;

    // Contrast
    col = smoothstep(0.0, 0.75, col);
    col *= brightness;

    // Vignette
    float vign = 1.0 - dot(uv * 0.4, uv * 0.4);
    col *= clamp(vign, 0.0, 1.0);

    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
