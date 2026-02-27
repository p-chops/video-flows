/*{
    "DESCRIPTION": "Voronoi refraction — space is divided into animated cellular regions, each acting as a refracting lens. Video is displaced toward each cell center, as if viewed through an array of glass beads or compound insect eyes. Cell boundaries optionally visible.",
    "CREDIT": "ULP Warp Series",
    "ISFVSN": "2",
    "CATEGORIES": ["Warp", "ULP"],
    "INPUTS": [
        {
            "NAME": "inputImage",
            "TYPE": "image",
            "LABEL": "Video Input"
        },
        {
            "NAME": "warp_strength",
            "TYPE": "float",
            "DEFAULT": 0.3,
            "MIN": 0.0,
            "MAX": 1.5,
            "LABEL": "Warp Strength"
        },
        {
            "NAME": "cell_scale",
            "TYPE": "float",
            "DEFAULT": 5.0,
            "MIN": 1.0,
            "MAX": 20.0,
            "LABEL": "Cell Scale"
        },
        {
            "NAME": "cell_jitter",
            "TYPE": "float",
            "DEFAULT": 0.8,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Cell Jitter"
        },
        {
            "NAME": "animate_speed",
            "TYPE": "float",
            "DEFAULT": 0.15,
            "MIN": 0.0,
            "MAX": 1.5,
            "LABEL": "Cell Animation Speed"
        },
        {
            "NAME": "warp_shape",
            "TYPE": "float",
            "DEFAULT": 0.0,
            "MIN": 0.0,
            "MAX": 2.0,
            "LABEL": "Warp Shape (0=toward center, 1=away, 2=edge-push)"
        },
        {
            "NAME": "edge_width",
            "TYPE": "float",
            "DEFAULT": 0.05,
            "MIN": 0.0,
            "MAX": 0.3,
            "LABEL": "Edge Overlay Width"
        },
        {
            "NAME": "edge_brightness",
            "TYPE": "float",
            "DEFAULT": 0.0,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Edge Overlay"
        },
        {
            "NAME": "edge_color",
            "TYPE": "color",
            "DEFAULT": [1.0, 1.0, 1.0, 1.0],
            "LABEL": "Edge Color"
        },
        {
            "NAME": "brightness",
            "TYPE": "float",
            "DEFAULT": 1.0,
            "MIN": 0.1,
            "MAX": 2.0,
            "LABEL": "Brightness"
        },
        {
            "NAME": "desaturate",
            "TYPE": "float",
            "DEFAULT": 0.0,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Desaturate"
        }
    ]
}*/

vec2 hash2(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)),
             dot(p, vec2(269.5, 183.3)));
    return fract(sin(p) * 43758.5453123);
}

struct VoronoiResult {
    vec2  toCenter;    // displacement from pixel to nearest cell center, already in UV space
    float dist1;       // distance to nearest center (in Voronoi space, for edge detection)
    float dist2;       // distance to second nearest center
};

VoronoiResult voronoi(vec2 p, float aspect, float cscale, float speed) {
    vec2 pi = floor(p);
    vec2 pf = fract(p);

    VoronoiResult res;
    res.dist1 = 1e10;
    res.dist2 = 1e10;
    res.toCenter = vec2(0.0);

    for (int x = -2; x <= 2; x++) {
        for (int y = -2; y <= 2; y++) {
            vec2 neighbor = vec2(float(x), float(y));
            vec2 cell = pi + neighbor;

            // Animated cell center using hash as seed
            vec2 seed = hash2(cell);
            vec2 center = neighbor + cell_jitter * (0.5 + 0.5 * sin(speed * TIME + seed * 6.28318));

            float d = length(center - pf);

            if (d < res.dist1) {
                res.dist2 = res.dist1;
                res.dist1 = d;
                // Displacement from current pixel to cell center, in Voronoi (asuv) space
                vec2 toCenterVoronoi = center - pf;
                // Convert to UV space: undo aspect correction and cell scale
                // asuv.x = uv.x * aspect * cscale  =>  uv.x = asuv.x / (aspect * cscale)
                // asuv.y = uv.y * cscale            =>  uv.y = asuv.y / cscale
                res.toCenter = vec2(toCenterVoronoi.x / (cscale * aspect),
                                    toCenterVoronoi.y / cscale);
            } else if (d < res.dist2) {
                res.dist2 = d;
            }
        }
    }

    return res;
}

void main() {
    vec2 uv = isf_FragNormCoord;

    // Aspect-corrected, cell-scaled space for Voronoi distance metric
    float aspect = RENDERSIZE.x / RENDERSIZE.y;
    vec2 asuv = vec2(uv.x * aspect, uv.y) * cell_scale;

    VoronoiResult vor = voronoi(asuv, aspect, cell_scale, animate_speed);

    // Displacement is already in UV space — no further aspect correction needed
    vec2 disp = vec2(0.0);

    if (warp_shape < 0.5) {
        // Toward cell center: convex lens — each cell magnifies toward its own center
        disp = vor.toCenter * warp_strength;

    } else if (warp_shape < 1.5) {
        // Away from cell center: concave, anti-lens
        disp = -vor.toCenter * warp_strength;

    } else {
        // Edge push: pixels near cell boundaries get shoved toward their cell interior
        float boundary = vor.dist2 - vor.dist1; // 0 at boundary edge, grows toward center
        float boundaryFactor = 1.0 - smoothstep(0.0, 0.3, boundary);
        disp = vor.toCenter * boundaryFactor * warp_strength;
    }

    vec2 sample_uv = clamp(uv + disp, 0.0, 1.0);

    vec4 tex = IMG_NORM_PIXEL(inputImage, sample_uv);
    vec3 col = tex.rgb;

    // Optional cell edge overlay
    if (edge_brightness > 0.001 && edge_width > 0.001) {
        float boundary = vor.dist2 - vor.dist1;
        float edge = 1.0 - smoothstep(0.0, edge_width, boundary);
        col = mix(col, edge_color.rgb, edge * edge_brightness);
    }

    if (desaturate > 0.001) {
        float luma = dot(col, vec3(0.299, 0.587, 0.114));
        col = mix(col, vec3(luma), desaturate);
    }

    col *= brightness;

    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
