/*{
    "DESCRIPTION": "Edge detection with style controls. Sobel filter on luminance with adjustable threshold, line width, and color modes.",
    "CREDIT": "Undersea Lair Project",
    "ISFVSN": "2",
    "CATEGORIES": ["Stylize"],
    "INPUTS": [
        {
            "NAME": "inputImage",
            "TYPE": "image"
        },
        {
            "NAME": "threshold",
            "TYPE": "float",
            "DEFAULT": 0.15,
            "MIN": 0.0,
            "MAX": 1.0
        },
        {
            "NAME": "line_width",
            "TYPE": "float",
            "DEFAULT": 1.0,
            "MIN": 0.5,
            "MAX": 4.0
        },
        {
            "NAME": "mix_amount",
            "TYPE": "float",
            "DEFAULT": 0.7,
            "MIN": 0.0,
            "MAX": 1.0
        },
        {
            "NAME": "invert_bg",
            "TYPE": "float",
            "DEFAULT": 0.0,
            "MIN": 0.0,
            "MAX": 1.0
        },
        {
            "NAME": "color_mode",
            "TYPE": "long",
            "DEFAULT": 1,
            "MIN": 0,
            "MAX": 2,
            "LABELS": ["White on black", "Source color", "Luminance"]
        }
    ]
}*/

float luminance(vec3 c) {
    return dot(c, vec3(0.299, 0.587, 0.114));
}

float sampleLuma(vec2 uv) {
    return luminance(IMG_NORM_PIXEL(inputImage, uv).rgb);
}

void main() {
    vec2 uv = isf_FragNormCoord;
    vec2 px = line_width / RENDERSIZE;
    vec4 src = IMG_NORM_PIXEL(inputImage, uv);

    // Sobel 3x3
    float tl = sampleLuma(uv + vec2(-px.x,  px.y));
    float tc = sampleLuma(uv + vec2( 0.0,   px.y));
    float tr = sampleLuma(uv + vec2( px.x,  px.y));
    float ml = sampleLuma(uv + vec2(-px.x,  0.0));
    float mr = sampleLuma(uv + vec2( px.x,  0.0));
    float bl = sampleLuma(uv + vec2(-px.x, -px.y));
    float bc = sampleLuma(uv + vec2( 0.0,  -px.y));
    float br = sampleLuma(uv + vec2( px.x, -px.y));

    float gx = -tl - 2.0*ml - bl + tr + 2.0*mr + br;
    float gy = -tl - 2.0*tc - tr + bl + 2.0*bc + br;
    float edge = length(vec2(gx, gy));

    // Soft threshold with smoothstep
    float t = threshold * 0.5;
    edge = smoothstep(t, t + 0.1, edge);

    // Edge color based on mode
    vec3 edge_color;
    if (color_mode == 0) {
        edge_color = vec3(edge);
    } else if (color_mode == 1) {
        edge_color = src.rgb * edge;
    } else {
        float lum = luminance(src.rgb);
        edge_color = vec3(lum * edge);
    }

    // Background: black or inverted source
    vec3 bg = mix(vec3(0.0), vec3(1.0) - src.rgb, invert_bg);

    // Composite edges over background
    vec3 edge_result = mix(bg, edge_color, edge);

    // Mix with original
    vec3 result = mix(src.rgb, edge_result, mix_amount);

    gl_FragColor = vec4(result, 1.0);
}
