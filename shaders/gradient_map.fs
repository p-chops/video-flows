/*{
    "DESCRIPTION": "Three-point gradient map: assigns separate colors to shadows, midtones, and highlights. Classic darkroom-style toning with full control over each tonal zone.",
    "CREDIT": "ULP",
    "ISFVSN": "2",
    "CATEGORIES": ["Color"],
    "INPUTS": [
        {
            "NAME": "inputImage",
            "TYPE": "image"
        },
        {
            "NAME": "shadows",
            "TYPE": "color",
            "LABEL": "Shadows",
            "DEFAULT": [0.0, 0.0, 0.2, 1.0]
        },
        {
            "NAME": "midtones",
            "TYPE": "color",
            "LABEL": "Midtones",
            "DEFAULT": [0.6, 0.1, 0.3, 1.0]
        },
        {
            "NAME": "highlights",
            "TYPE": "color",
            "LABEL": "Highlights",
            "DEFAULT": [1.0, 0.9, 0.6, 1.0]
        },
        {
            "NAME": "mid_point",
            "TYPE": "float",
            "LABEL": "Midtone Position",
            "DEFAULT": 0.5,
            "MIN": 0.1,
            "MAX": 0.9
        },
        {
            "NAME": "mix_amount",
            "TYPE": "float",
            "LABEL": "Mix",
            "DEFAULT": 1.0,
            "MIN": 0.0,
            "MAX": 1.0
        }
    ]
}*/

void main() {
    vec4 src = IMG_NORM_PIXEL(inputImage, isf_FragNormCoord);
    float luma = dot(src.rgb, vec3(0.299, 0.587, 0.114));

    vec3 mapped;
    if (luma < mid_point) {
        mapped = mix(shadows.rgb, midtones.rgb, luma / mid_point);
    } else {
        mapped = mix(midtones.rgb, highlights.rgb, (luma - mid_point) / (1.0 - mid_point));
    }

    gl_FragColor = vec4(mix(src.rgb, mapped, mix_amount), src.a);
}
