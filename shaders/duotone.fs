/*{
    "DESCRIPTION": "Maps shadows to one color and highlights to another. Classic duotone/split-tone effect for adding mood to monochrome footage.",
    "CREDIT": "ULP",
    "ISFVSN": "2",
    "CATEGORIES": ["Color"],
    "INPUTS": [
        {
            "NAME": "inputImage",
            "TYPE": "image"
        },
        {
            "NAME": "shadow_color",
            "TYPE": "color",
            "LABEL": "Shadow Color",
            "DEFAULT": [0.05, 0.0, 0.3, 1.0]
        },
        {
            "NAME": "highlight_color",
            "TYPE": "color",
            "LABEL": "Highlight Color",
            "DEFAULT": [1.0, 0.6, 0.1, 1.0]
        },
        {
            "NAME": "contrast",
            "TYPE": "float",
            "LABEL": "Contrast",
            "DEFAULT": 1.0,
            "MIN": 0.2,
            "MAX": 3.0
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

    // Apply contrast curve
    luma = clamp((luma - 0.5) * contrast + 0.5, 0.0, 1.0);

    vec3 toned = mix(shadow_color.rgb, highlight_color.rgb, luma);
    gl_FragColor = vec4(mix(src.rgb, toned, mix_amount), src.a);
}
