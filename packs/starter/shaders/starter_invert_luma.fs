/*{
    "DESCRIPTION": "Selective luminance inversion with mix control. Inverts brightness while preserving hue, blendable with the original.",
    "CREDIT": "Undersea Lair Project",
    "ISFVSN": "2",
    "CATEGORIES": ["Color", "Stylize"],
    "INPUTS": [
        {
            "NAME": "inputImage",
            "TYPE": "image"
        },
        {
            "NAME": "mix_amount",
            "TYPE": "float",
            "DEFAULT": 1.0,
            "MIN": 0.0,
            "MAX": 1.0
        },
        {
            "NAME": "threshold",
            "TYPE": "float",
            "DEFAULT": 0.0,
            "MIN": 0.0,
            "MAX": 1.0
        }
    ]
}*/

void main() {
    vec2 uv = isf_FragNormCoord;
    vec4 tex = IMG_NORM_PIXEL(inputImage, uv);

    float luma = dot(tex.rgb, vec3(0.299, 0.587, 0.114));

    // Only invert pixels above the threshold
    float mask = smoothstep(threshold - 0.05, threshold + 0.05, luma);
    vec3 inverted = vec3(1.0) - tex.rgb;
    vec3 result = mix(tex.rgb, mix(tex.rgb, inverted, mask), mix_amount);

    gl_FragColor = vec4(result, 1.0);
}
