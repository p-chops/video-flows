/*{
    "DESCRIPTION": "Duotone — map luminance to two colors with contrast curve",
    "CREDIT": "Starter Pack",
    "ISFVSN": "2",
    "CATEGORIES": ["Color"],
    "INPUTS": [
        {
            "NAME": "inputImage",
            "TYPE": "image"
        },
        {
            "NAME": "shadow_hue",
            "TYPE": "float",
            "DEFAULT": 0.6,
            "MIN": 0.0,
            "MAX": 1.0
        },
        {
            "NAME": "highlight_hue",
            "TYPE": "float",
            "DEFAULT": 0.1,
            "MIN": 0.0,
            "MAX": 1.0
        },
        {
            "NAME": "saturation",
            "TYPE": "float",
            "DEFAULT": 0.7,
            "MIN": 0.0,
            "MAX": 1.0
        },
        {
            "NAME": "contrast",
            "TYPE": "float",
            "DEFAULT": 1.2,
            "MIN": 0.5,
            "MAX": 3.0
        },
        {
            "NAME": "mix_amount",
            "TYPE": "float",
            "DEFAULT": 1.0,
            "MIN": 0.0,
            "MAX": 1.0
        }
    ]
}*/

vec3 hsv2rgb(vec3 c) {
    vec3 p = abs(fract(c.xxx + vec3(1.0, 2.0/3.0, 1.0/3.0)) * 6.0 - 3.0);
    return c.z * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), c.y);
}

void main() {
    vec2 uv = isf_FragNormCoord;
    vec4 src = IMG_NORM_PIXEL(inputImage, uv);

    // Luminance
    float luma = dot(src.rgb, vec3(0.2126, 0.7152, 0.0722));

    // Contrast curve — power function centered at 0.5
    luma = pow(luma, contrast);
    luma = clamp(luma, 0.0, 1.0);

    // Shadow and highlight colors from hue
    vec3 shadow = hsv2rgb(vec3(shadow_hue, saturation, 0.25));
    vec3 highlight = hsv2rgb(vec3(highlight_hue, saturation * 0.6, 1.0));

    // Map luminance to duotone
    vec3 toned = mix(shadow, highlight, luma);

    // Blend with original
    vec3 result = mix(src.rgb, toned, mix_amount);
    gl_FragColor = vec4(result, 1.0);
}
