/*{
    "DESCRIPTION": "Offsets RGB channels spatially in different directions. Creates color fringing and chromatic aberration from monochrome source \u2014 even B&W footage gains color where there are edges and contrast.",
    "CREDIT": "ULP",
    "ISFVSN": "2",
    "CATEGORIES": ["Color", "Distortion"],
    "INPUTS": [
        {
            "NAME": "inputImage",
            "TYPE": "image"
        },
        {
            "NAME": "shift_amount",
            "TYPE": "float",
            "LABEL": "Shift Amount",
            "DEFAULT": 0.008,
            "MIN": 0.0,
            "MAX": 0.05
        },
        {
            "NAME": "angle",
            "TYPE": "float",
            "LABEL": "Angle",
            "DEFAULT": 0.0,
            "MIN": 0.0,
            "MAX": 6.28318
        },
        {
            "NAME": "animate",
            "TYPE": "float",
            "LABEL": "Animation Speed",
            "DEFAULT": 0.0,
            "MIN": 0.0,
            "MAX": 2.0
        }
    ]
}*/

void main() {
    vec2 uv = isf_FragNormCoord;
    float a = angle + TIME * animate;

    vec2 dir = vec2(cos(a), sin(a)) * shift_amount;

    // Each channel samples from a different offset
    float r = IMG_NORM_PIXEL(inputImage, uv + dir).r;
    float g = IMG_NORM_PIXEL(inputImage, uv).g;
    float b = IMG_NORM_PIXEL(inputImage, uv - dir).b;

    gl_FragColor = vec4(r, g, b, 1.0);
}
