/*{
    "DESCRIPTION": "RGB channel offset. Splits the image into its color channels and shifts them apart for a chromatic aberration look.",
    "CREDIT": "Undersea Lair Project",
    "ISFVSN": "2",
    "CATEGORIES": ["Color", "Glitch"],
    "INPUTS": [
        {
            "NAME": "inputImage",
            "TYPE": "image"
        },
        {
            "NAME": "amount",
            "TYPE": "float",
            "DEFAULT": 0.01,
            "MIN": 0.0,
            "MAX": 0.05
        },
        {
            "NAME": "angle",
            "TYPE": "float",
            "DEFAULT": 0.0,
            "MIN": 0.0,
            "MAX": 6.28318
        }
    ]
}*/

void main() {
    vec2 uv = isf_FragNormCoord;

    vec2 dir = vec2(cos(angle), sin(angle)) * amount;

    float r = IMG_NORM_PIXEL(inputImage, uv + dir).r;
    float g = IMG_NORM_PIXEL(inputImage, uv).g;
    float b = IMG_NORM_PIXEL(inputImage, uv - dir).b;

    gl_FragColor = vec4(r, g, b, 1.0);
}
