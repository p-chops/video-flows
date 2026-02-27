/*{
    "DESCRIPTION": "VHS-style horizontal scan tears. Rows of the image shift sideways in jagged blocks that jump at irregular intervals. Includes scanline darkening and vertical hold drift for full analog damage aesthetic.",
    "CREDIT": "P-Chops / Undersea Lair Project",
    "ISFVSN": "2",
    "INPUTS": [
        {
            "NAME": "inputImage",
            "TYPE": "image"
        },
        {
            "NAME": "intensity",
            "TYPE": "float",
            "DEFAULT": 0.60,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Intensity"
        },
        {
            "NAME": "tear_count",
            "TYPE": "float",
            "DEFAULT": 0.40,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Tear Density"
        },
        {
            "NAME": "drift",
            "TYPE": "float",
            "DEFAULT": 0.30,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Vertical Drift"
        },
        {
            "NAME": "scanlines",
            "TYPE": "float",
            "DEFAULT": 0.40,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Scanline Strength"
        }
    ]
}*/

float hash1(float n) { return fract(sin(n) * 43758.5453123); }

void main() {
    vec2 uv = isf_FragNormCoord;

    // Vertical hold drift — whole image slides up/down slowly
    float vDrift = sin(TIME * 0.3) * drift * 0.05
                 + sin(TIME * 1.7) * drift * 0.02;
    uv.y = fract(uv.y + vDrift);

    // Temporal gate — tears jump at irregular intervals
    float tBlock = floor(TIME * mix(2.0, 8.0, tear_count));

    // Divide screen into horizontal bands
    float bandCount = mix(4.0, 25.0, tear_count);
    float bandIdx = floor(uv.y * bandCount);

    // Each band decides independently whether to tear
    float tearActive = step(1.0 - tear_count * 0.6, hash1(bandIdx * 73.1 + tBlock));

    // Tear offset — coherent per band, changes each time block
    float tearOff = (hash1(bandIdx * 41.3 + tBlock * 17.1) - 0.5) * 0.3 * tearActive;

    // Sub-band jitter — within a tear, rows jitter slightly
    float rowIdx = floor(uv.y * RENDERSIZE.y);
    float rowJitter = (hash1(rowIdx * 13.7 + tBlock) - 0.5) * 0.02 * tearActive;

    vec2 tearUV = vec2(fract(uv.x + tearOff + rowJitter), uv.y);
    vec3 col = IMG_NORM_PIXEL(inputImage, tearUV).rgb;

    // Scanline darkening
    float scanline = 1.0 - scanlines * 0.3 * step(0.5, fract(uv.y * RENDERSIZE.y * 0.5));

    // Slight colour bleed on torn regions — shift red channel
    float bleed = tearActive * 0.003;
    col.r = IMG_NORM_PIXEL(inputImage, vec2(fract(tearUV.x + bleed), tearUV.y)).r;

    vec3 src = IMG_NORM_PIXEL(inputImage, isf_FragNormCoord).rgb;
    gl_FragColor = vec4(mix(src, col * scanline, intensity), 1.0);
}
