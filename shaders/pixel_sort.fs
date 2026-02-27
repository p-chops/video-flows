/*{
    "DESCRIPTION": "Luminance-driven horizontal smear / sort approximation. The image is divided into randomised scanline bands; within active bands, pixels are displaced proportional to local luma — bright pixels streak further than dark ones, approximating the look of a real pixel-sort without multi-pass sorting. A secondary long-range streak is added for high-luma regions.",
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
            "DEFAULT": 0.70,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Intensity"
        },
        {
            "NAME": "coverage",
            "TYPE": "float",
            "DEFAULT": 0.35,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Band Coverage"
        },
        {
            "NAME": "stretch",
            "TYPE": "float",
            "DEFAULT": 0.50,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Sort Stretch"
        }
    ]
}*/

float hash1(float n) { return fract(sin(n) * 43758.5453123); }

void main() {
    vec2  uv   = isf_FragNormCoord;
    vec4  src  = IMG_NORM_PIXEL(inputImage, uv);
    float luma = dot(src.rgb, vec3(0.299, 0.587, 0.114));

    // Band partitioning — denser bands at higher coverage.
    float bandCount = mix(6.0, 70.0, coverage);
    float bandIdx   = floor(uv.y * bandCount);
    float bandTime  = floor(TIME * mix(2.0, 14.0, intensity));
    float isActive  = step(1.0 - coverage, hash1(bandIdx * 113.7 + bandTime));

    // Each band independently picks a sort direction (left or right).
    float dir = sign(hash1(bandIdx * 71.3 + floor(TIME * 3.0)) - 0.5);

    // Smooth threshold so only mid–high luma pixels streak.
    float threshold = 0.35;
    float pull = smoothstep(threshold, threshold + 0.25, luma)
                 * stretch * 0.45 * dir * isActive;

    vec3 sorted = IMG_NORM_PIXEL(inputImage, clamp(vec2(uv.x + pull, uv.y), 0.0, 1.0)).rgb;

    // Secondary long-range streak for the bright tail of a sorted run.
    float longPull   = pull * 1.8;
    vec3  longStreak = IMG_NORM_PIXEL(inputImage, clamp(vec2(uv.x + longPull, uv.y), 0.0, 1.0)).rgb;

    float blendA = isActive * smoothstep(threshold, threshold + 0.3, luma);
    float blendB = isActive * smoothstep(0.6, 0.9, luma);

    vec3 result = mix(src.rgb, sorted,     blendA * intensity);
    result      = mix(result,  longStreak, blendB * intensity * 0.6);

    gl_FragColor = vec4(result, 1.0);
}
