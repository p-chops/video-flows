/*{
    "DESCRIPTION": "Reduces colour depth and spatial resolution in a non-uniform way: the image is quantized to fewer colour levels, and blocks of varying size snap to a coarse pixel grid. Block size oscillates over time so the image breathes between sharp and chunky. A dither pattern breaks up banding.",
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
            "DEFAULT": 0.65,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Intensity"
        },
        {
            "NAME": "crush",
            "TYPE": "float",
            "DEFAULT": 0.50,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Bit Depth Crush"
        },
        {
            "NAME": "block_size",
            "TYPE": "float",
            "DEFAULT": 0.40,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Block Size"
        },
        {
            "NAME": "animate",
            "TYPE": "float",
            "DEFAULT": 0.30,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Animation Speed"
        }
    ]
}*/

float hash2(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123); }

void main() {
    vec2 uv = isf_FragNormCoord;
    vec2 res = RENDERSIZE;

    // Block size oscillates over time — breathes between sharp and chunky
    float tOsc = sin(TIME * mix(0.2, 1.5, animate)) * 0.5 + 0.5;
    float bSize = mix(1.0, mix(4.0, 24.0, block_size), tOsc);

    // Snap UV to block grid
    vec2 blockUV = floor(uv * res / bSize) * bSize / res;

    // Sample at block center
    vec3 col = IMG_NORM_PIXEL(inputImage, blockUV + bSize * 0.5 / res).rgb;

    // Colour quantization — reduce to N levels per channel
    float levels = mix(256.0, 4.0, crush);
    col = floor(col * levels) / levels;

    // Ordered dither to break up banding (Bayer 2x2 approximation)
    vec2 ditherPos = mod(floor(uv * res), 2.0);
    float dither = (ditherPos.x + ditherPos.y * 2.0) / 4.0;
    float ditherStrength = crush * 0.5 / levels;
    col += (dither - 0.375) * ditherStrength;

    // Per-block colour channel shift — some blocks offset one channel
    float blockHash = hash2(floor(uv * res / bSize) + floor(TIME * mix(0.5, 3.0, animate)));
    float channelShift = step(0.85, blockHash) * crush * 0.01;
    col.g = IMG_NORM_PIXEL(inputImage, blockUV + vec2(channelShift, 0.0)).g;
    col.g = floor(col.g * levels) / levels;

    vec3 src = IMG_NORM_PIXEL(inputImage, uv).rgb;
    gl_FragColor = vec4(mix(src, clamp(col, 0.0, 1.0), intensity), 1.0);
}
