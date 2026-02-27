/*{
    "DESCRIPTION": "Glitch block displacement: the image is divided into irregular blocks that shift horizontally and vertically at random intervals. Blocks snap to new positions abruptly, with optional colour channel separation per block. Think JPEG corruption meets datamosh.",
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
            "NAME": "block_density",
            "TYPE": "float",
            "DEFAULT": 0.50,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Block Density"
        },
        {
            "NAME": "shift_amount",
            "TYPE": "float",
            "DEFAULT": 0.50,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Shift Amount"
        },
        {
            "NAME": "channel_split",
            "TYPE": "float",
            "DEFAULT": 0.30,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Channel Split"
        }
    ]
}*/

float hash2(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

void main() {
    vec2 uv = isf_FragNormCoord;
    vec3 src = IMG_NORM_PIXEL(inputImage, uv).rgb;

    // Temporal gate — blocks jump at irregular intervals
    float tRate = mix(1.5, 10.0, block_density);
    float tBlock = floor(TIME * tRate);

    // Non-uniform block grid — different density for X and Y
    float gridX = mix(4.0, 20.0, block_density);
    float gridY = mix(3.0, 15.0, block_density);

    // Jitter the grid slightly to break regularity
    float gJitterX = hash2(vec2(tBlock, 0.0)) * 0.3;
    float gJitterY = hash2(vec2(0.0, tBlock)) * 0.3;
    vec2 blockPos = floor(vec2(uv.x * (gridX + gJitterX * gridX),
                               uv.y * (gridY + gJitterY * gridY)));

    // Each block decides independently whether to shift
    float isActive = step(0.55 - block_density * 0.3,
                        hash2(blockPos + tBlock * 7.13));

    // Shift offset per block — coherent within block, random per time step
    float shiftX = (hash2(blockPos * 31.7 + tBlock * 3.1) - 0.5)
                   * shift_amount * 0.4 * isActive;
    float shiftY = (hash2(blockPos * 17.3 + tBlock * 5.7) - 0.5)
                   * shift_amount * 0.15 * isActive;

    vec2 shifted = fract(uv + vec2(shiftX, shiftY));

    // Per-channel displacement for glitch colour fringing
    float csOff = channel_split * 0.02 * isActive;
    float r = IMG_NORM_PIXEL(inputImage, fract(shifted + vec2(csOff, 0.0))).r;
    float g = IMG_NORM_PIXEL(inputImage, shifted).g;
    float b = IMG_NORM_PIXEL(inputImage, fract(shifted - vec2(csOff, 0.0))).b;
    vec3 col = vec3(r, g, b);

    // Occasional full-row glitch — thin horizontal stripe that shifts hard
    float rowIdx = floor(uv.y * RENDERSIZE.y);
    float rowHash = hash2(vec2(rowIdx * 0.37, tBlock * 13.0));
    float rowActive = step(0.97 - block_density * 0.05, rowHash);
    float rowShift = (rowHash - 0.5) * 0.5 * rowActive;
    if (rowActive > 0.5) {
        vec2 rowUV = fract(vec2(uv.x + rowShift, uv.y));
        col = IMG_NORM_PIXEL(inputImage, rowUV).rgb;
    }

    gl_FragColor = vec4(mix(src, col, intensity), 1.0);
}
