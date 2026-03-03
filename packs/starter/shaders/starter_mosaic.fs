/*{
    "DESCRIPTION": "Pixelation mosaic with color quantization. Reduces the image to chunky blocks with limited color palette.",
    "CREDIT": "Undersea Lair Project",
    "ISFVSN": "2",
    "CATEGORIES": ["Stylize", "Glitch"],
    "INPUTS": [
        {
            "NAME": "inputImage",
            "TYPE": "image"
        },
        {
            "NAME": "block_size",
            "TYPE": "float",
            "DEFAULT": 0.02,
            "MIN": 0.005,
            "MAX": 0.08
        },
        {
            "NAME": "color_levels",
            "TYPE": "float",
            "DEFAULT": 8.0,
            "MIN": 2.0,
            "MAX": 16.0
        }
    ]
}*/

void main() {
    vec2 uv = isf_FragNormCoord;

    // Snap UV to grid
    vec2 grid = floor(uv / block_size) * block_size + block_size * 0.5;
    vec4 tex = IMG_NORM_PIXEL(inputImage, grid);

    // Quantize color
    vec3 col = floor(tex.rgb * color_levels + 0.5) / color_levels;

    gl_FragColor = vec4(col, 1.0);
}
