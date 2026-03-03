/*{
    "DESCRIPTION": "CRT scanlines with phosphor glow and barrel distortion",
    "CREDIT": "Starter Pack",
    "ISFVSN": "2",
    "CATEGORIES": ["Stylize", "Film"],
    "INPUTS": [
        {
            "NAME": "inputImage",
            "TYPE": "image"
        },
        {
            "NAME": "line_count",
            "TYPE": "float",
            "DEFAULT": 300.0,
            "MIN": 80.0,
            "MAX": 600.0
        },
        {
            "NAME": "line_darkness",
            "TYPE": "float",
            "DEFAULT": 0.4,
            "MIN": 0.0,
            "MAX": 0.8
        },
        {
            "NAME": "barrel",
            "TYPE": "float",
            "DEFAULT": 0.15,
            "MIN": 0.0,
            "MAX": 0.5
        },
        {
            "NAME": "phosphor",
            "TYPE": "float",
            "DEFAULT": 0.3,
            "MIN": 0.0,
            "MAX": 1.0
        },
        {
            "NAME": "roll_speed",
            "TYPE": "float",
            "DEFAULT": 0.0,
            "MIN": 0.0,
            "MAX": 2.0
        }
    ]
}*/

void main() {
    vec2 uv = isf_FragNormCoord;

    // Barrel distortion — push edges outward
    vec2 cent = uv - 0.5;
    float r2 = dot(cent, cent);
    vec2 distorted = uv + cent * r2 * barrel;

    // Vignette from barrel (darken edges)
    float vignette = 1.0 - r2 * barrel * 2.0;

    // Out-of-bounds check
    if (distorted.x < 0.0 || distorted.x > 1.0 || distorted.y < 0.0 || distorted.y > 1.0) {
        gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    vec4 color = IMG_NORM_PIXEL(inputImage, distorted);

    // Phosphor RGB subpixels — per-column tinting
    float col = distorted.x * RENDERSIZE.x;
    float sub = mod(col, 3.0);
    vec3 phosphor_mask = vec3(1.0);
    float p = phosphor * 0.5;
    if (sub < 1.0) {
        phosphor_mask = vec3(1.0 + p, 1.0 - p * 0.5, 1.0 - p * 0.5);
    } else if (sub < 2.0) {
        phosphor_mask = vec3(1.0 - p * 0.5, 1.0 + p, 1.0 - p * 0.5);
    } else {
        phosphor_mask = vec3(1.0 - p * 0.5, 1.0 - p * 0.5, 1.0 + p);
    }

    // Scanline darkening — rolling sine wave
    float scan_y = distorted.y * line_count + TIME * roll_speed * line_count * 0.1;
    float scanline = 1.0 - line_darkness * (0.5 + 0.5 * cos(scan_y * 6.2832));

    color.rgb *= phosphor_mask * scanline * vignette;
    gl_FragColor = vec4(color.rgb, 1.0);
}
