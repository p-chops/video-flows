/*{
    "DESCRIPTION": "Kaleidoscope warp — angular mirror symmetry that turns video into mandala-like patterns. Adjustable segment count, rotation, and zoom.",
    "CREDIT": "ULP Warp Series",
    "ISFVSN": "2",
    "CATEGORIES": ["Warp", "Brain Wipe"],
    "INPUTS": [
        {
            "NAME": "inputImage",
            "TYPE": "image",
            "LABEL": "Video Input"
        },
        {
            "NAME": "segments",
            "TYPE": "float",
            "DEFAULT": 6.0,
            "MIN": 2.0,
            "MAX": 16.0,
            "LABEL": "Segments"
        },
        {
            "NAME": "rotation",
            "TYPE": "float",
            "DEFAULT": 0.1,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Rotation Speed"
        },
        {
            "NAME": "zoom_amount",
            "TYPE": "float",
            "DEFAULT": 1.0,
            "MIN": 0.5,
            "MAX": 3.0,
            "LABEL": "Zoom"
        },
        {
            "NAME": "center_x",
            "TYPE": "float",
            "DEFAULT": 0.5,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Center X"
        },
        {
            "NAME": "center_y",
            "TYPE": "float",
            "DEFAULT": 0.5,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Center Y"
        },
        {
            "NAME": "brightness",
            "TYPE": "float",
            "DEFAULT": 1.0,
            "MIN": 0.1,
            "MAX": 2.0,
            "LABEL": "Brightness"
        },
        {
            "NAME": "desaturate",
            "TYPE": "float",
            "DEFAULT": 0.0,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Desaturate"
        }
    ]
}*/

const float PI = 3.14159265359;
const float TAU = 6.28318530718;

void main() {
    vec2 uv = v_texcoord;
    vec2 center = vec2(center_x, center_y);

    // Convert to polar coordinates relative to center
    vec2 p = (uv - center) * zoom_amount;
    float r = length(p);
    float angle = atan(p.y, p.x);

    // Add rotation over time
    angle += TIME * rotation * TAU;

    // Kaleidoscope: fold the angle into one segment, then mirror
    float seg_angle = TAU / segments;
    angle = mod(angle, seg_angle);
    // Mirror every other segment for seamless symmetry
    if (angle > seg_angle * 0.5) {
        angle = seg_angle - angle;
    }

    // Back to cartesian, then to UV space
    vec2 warped = center + vec2(cos(angle), sin(angle)) * r;

    // Wrap UV to stay in valid range
    warped = fract(warped);

    vec4 col = IMG_NORM_PIXEL(inputImage, warped);

    // Brightness and desaturation
    col.rgb *= brightness;
    float luma = dot(col.rgb, vec3(0.299, 0.587, 0.114));
    col.rgb = mix(col.rgb, vec3(luma), desaturate);

    gl_FragColor = col;
}
