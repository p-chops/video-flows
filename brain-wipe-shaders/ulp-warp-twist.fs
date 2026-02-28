/*{
    "DESCRIPTION": "Radial twist warp — rotates pixels by an angle proportional to their distance from center, creating a spiral distortion. Animated rotation with adjustable falloff.",
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
            "NAME": "twist_amount",
            "TYPE": "float",
            "DEFAULT": 2.0,
            "MIN": 0.0,
            "MAX": 10.0,
            "LABEL": "Twist Amount"
        },
        {
            "NAME": "twist_radius",
            "TYPE": "float",
            "DEFAULT": 0.8,
            "MIN": 0.1,
            "MAX": 2.0,
            "LABEL": "Twist Radius"
        },
        {
            "NAME": "twist_speed",
            "TYPE": "float",
            "DEFAULT": 0.3,
            "MIN": 0.0,
            "MAX": 2.0,
            "LABEL": "Rotation Speed"
        },
        {
            "NAME": "falloff",
            "TYPE": "float",
            "DEFAULT": 2.0,
            "MIN": 0.5,
            "MAX": 6.0,
            "LABEL": "Falloff"
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

void main() {
    vec2 uv = v_texcoord;
    vec2 center = vec2(center_x, center_y);

    // Aspect-correct coordinates
    float aspect = RENDERSIZE.x / RENDERSIZE.y;
    vec2 p = uv - center;
    p.x *= aspect;

    float dist = length(p);

    // Twist angle: strongest at center, falls off with distance
    // Smoothstep gives a nice smooth edge to the twist region
    float strength = 1.0 - smoothstep(0.0, twist_radius, pow(dist, 1.0 / falloff));
    float angle = strength * twist_amount + TIME * twist_speed;

    // Rotate the UV coordinates
    float s = sin(angle);
    float c = cos(angle);
    vec2 rotated = vec2(c * p.x - s * p.y, s * p.x + c * p.y);

    // Back to UV space
    rotated.x /= aspect;
    vec2 warped = rotated + center;

    // Wrap for seamless edges
    warped = fract(warped);

    vec4 col = IMG_NORM_PIXEL(inputImage, warped);

    // Brightness and desaturation
    col.rgb *= brightness;
    float luma = dot(col.rgb, vec3(0.299, 0.587, 0.114));
    col.rgb = mix(col.rgb, vec3(luma), desaturate);

    gl_FragColor = col;
}
