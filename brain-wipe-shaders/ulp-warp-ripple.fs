/*{
    "DESCRIPTION": "Concentric ripple warp — sine-wave displacement radiating from a center point, like stones dropped in water. Animated with time-varying phase.",
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
            "NAME": "amplitude",
            "TYPE": "float",
            "DEFAULT": 0.03,
            "MIN": 0.0,
            "MAX": 0.15,
            "LABEL": "Amplitude"
        },
        {
            "NAME": "frequency",
            "TYPE": "float",
            "DEFAULT": 12.0,
            "MIN": 2.0,
            "MAX": 40.0,
            "LABEL": "Frequency"
        },
        {
            "NAME": "speed",
            "TYPE": "float",
            "DEFAULT": 2.0,
            "MIN": 0.0,
            "MAX": 8.0,
            "LABEL": "Speed"
        },
        {
            "NAME": "decay",
            "TYPE": "float",
            "DEFAULT": 2.0,
            "MIN": 0.0,
            "MAX": 8.0,
            "LABEL": "Decay"
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

    // Vector from center to current pixel
    vec2 delta = uv - center;
    float dist = length(delta);

    // Avoid division by zero
    vec2 dir = dist > 0.001 ? delta / dist : vec2(0.0);

    // Ripple: sinusoidal displacement that decays with distance
    float phase = dist * frequency * PI * 2.0 - TIME * speed;
    float wave = sin(phase);

    // Decay: ripples weaken further from center
    float falloff = exp(-dist * decay);

    // Displace along the radial direction
    vec2 offset = dir * wave * amplitude * falloff;
    vec2 warped = uv + offset;

    // Clamp to valid UV range
    warped = clamp(warped, 0.0, 1.0);

    vec4 col = IMG_NORM_PIXEL(inputImage, warped);

    // Brightness and desaturation
    col.rgb *= brightness;
    float luma = dot(col.rgb, vec3(0.299, 0.587, 0.114));
    col.rgb = mix(col.rgb, vec3(luma), desaturate);

    gl_FragColor = col;
}
