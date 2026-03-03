/*{
    "DESCRIPTION": "Sine-wave domain warping — animated displacement field",
    "CREDIT": "Starter Pack",
    "ISFVSN": "2",
    "CATEGORIES": ["Warp", "Brain Wipe"],
    "INPUTS": [
        {
            "NAME": "inputImage",
            "TYPE": "image"
        },
        {
            "NAME": "amplitude",
            "TYPE": "float",
            "DEFAULT": 0.03,
            "MIN": 0.0,
            "MAX": 0.15
        },
        {
            "NAME": "frequency",
            "TYPE": "float",
            "DEFAULT": 4.0,
            "MIN": 1.0,
            "MAX": 20.0
        },
        {
            "NAME": "speed",
            "TYPE": "float",
            "DEFAULT": 0.5,
            "MIN": 0.05,
            "MAX": 2.0
        },
        {
            "NAME": "layers",
            "TYPE": "float",
            "DEFAULT": 3.0,
            "MIN": 1.0,
            "MAX": 6.0
        }
    ]
}*/

void main() {
    vec2 uv = isf_FragNormCoord;
    float t = TIME * speed;

    // Layered sine displacement — each layer at different angle and frequency
    vec2 offset = vec2(0.0);
    for (float i = 0.0; i < 6.0; i++) {
        if (i >= layers) break;
        float angle = i * 2.39996 + t * 0.3; // golden angle rotation per layer
        float f = frequency * (1.0 + i * 0.7);
        vec2 dir = vec2(cos(angle), sin(angle));
        float wave = sin(dot(uv, dir) * f * 6.2832 + t * (1.0 + i * 0.4));
        // Perpendicular displacement
        offset += vec2(-dir.y, dir.x) * wave * amplitude / (1.0 + i * 0.5);
    }

    vec2 warped = uv + offset;
    gl_FragColor = IMG_NORM_PIXEL(inputImage, warped);
}
