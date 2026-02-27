/*{
    "DESCRIPTION": "Curl noise flow — divergence-free vector field derived from the curl of a noise function. Video is advected through the field, producing swirling, fluid-like distortion with no tearing or compression artifacts.",
    "CREDIT": "ULP Warp Series",
    "ISFVSN": "2",
    "CATEGORIES": ["Warp", "ULP"],
    "INPUTS": [
        {
            "NAME": "inputImage",
            "TYPE": "image",
            "LABEL": "Video Input"
        },
        {
            "NAME": "warp_strength",
            "TYPE": "float",
            "DEFAULT": 0.2,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Warp Strength"
        },
        {
            "NAME": "scale",
            "TYPE": "float",
            "DEFAULT": 2.0,
            "MIN": 0.25,
            "MAX": 8.0,
            "LABEL": "Scale"
        },
        {
            "NAME": "flow_speed",
            "TYPE": "float",
            "DEFAULT": 0.12,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Flow Speed"
        },
        {
            "NAME": "flow_angle",
            "TYPE": "float",
            "DEFAULT": 0.0,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Flow Angle"
        },
        {
            "NAME": "octaves",
            "TYPE": "float",
            "DEFAULT": 3.0,
            "MIN": 1.0,
            "MAX": 5.0,
            "LABEL": "Octaves"
        },
        {
            "NAME": "swirl",
            "TYPE": "float",
            "DEFAULT": 0.3,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Swirl Bias"
        },
        {
            "NAME": "eps_scale",
            "TYPE": "float",
            "DEFAULT": 0.5,
            "MIN": 0.05,
            "MAX": 2.0,
            "LABEL": "Field Granularity"
        },
        {
            "NAME": "time_offset",
            "TYPE": "float",
            "DEFAULT": 0.0,
            "MIN": 0.0,
            "MAX": 100.0,
            "LABEL": "Time Offset"
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

vec2 hash2(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)),
             dot(p, vec2(269.5, 183.3)));
    return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
}

float vnoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(dot(hash2(i + vec2(0,0)), f - vec2(0,0)),
                   dot(hash2(i + vec2(1,0)), f - vec2(1,0)), u.x),
               mix(dot(hash2(i + vec2(0,1)), f - vec2(0,1)),
                   dot(hash2(i + vec2(1,1)), f - vec2(1,1)), u.x), u.y);
}

// Curl of 2D scalar noise field: (∂f/∂y, -∂f/∂x)
// Divergence-free by construction — no stretching or tearing
vec2 curlNoise(vec2 p, float eps) {
    float n_py = vnoise(p + vec2(0.0, eps));
    float n_my = vnoise(p - vec2(0.0, eps));
    float n_px = vnoise(p + vec2(eps, 0.0));
    float n_mx = vnoise(p - vec2(eps, 0.0));
    float dfdy = (n_py - n_my) / (2.0 * eps);
    float dfdx = (n_px - n_mx) / (2.0 * eps);
    return vec2(dfdy, -dfdx);
}

void main() {
    vec2 uv = isf_FragNormCoord;
    uv.x *= RENDERSIZE.x / RENDERSIZE.y;

    float t = (TIME + time_offset) * flow_speed;
    float fa = flow_angle * 6.28318;
    vec2 drift = vec2(cos(fa), sin(fa)) * t;

    vec2 p = uv * scale + drift;
    float eps = 0.008 * eps_scale;

    // Multi-octave curl accumulation
    vec2 curl = vec2(0.0);
    float amplitude = 1.0;
    float freq = 1.0;
    int oct = int(clamp(octaves, 1.0, 5.0));

    for (int i = 0; i < 5; i++) {
        if (i >= oct) break;
        // Offset each octave spatially so they don't self-cancel
        vec2 offset = vec2(float(i) * 31.41, float(i) * 17.31);
        curl += amplitude * curlNoise(p * freq + offset, eps * freq);
        freq *= 2.0;
        amplitude *= 0.5;
    }

    // Swirl bias: adds a global rotational component to the flow
    if (swirl > 0.001) {
        vec2 centered = uv * 2.0 - 1.0;
        vec2 radialSwirl = vec2(-centered.y, centered.x); // perpendicular = rotation
        curl += radialSwirl * swirl;
    }

    // Displace video UV by curl field
    vec2 sample_uv = isf_FragNormCoord + curl * warp_strength;
    sample_uv = clamp(sample_uv, 0.0, 1.0);

    vec4 tex = IMG_NORM_PIXEL(inputImage, sample_uv);
    vec3 col = tex.rgb;

    if (desaturate > 0.001) {
        float luma = dot(col, vec3(0.299, 0.587, 0.114));
        col = mix(col, vec3(luma), desaturate);
    }

    col *= brightness;

    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
