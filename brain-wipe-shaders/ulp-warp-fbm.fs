/*{
    "DESCRIPTION": "FBM domain warp — iterative fractal noise warping noise warping noise. Inigo Quilez-style multi-level turbulence. Organic, roiling, endlessly complex. Three warp passes build on each other for deeply structured distortion.",
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
            "DEFAULT": 0.25,
            "MIN": 0.0,
            "MAX": 1.5,
            "LABEL": "Warp Strength"
        },
        {
            "NAME": "warp_passes",
            "TYPE": "float",
            "DEFAULT": 2.0,
            "MIN": 1.0,
            "MAX": 3.0,
            "LABEL": "Warp Passes (1-3)"
        },
        {
            "NAME": "scale",
            "TYPE": "float",
            "DEFAULT": 2.5,
            "MIN": 0.25,
            "MAX": 8.0,
            "LABEL": "Scale"
        },
        {
            "NAME": "octaves",
            "TYPE": "float",
            "DEFAULT": 5.0,
            "MIN": 1.0,
            "MAX": 8.0,
            "LABEL": "FBM Octaves"
        },
        {
            "NAME": "lacunarity",
            "TYPE": "float",
            "DEFAULT": 2.0,
            "MIN": 1.2,
            "MAX": 4.0,
            "LABEL": "Lacunarity"
        },
        {
            "NAME": "gain",
            "TYPE": "float",
            "DEFAULT": 0.5,
            "MIN": 0.2,
            "MAX": 0.8,
            "LABEL": "Gain"
        },
        {
            "NAME": "drift_speed",
            "TYPE": "float",
            "DEFAULT": 0.08,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Drift Speed"
        },
        {
            "NAME": "drift_angle",
            "TYPE": "float",
            "DEFAULT": 0.0,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Drift Angle"
        },
        {
            "NAME": "rot_per_octave",
            "TYPE": "float",
            "DEFAULT": 0.5,
            "MIN": 0.0,
            "MAX": 1.57,
            "LABEL": "Rotation Per Octave"
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

// --- Noise infrastructure ---

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

mat2 rot2(float a) { return mat2(cos(a), sin(a), -sin(a), cos(a)); }

// FBM: sum of noise octaves with rotation between each
float fbm(vec2 p) {
    float v = 0.0;
    float amplitude = 0.5;
    float freq = 1.0;
    int oct = int(clamp(octaves, 1.0, 8.0));
    mat2 R = rot2(rot_per_octave);
    for (int i = 0; i < 8; i++) {
        if (i >= oct) break;
        v += amplitude * vnoise(p * freq);
        p = R * p + vec2(31.41, 27.18);
        freq *= lacunarity;
        amplitude *= gain;
    }
    return v;
}

// FBM returning a 2D vector for use as displacement
vec2 fbm2(vec2 p) {
    return vec2(fbm(p), fbm(p + vec2(5.2, 1.3)));
}

void main() {
    vec2 uv = isf_FragNormCoord;
    uv.x *= RENDERSIZE.x / RENDERSIZE.y;

    // Drift direction
    float da = drift_angle * 6.28318;
    vec2 drift = vec2(cos(da), sin(da)) * TIME * drift_speed;

    vec2 p = uv * scale + drift;

    // Pass 1: warp space with fbm
    vec2 q = fbm2(p);

    vec2 disp = q;

    // Pass 2: warp warped space
    if (warp_passes >= 1.5) {
        vec2 r = fbm2(p + q * 1.0 + vec2(1.7, 9.2));
        disp = q + r * 0.5;

        // Pass 3: warp twice-warped space
        if (warp_passes >= 2.5) {
            vec2 s = fbm2(p + q * 0.8 + r * 0.6 + vec2(8.3, 2.8));
            disp = q * 0.5 + r * 0.35 + s * 0.15;
        }
    }

    // Sample video with displaced UV
    vec2 sample_uv = isf_FragNormCoord + disp * warp_strength;
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
