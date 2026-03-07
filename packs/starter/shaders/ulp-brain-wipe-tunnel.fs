/*{
    "DESCRIPTION": "Hypnotic tunnel — infinite geometric corridor rushing toward the viewer. Supports circular and polygonal cross-sections with angular striping.",
    "CREDIT": "ULP Brain Wipe Series",
    "ISFVSN": "2",
    "CATEGORIES": ["Brain Wipe", "ULP"],
    "INPUTS": [
        {
            "NAME": "speed",
            "TYPE": "float",
            "DEFAULT": 0.5,
            "MIN": -3.0,
            "MAX": 3.0,
            "LABEL": "Tunnel Speed"
        },
        {
            "NAME": "rotation_speed",
            "TYPE": "float",
            "DEFAULT": 0.15,
            "MIN": -2.0,
            "MAX": 2.0,
            "LABEL": "Rotation Speed"
        },
        {
            "NAME": "sides",
            "TYPE": "float",
            "DEFAULT": 0.0,
            "MIN": 0.0,
            "MAX": 8.0,
            "LABEL": "Sides (0=circle)"
        },
        {
            "NAME": "depth_freq",
            "TYPE": "float",
            "DEFAULT": 6.0,
            "MIN": 1.0,
            "MAX": 24.0,
            "LABEL": "Depth Rings"
        },
        {
            "NAME": "angular_freq",
            "TYPE": "float",
            "DEFAULT": 8.0,
            "MIN": 1.0,
            "MAX": 32.0,
            "LABEL": "Angular Segments"
        },
        {
            "NAME": "stripe_balance",
            "TYPE": "float",
            "DEFAULT": 0.5,
            "MIN": 0.05,
            "MAX": 0.95,
            "LABEL": "Stripe Balance"
        },
        {
            "NAME": "warp",
            "TYPE": "float",
            "DEFAULT": 0.0,
            "MIN": 0.0,
            "MAX": 1.5,
            "LABEL": "Warp"
        },
        {
            "NAME": "color_hue",
            "TYPE": "float",
            "DEFAULT": 0.6,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Hue"
        },
        {
            "NAME": "saturation",
            "TYPE": "float",
            "DEFAULT": 0.0,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Saturation"
        },
        {
            "NAME": "brightness",
            "TYPE": "float",
            "DEFAULT": 0.85,
            "MIN": 0.1,
            "MAX": 2.0,
            "LABEL": "Brightness"
        },
        {
            "NAME": "invert",
            "TYPE": "bool",
            "DEFAULT": false,
            "LABEL": "Invert"
        }
    ]
}*/

#define PI  3.14159265359
#define TAU 6.28318530718

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
    vec2 uv = isf_FragNormCoord * 2.0 - 1.0;
    uv.x *= RENDERSIZE.x / RENDERSIZE.y;

    // Optional spatial warp
    if (warp > 0.001) {
        float t2 = TIME * 0.3;
        uv.x += warp * sin(uv.y * 2.1 + t2);
        uv.y += warp * cos(uv.x * 1.9 + t2 * 0.7);
    }

    float t = TIME;
    float r = length(uv);
    float a = atan(uv.y, uv.x) + rotation_speed * t;

    // Polygon shaping: smoothly blend circle (sides=0) toward n-gon
    float shape_r = r;
    if (sides >= 2.0) {
        float s = max(sides, 2.0);
        float sector = TAU / s;
        float aa = mod(a, sector) - sector * 0.5;
        shape_r = r * (cos(PI / s) / max(cos(aa), 0.001));
    }

    // Depth coordinate: fract of inverse radius = infinite rush
    float depth = fract(depth_freq * 0.1 / max(shape_r, 0.001) + t * speed);

    // Angular stripe
    float ang = fract((a / TAU + 1.0) * angular_freq * 0.5);

    // XOR checkerboard pattern in polar space — smoothstep for gradient edges
    float d_stripe = smoothstep(stripe_balance - 0.08, stripe_balance + 0.08, depth);
    float a_stripe = smoothstep(stripe_balance - 0.08, stripe_balance + 0.08, ang);
    float v = abs(d_stripe - a_stripe);

    // Depth-tinted hue shift
    float hue_shift = depth * 0.08;

    if (invert > 0.5) v = 1.0 - v;

    vec3 col = hsv2rgb(vec3(fract(color_hue + hue_shift), saturation, v * brightness));

    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
