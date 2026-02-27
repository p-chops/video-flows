/*{
    "DESCRIPTION": "Layered sinusoidal and value-noise UV distortion. Two octaves of low-frequency sine waves create large rolling waves; a turbulence layer adds fine-grain warping. A subtle downward melt bias makes the distortion feel gravity-fed rather than symmetric. An edge-fringe samples at slightly larger displacement to add colour aberration along warp boundaries.",
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
            "DEFAULT": 0.60,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Intensity"
        },
        {
            "NAME": "warp_amount",
            "TYPE": "float",
            "DEFAULT": 0.50,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Warp Amount"
        },
        {
            "NAME": "warp_speed",
            "TYPE": "float",
            "DEFAULT": 0.35,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Warp Speed"
        },
        {
            "NAME": "turbulence",
            "TYPE": "float",
            "DEFAULT": 0.35,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Turbulence"
        }
    ]
}*/

// Cheap 2-D hash used as noise base.
float hash2(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123); }

// Bilinear value noise, smooth C1 via smoothstep interpolation.
float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(hash2(i),                hash2(i + vec2(1.0, 0.0)), f.x),
        mix(hash2(i + vec2(0.0, 1.0)), hash2(i + vec2(1.0, 1.0)), f.x),
        f.y
    );
}

void main() {
    vec2  uv = isf_FragNormCoord;
    float t  = TIME * mix(0.08, 0.45, warp_speed);
    float mx = warp_amount * 0.09;

    // Low-frequency sine warp — two octaves per axis.
    float wx = sin(uv.y * 6.28  + t * 1.1) * 0.5
             + sin(uv.y * 15.7  + t * 0.7) * 0.25;
    float wy = sin(uv.x * 6.28  + t * 0.9) * 0.5
             + sin(uv.x * 14.1  + t * 1.3) * 0.25;

    // High-frequency noise warp.
    float ns = mix(2.5, 7.0, turbulence);
    float nx = noise(uv * ns + vec2(t * 0.4, 0.0)) * 2.0 - 1.0;
    float ny = noise(uv * ns + vec2(0.0, t * 0.4)) * 2.0 - 1.0;

    // Combine with a downward melt bias (positive y = down in screen space).
    vec2 disp;
    disp.x = wx * mx * 0.6 + nx * mx * turbulence * 0.4;
    disp.y = wy * mx * 0.3 + ny * mx * turbulence * 0.2
             + noise(vec2(uv.x * 2.0, t * 0.15)) * mx * 0.45;

    vec3 warped = IMG_NORM_PIXEL(inputImage, clamp(uv + disp, 0.0, 1.0)).rgb;

    // Fringe: sample at slightly larger displacement; blend into red channel
    // proportional to warp magnitude, adding colour aberration at warp edges.
    float mag    = length(disp) / max(mx, 0.0001);
    vec3  fringe = IMG_NORM_PIXEL(inputImage, clamp(uv + disp * 1.1, 0.0, 1.0)).rgb;
    warped.r     = mix(warped.r, fringe.r, mag * 0.45);

    vec3 src = IMG_NORM_PIXEL(inputImage, uv).rgb;
    gl_FragColor = vec4(mix(src, warped, intensity), 1.0);
}
