/*{
    "DESCRIPTION": "Copper oxide patina. Warm amber-brown for highlights, cool verdigris teal-green for shadows. Color varies with subtle spatial noise for an organic, corroded-metal look.",
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
            "DEFAULT": 0.75,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Intensity"
        },
        {
            "NAME": "patina",
            "TYPE": "float",
            "DEFAULT": 0.50,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Patina Amount"
        },
        {
            "NAME": "texture_scale",
            "TYPE": "float",
            "DEFAULT": 0.40,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Texture Scale"
        }
    ]
}*/

float noiseOx(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

float smoothNoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = noiseOx(i);
    float b = noiseOx(i + vec2(1.0, 0.0));
    float c = noiseOx(i + vec2(0.0, 1.0));
    float d = noiseOx(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

void main() {
    vec2 uv = isf_FragNormCoord;
    vec3 src = IMG_NORM_PIXEL(inputImage, uv).rgb;
    float luma = dot(src, vec3(0.299, 0.587, 0.114));

    // Spatial noise — organic variation in coloring
    float scale = mix(3.0, 20.0, texture_scale);
    float n = smoothNoise(uv * scale + TIME * 0.02);
    float n2 = smoothNoise(uv * scale * 2.3 + vec2(50.0, 80.0));

    // Patina shifts the crossover point — more patina means more green
    float crossover = mix(0.55, 0.3, patina);

    // Noise modulates the crossover locally
    float localCross = crossover + (n - 0.5) * 0.15;

    // Copper / amber for highlights
    vec3 copper = vec3(0.75, 0.5, 0.25);
    vec3 amber = vec3(0.9, 0.65, 0.3);
    vec3 warm = mix(copper, amber, n2);

    // Verdigris / teal for shadows — bright enough to read
    vec3 verdigris = vec3(0.25, 0.55, 0.48);
    vec3 teal = vec3(0.18, 0.48, 0.44);
    vec3 cool = mix(verdigris, teal, n);

    // Shadow base — lifted, still tinted
    vec3 dark = vec3(0.12, 0.2, 0.18);

    // Blend based on luminance
    vec3 col;
    if (luma < localCross * 0.5) {
        col = mix(dark, cool, luma / (localCross * 0.5));
    } else if (luma < localCross) {
        col = mix(cool, warm, (luma - localCross * 0.5) / (localCross * 0.5));
    } else {
        col = mix(warm, vec3(0.95, 0.9, 0.8), (luma - localCross) / (1.0 - localCross));
    }

    gl_FragColor = vec4(mix(src, col, intensity), 1.0);
}
