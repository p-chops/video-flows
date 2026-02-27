/*{
    "DESCRIPTION": "Posterization — reduces tonal range to flat bands. Pairs well with edge detection for a graphic/woodcut look. Adjustable number of levels with optional edge darkening built in.",
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
            "DEFAULT": 0.80,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Intensity"
        },
        {
            "NAME": "levels",
            "TYPE": "float",
            "DEFAULT": 0.30,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Levels"
        },
        {
            "NAME": "edge_ink",
            "TYPE": "float",
            "DEFAULT": 0.40,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Edge Ink"
        },
        {
            "NAME": "gamma",
            "TYPE": "float",
            "DEFAULT": 0.50,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Gamma"
        }
    ]
}*/

void main() {
    vec2 uv = isf_FragNormCoord;
    vec3 src = IMG_NORM_PIXEL(inputImage, uv).rgb;

    // Number of quantization levels: 2 (silhouette) to 12 (subtle)
    float n = floor(mix(2.0, 12.0, levels));

    // Gamma curve before quantization — shapes the band distribution
    float g = mix(0.5, 2.0, gamma);
    vec3 curved = pow(src, vec3(g));

    // Quantize
    vec3 poster = floor(curved * n + 0.5) / n;

    // Undo gamma so output matches source brightness range
    poster = pow(poster, vec3(1.0 / g));

    // Edge detection — Sobel-like luminance gradient
    if (edge_ink > 0.01) {
        float px = 1.0 / RENDERSIZE.x;
        float py = 1.0 / RENDERSIZE.y;

        float tl = dot(IMG_NORM_PIXEL(inputImage, uv + vec2(-px,  py)).rgb, vec3(0.299, 0.587, 0.114));
        float t  = dot(IMG_NORM_PIXEL(inputImage, uv + vec2(0.0,  py)).rgb, vec3(0.299, 0.587, 0.114));
        float tr = dot(IMG_NORM_PIXEL(inputImage, uv + vec2( px,  py)).rgb, vec3(0.299, 0.587, 0.114));
        float l  = dot(IMG_NORM_PIXEL(inputImage, uv + vec2(-px, 0.0)).rgb, vec3(0.299, 0.587, 0.114));
        float r  = dot(IMG_NORM_PIXEL(inputImage, uv + vec2( px, 0.0)).rgb, vec3(0.299, 0.587, 0.114));
        float bl = dot(IMG_NORM_PIXEL(inputImage, uv + vec2(-px, -py)).rgb, vec3(0.299, 0.587, 0.114));
        float b  = dot(IMG_NORM_PIXEL(inputImage, uv + vec2(0.0, -py)).rgb, vec3(0.299, 0.587, 0.114));
        float br = dot(IMG_NORM_PIXEL(inputImage, uv + vec2( px, -py)).rgb, vec3(0.299, 0.587, 0.114));

        float gx = -tl - 2.0*l - bl + tr + 2.0*r + br;
        float gy = -tl - 2.0*t - tr + bl + 2.0*b + br;
        float edge = sqrt(gx*gx + gy*gy);

        // Darken edges — ink lines
        float ink = smoothstep(0.05, 0.3, edge) * edge_ink;
        poster *= (1.0 - ink);
    }

    gl_FragColor = vec4(mix(src, poster, intensity), 1.0);
}
