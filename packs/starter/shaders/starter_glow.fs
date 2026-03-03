/*{
    "DESCRIPTION": "Edge-detecting glow with color tinting. Applies a Sobel edge filter and adds a colored glow at edges.",
    "CREDIT": "Undersea Lair Project",
    "ISFVSN": "2",
    "CATEGORIES": ["Color", "Stylize"],
    "INPUTS": [
        {
            "NAME": "inputImage",
            "TYPE": "image"
        },
        {
            "NAME": "glow_strength",
            "TYPE": "float",
            "DEFAULT": 0.5,
            "MIN": 0.0,
            "MAX": 1.0
        },
        {
            "NAME": "hue",
            "TYPE": "float",
            "DEFAULT": 0.55,
            "MIN": 0.0,
            "MAX": 1.0
        },
        {
            "NAME": "brightness",
            "TYPE": "float",
            "DEFAULT": 1.0,
            "MIN": 0.5,
            "MAX": 1.5
        }
    ]
}*/

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

float luminance(vec3 c) {
    return dot(c, vec3(0.299, 0.587, 0.114));
}

void main() {
    vec2 uv = isf_FragNormCoord;
    vec4 tex = IMG_NORM_PIXEL(inputImage, uv);
    vec2 px = 1.0 / RENDERSIZE;

    // Sobel 3x3 on luminance
    float tl = luminance(IMG_NORM_PIXEL(inputImage, uv + vec2(-px.x,  px.y)).rgb);
    float tc = luminance(IMG_NORM_PIXEL(inputImage, uv + vec2( 0.0,   px.y)).rgb);
    float tr = luminance(IMG_NORM_PIXEL(inputImage, uv + vec2( px.x,  px.y)).rgb);
    float ml = luminance(IMG_NORM_PIXEL(inputImage, uv + vec2(-px.x,  0.0)).rgb);
    float mr = luminance(IMG_NORM_PIXEL(inputImage, uv + vec2( px.x,  0.0)).rgb);
    float bl = luminance(IMG_NORM_PIXEL(inputImage, uv + vec2(-px.x, -px.y)).rgb);
    float bc = luminance(IMG_NORM_PIXEL(inputImage, uv + vec2( 0.0,  -px.y)).rgb);
    float br = luminance(IMG_NORM_PIXEL(inputImage, uv + vec2( px.x, -px.y)).rgb);

    float gx = -tl - 2.0*ml - bl + tr + 2.0*mr + br;
    float gy = -tl - 2.0*tc - tr + bl + 2.0*bc + br;
    float edge_mag = length(vec2(gx, gy));

    vec3 glow_color = hsv2rgb(vec3(hue, 0.7, 1.0));
    vec3 result = tex.rgb + glow_color * glow_strength * edge_mag;
    result *= brightness;

    gl_FragColor = vec4(clamp(result, 0.0, 1.0), 1.0);
}
