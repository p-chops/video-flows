/*{
    "DESCRIPTION": "Detects edges via Sobel filter and renders them as colored glow over the darkened source. Edges become luminous color lines \u2014 great for giving B&W footage a neon-wireframe look.",
    "CREDIT": "ULP",
    "ISFVSN": "2",
    "CATEGORIES": ["Color", "Stylize"],
    "INPUTS": [
        {
            "NAME": "inputImage",
            "TYPE": "image"
        },
        {
            "NAME": "glow_color",
            "TYPE": "color",
            "LABEL": "Glow Color",
            "DEFAULT": [0.2, 0.8, 1.0, 1.0]
        },
        {
            "NAME": "edge_strength",
            "TYPE": "float",
            "LABEL": "Edge Strength",
            "DEFAULT": 2.0,
            "MIN": 0.5,
            "MAX": 8.0
        },
        {
            "NAME": "darken",
            "TYPE": "float",
            "LABEL": "Background Darken",
            "DEFAULT": 0.3,
            "MIN": 0.0,
            "MAX": 1.0
        },
        {
            "NAME": "color_by_angle",
            "TYPE": "float",
            "LABEL": "Color by Edge Angle",
            "DEFAULT": 0.0,
            "MIN": 0.0,
            "MAX": 1.0
        }
    ]
}*/

float luma(vec3 c) {
    return dot(c, vec3(0.299, 0.587, 0.114));
}

void main() {
    vec2 uv = isf_FragNormCoord;
    vec2 px = 1.0 / RENDERSIZE;

    // Sobel kernel sampling
    float tl = luma(IMG_NORM_PIXEL(inputImage, uv + vec2(-px.x,  px.y)).rgb);
    float t  = luma(IMG_NORM_PIXEL(inputImage, uv + vec2( 0.0,   px.y)).rgb);
    float tr = luma(IMG_NORM_PIXEL(inputImage, uv + vec2( px.x,  px.y)).rgb);
    float l  = luma(IMG_NORM_PIXEL(inputImage, uv + vec2(-px.x,  0.0)).rgb);
    float r  = luma(IMG_NORM_PIXEL(inputImage, uv + vec2( px.x,  0.0)).rgb);
    float bl = luma(IMG_NORM_PIXEL(inputImage, uv + vec2(-px.x, -px.y)).rgb);
    float b  = luma(IMG_NORM_PIXEL(inputImage, uv + vec2( 0.0,  -px.y)).rgb);
    float br = luma(IMG_NORM_PIXEL(inputImage, uv + vec2( px.x, -px.y)).rgb);

    float gx = -tl - 2.0*l - bl + tr + 2.0*r + br;
    float gy = -tl - 2.0*t - tr + bl + 2.0*b + br;
    float edge = clamp(length(vec2(gx, gy)) * edge_strength, 0.0, 1.0);

    // Optionally color edges by their angle
    float angle_hue = atan(gy, gx) / 6.28318 + 0.5;
    vec3 angle_color = 0.5 + 0.5 * cos(6.28318 * (angle_hue + vec3(0.0, 0.33, 0.67)));
    vec3 final_glow = mix(glow_color.rgb, angle_color, color_by_angle);

    vec4 src = IMG_NORM_PIXEL(inputImage, uv);
    // Cap darkening to 50% so stacking doesn't crush to black
    vec3 darkened = src.rgb * (1.0 - darken * 0.5);

    // Additive glow on edges
    vec3 result = darkened + final_glow * edge;
    gl_FragColor = vec4(result, src.a);
}
