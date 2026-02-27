/*{
    "DESCRIPTION": "Stereographic projection — maps the image onto a sphere and projects it back, creating a 'tiny planet' or fisheye warp. The projection centre drifts slowly, producing continuous spatial distortion.",
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
            "DEFAULT": 0.70,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Intensity"
        },
        {
            "NAME": "curvature",
            "TYPE": "float",
            "DEFAULT": 0.50,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Curvature"
        },
        {
            "NAME": "drift_speed",
            "TYPE": "float",
            "DEFAULT": 0.30,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Drift Speed"
        },
        {
            "NAME": "zoom",
            "TYPE": "float",
            "DEFAULT": 0.50,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Zoom"
        }
    ]
}*/

void main() {
    vec2 uv = isf_FragNormCoord;
    vec3 src = IMG_NORM_PIXEL(inputImage, uv).rgb;

    // Drifting projection centre
    float spd = mix(0.05, 0.3, drift_speed);
    vec2 centre = vec2(
        0.5 + 0.15 * sin(TIME * spd * 0.7),
        0.5 + 0.15 * cos(TIME * spd)
    );

    // Aspect-corrected coordinates centred on projection point
    float aspect = RENDERSIZE.x / RENDERSIZE.y;
    vec2 p = uv - centre;
    p.x *= aspect;

    // Stereographic projection: map plane to sphere and back
    float r = length(p);
    float power = mix(0.5, 3.0, curvature);
    float zoomScale = mix(0.6, 2.0, zoom);

    // Forward stereographic: r -> 2*atan(r) mapped back
    float theta = atan(r * power);
    float newR = tan(theta * 0.5) * 2.0 / power;

    // Remap
    vec2 projected = p;
    if (r > 0.001) {
        projected = p * (newR / r);
    }
    projected *= zoomScale;
    projected.x /= aspect;
    projected += centre;

    // Wrap coordinates
    projected = fract(projected);

    vec3 col = IMG_NORM_PIXEL(inputImage, projected).rgb;
    gl_FragColor = vec4(mix(src, col, intensity), 1.0);
}
