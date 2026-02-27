/*{
    "DESCRIPTION": "Bruise tones — deep purples, sick yellows, muted reds. The palette of damaged tissue and old film stock. Shadows go purple-black, midtones get sickly warmth, highlights yellow out.",
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
            "NAME": "age",
            "TYPE": "float",
            "DEFAULT": 0.50,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Age"
        },
        {
            "NAME": "sickness",
            "TYPE": "float",
            "DEFAULT": 0.40,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Sickness"
        }
    ]
}*/

void main() {
    vec2 uv = isf_FragNormCoord;
    vec3 src = IMG_NORM_PIXEL(inputImage, uv).rgb;
    float luma = dot(src, vec3(0.299, 0.587, 0.114));

    // Age shifts the palette: fresh bruise (purple-red) → old bruise (yellow-green)
    // Fresh: deep purple shadows, red-brown mids, pale highlights
    // Old: blue-black shadows, sickly yellow-green mids, jaundiced highlights

    // Shadow color — lifted so stacking doesn't crush
    vec3 shadowFresh = vec3(0.25, 0.1, 0.32);    // muted purple
    vec3 shadowOld = vec3(0.15, 0.12, 0.22);      // dusky blue
    vec3 shadow = mix(shadowFresh, shadowOld, age);

    // Midtone color — brighter, more chromatic
    vec3 midFresh = vec3(0.55, 0.2, 0.28);       // bruised red
    vec3 midOld = vec3(0.52, 0.46, 0.18);         // sickly yellow-brown
    vec3 mid = mix(midFresh, midOld, age);
    // Sickness pushes midtones more green/yellow
    mid = mix(mid, vec3(0.45, 0.5, 0.15), sickness * 0.5);

    // Highlight color
    vec3 hiFresh = vec3(0.88, 0.72, 0.74);        // pale pink
    vec3 hiOld = vec3(0.85, 0.8, 0.5);             // jaundiced yellow
    vec3 hi = mix(hiFresh, hiOld, age);
    hi = mix(hi, vec3(0.78, 0.82, 0.45), sickness * 0.3);

    // Map luminance through the palette — no extra darkening
    vec3 col;
    if (luma < 0.25) {
        col = mix(shadow, mid, luma * 4.0);
    } else if (luma < 0.55) {
        col = mix(mid, hi, (luma - 0.25) / 0.3);
    } else {
        col = mix(hi, vec3(0.95, 0.92, 0.88), (luma - 0.55) / 0.45);
    }

    gl_FragColor = vec4(mix(src, col, intensity), 1.0);
}
