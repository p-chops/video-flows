/*{
    "DESCRIPTION": "Multi-axis displacement — strips of the image shear along arbitrary angles. Vertical columns shift up/down, diagonal slabs slide at oblique angles. Not limited to horizontal. Zones are coherent blocks that move as a unit.",
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
            "DEFAULT": 0.65,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Intensity"
        },
        {
            "NAME": "density",
            "TYPE": "float",
            "DEFAULT": 0.40,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Strip Density"
        },
        {
            "NAME": "shift_amount",
            "TYPE": "float",
            "DEFAULT": 0.25,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Shift Amount"
        },
        {
            "NAME": "angle_range",
            "TYPE": "float",
            "DEFAULT": 0.50,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Angle Range"
        }
    ]
}*/

float hash1(float n) { return fract(sin(n) * 43758.5453123); }

void main() {
    vec2 uv = isf_FragNormCoord;

    float tBlock = floor(TIME * mix(1.5, 6.0, density));

    // Multiple displacement layers at different angles
    vec2 totalShift = vec2(0.0);

    for (int layer = 0; layer < 3; layer++) {
        float layerSeed = float(layer) * 137.0 + tBlock * 7.3;

        // Each layer picks a different angle
        // angle_range=0 → only vertical/horizontal, angle_range=1 → any direction
        float baseAngle = float(layer) * 1.047; // 0, 60, 120 degrees
        float angle = baseAngle + hash1(layerSeed * 0.3) * angle_range * 3.14159;

        // Direction perpendicular to the strip axis
        vec2 stripDir = vec2(cos(angle), sin(angle));
        vec2 shiftDir = vec2(-stripDir.y, stripDir.x);

        // Project UV onto strip axis to determine which strip we're in
        float proj = dot(uv, stripDir);
        float stripCount = mix(3.0, 15.0, density);
        float stripIdx = floor(proj * stripCount);

        // Is this strip active?
        float gate = step(0.7 - density * 0.3, hash1(stripIdx * 53.0 + layerSeed));

        // Global burst gate — not all layers fire at once
        float layerGate = step(0.4, hash1(float(layer) * 19.0 + tBlock));

        // Shift amount for this strip
        float s = (hash1(stripIdx * 91.0 + layerSeed) - 0.5)
                  * shift_amount * 0.2
                  * gate * layerGate;

        totalShift += shiftDir * s;
    }

    vec2 displaced = fract(uv + totalShift * intensity);
    vec3 col = IMG_NORM_PIXEL(inputImage, displaced).rgb;

    vec3 src = IMG_NORM_PIXEL(inputImage, uv).rgb;
    gl_FragColor = vec4(mix(src, col, intensity), 1.0);
}
