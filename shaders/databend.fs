/*{
    "DESCRIPTION": "Simulates raw byte-level data corruption: coherent row zones shift horizontally as a slab, colour channels swap between three variants (GBR / BRG / BGR), and a global temporal gate fires the whole effect in intermittent bursts. All transitions are hard-edged (floor / step) for an authentic digital-corruption feel.",
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
            "NAME": "corruption",
            "TYPE": "float",
            "DEFAULT": 0.25,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Corruption Density"
        },
        {
            "NAME": "row_shift",
            "TYPE": "float",
            "DEFAULT": 0.15,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Row Shift"
        }
    ]
}*/

float hash1(float n) { return fract(sin(n)             * 43758.5453123); }
float hash2(vec2 p)  { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123); }

void main() {
    vec2  uv = isf_FragNormCoord;

    // Two clocks: coarse for zone decisions, fine for channel-swap stutter.
    float tCoarse = floor(TIME * mix(1.0, 4.0, corruption));
    float tFine   = floor(TIME * 6.0);

    // ── Zone-based row shift ────────────────────────────────────────────────
    // A global gate fires the effect in intermittent bursts (~55 % quiet).
    float globalGate = step(0.45, hash1(tCoarse * 31.7));

    float zoneCount  = mix(3.0, 10.0, corruption);
    float zoneIdx    = floor(uv.y * zoneCount);
    float zoneActive = step(1.0 - corruption * 0.45, hash1(zoneIdx * 97.3 + tCoarse))
                       * globalGate;

    // All rows in an active zone share one shift → coherent displaced slab.
    float rShift = (hash1(zoneIdx * 53.1 + tCoarse * 0.7) - 0.5)
                   * row_shift * 0.22 * zoneActive;

    vec4 src = IMG_NORM_PIXEL(inputImage, vec2(fract(uv.x + rShift), uv.y));
    vec3 col = src.rgb;

    // ── Channel swap (3 variants) ───────────────────────────────────────────
    float swapG   = floor(uv.y * 40.0);
    float swNoise = hash1(swapG * 41.7 + tFine * 1.3);
    float swType  = hash1(swapG * 23.9 + tFine * 0.5);
    float doSwap  = step(1.0 - corruption * 0.35, swNoise) * globalGate;

    // Ternary on vec3 is valid GLSL 1.20+; use mix/step if targeting ES 1.0.
    vec3 swGBR    = col.gbr;
    vec3 swBRG    = col.brg;
    vec3 swBGR    = col.bgr;
    // Select variant: swType < 0.33 → GBR, < 0.66 → BRG, else → BGR
    vec3 swapped  = mix(swGBR,
                        mix(swBRG, swBGR, step(0.66, swType)),
                        step(0.33, swType));

    col = mix(col, swapped, doSwap);

    gl_FragColor = vec4(mix(src.rgb, col, intensity), 1.0);
}
