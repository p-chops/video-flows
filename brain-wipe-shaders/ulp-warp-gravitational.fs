/*{
    "DESCRIPTION": "Gravitational lensing — multiple animated point masses bend video like light around massive objects. Each mass warps space radially. Masses orbit, drift, and pulse. At high strength, creates black-hole-like singularities.",
    "CREDIT": "ULP Warp Series",
    "ISFVSN": "2",
    "CATEGORIES": ["Warp", "ULP"],
    "INPUTS": [
        {
            "NAME": "inputImage",
            "TYPE": "image",
            "LABEL": "Video Input"
        },
        {
            "NAME": "warp_strength",
            "TYPE": "float",
            "DEFAULT": 0.15,
            "MIN": 0.0,
            "MAX": 2.0,
            "LABEL": "Warp Strength"
        },
        {
            "NAME": "num_masses",
            "TYPE": "float",
            "DEFAULT": 3.0,
            "MIN": 1.0,
            "MAX": 5.0,
            "LABEL": "Number of Masses"
        },
        {
            "NAME": "softening",
            "TYPE": "float",
            "DEFAULT": 0.02,
            "MIN": 0.001,
            "MAX": 0.2,
            "LABEL": "Softening (singularity radius)"
        },
        {
            "NAME": "orbit_speed",
            "TYPE": "float",
            "DEFAULT": 0.2,
            "MIN": 0.0,
            "MAX": 2.0,
            "LABEL": "Orbit Speed"
        },
        {
            "NAME": "orbit_radius",
            "TYPE": "float",
            "DEFAULT": 0.3,
            "MIN": 0.0,
            "MAX": 0.5,
            "LABEL": "Orbit Radius"
        },
        {
            "NAME": "mass_pulse",
            "TYPE": "float",
            "DEFAULT": 0.0,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Mass Pulsing"
        },
        {
            "NAME": "attraction",
            "TYPE": "float",
            "DEFAULT": 1.0,
            "MIN": -1.0,
            "MAX": 1.0,
            "LABEL": "Attraction / Repulsion"
        },
        {
            "NAME": "falloff",
            "TYPE": "float",
            "DEFAULT": 2.0,
            "MIN": 0.5,
            "MAX": 4.0,
            "LABEL": "Falloff Power"
        },
        {
            "NAME": "time_offset",
            "TYPE": "float",
            "DEFAULT": 0.0,
            "MIN": 0.0,
            "MAX": 100.0,
            "LABEL": "Time Offset"
        },
        {
            "NAME": "brightness",
            "TYPE": "float",
            "DEFAULT": 1.0,
            "MIN": 0.1,
            "MAX": 2.0,
            "LABEL": "Brightness"
        },
        {
            "NAME": "desaturate",
            "TYPE": "float",
            "DEFAULT": 0.0,
            "MIN": 0.0,
            "MAX": 1.0,
            "LABEL": "Desaturate"
        }
    ]
}*/

#define PI 3.14159265359
#define TAU 6.28318530718
#define MAX_MASSES 5

void main() {
    vec2 uv = isf_FragNormCoord;
    // Aspect-correct for distance calculations, then map back
    float aspect = RENDERSIZE.x / RENDERSIZE.y;
    vec2 asuv = vec2(uv.x * aspect, uv.y);

    float t = (TIME + time_offset) * orbit_speed;
    int nm = int(clamp(num_masses, 1.0, float(MAX_MASSES)));

    // Build mass positions — each orbits at a different phase and radius
    // Phase offsets are irrational multiples to avoid alignment
    vec2 massPos[MAX_MASSES];
    float massWeight[MAX_MASSES];

    float phases[5];
    phases[0] = 0.0;
    phases[1] = TAU / 2.618;      // golden angle
    phases[2] = TAU * 2.0 / 2.618;
    phases[3] = TAU * 3.0 / 2.618;
    phases[4] = TAU * 4.0 / 2.618;

    float radii[5];
    radii[0] = 1.0;
    radii[1] = 0.85;
    radii[2] = 0.7;
    radii[3] = 1.1;
    radii[4] = 0.6;

    float speeds[5];
    speeds[0] = 1.0;
    speeds[1] = -0.618;
    speeds[2] = 1.414;
    speeds[3] = -0.732;
    speeds[4] = 0.5;

    for (int i = 0; i < MAX_MASSES; i++) {
        float ang = phases[i] + t * speeds[i];
        float r = orbit_radius * radii[i];
        massPos[i] = vec2(0.5 * aspect + cos(ang) * r,
                          0.5         + sin(ang) * r);
        // Pulsing mass weight
        float pulse = 1.0 + mass_pulse * sin(t * speeds[i] * 2.0 + phases[i]);
        massWeight[i] = pulse;
    }

    // Accumulate gravitational deflection from all active masses
    vec2 deflection = vec2(0.0);

    for (int i = 0; i < MAX_MASSES; i++) {
        if (i >= nm) break;
        vec2 delta = asuv - massPos[i];
        float r2 = dot(delta, delta) + softening * softening;
        float r_pow = pow(r2, falloff * 0.5);
        // Deflection toward mass (attraction>0) or away (attraction<0)
        deflection -= normalize(delta) * massWeight[i] * attraction / r_pow;
    }

    // Apply to video UV (deflection is in aspect-space, convert back)
    deflection.x /= aspect;
    vec2 sample_uv = uv + deflection * warp_strength * 0.05;
    sample_uv = clamp(sample_uv, 0.0, 1.0);

    vec4 tex = IMG_NORM_PIXEL(inputImage, sample_uv);
    vec3 col = tex.rgb;

    if (desaturate > 0.001) {
        float luma = dot(col, vec3(0.299, 0.587, 0.114));
        col = mix(col, vec3(luma), desaturate);
    }

    col *= brightness;

    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
