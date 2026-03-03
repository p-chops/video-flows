/*{
    "DESCRIPTION": "Animated Voronoi cells with drifting seed points. Produces organic, cell-like moving textures.",
    "CREDIT": "Undersea Lair Project",
    "ISFVSN": "2",
    "CATEGORIES": ["Generator", "Brain Wipe"],
    "INPUTS": [
        {
            "NAME": "cell_count",
            "TYPE": "float",
            "DEFAULT": 5.0,
            "MIN": 3.0,
            "MAX": 12.0
        },
        {
            "NAME": "speed",
            "TYPE": "float",
            "DEFAULT": 0.3,
            "MIN": 0.05,
            "MAX": 1.0
        },
        {
            "NAME": "edge_width",
            "TYPE": "float",
            "DEFAULT": 0.05,
            "MIN": 0.01,
            "MAX": 0.15
        },
        {
            "NAME": "brightness",
            "TYPE": "float",
            "DEFAULT": 0.8,
            "MIN": 0.3,
            "MAX": 1.5
        }
    ]
}*/

vec2 hash2(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)),
             dot(p, vec2(269.5, 183.3)));
    return fract(sin(p) * 43758.5453);
}

void main() {
    vec2 uv = isf_FragNormCoord;
    float t = TIME * speed;

    vec2 p = uv * cell_count;
    vec2 ip = floor(p);
    vec2 fp = fract(p);

    float d1 = 8.0;  // closest distance
    float d2 = 8.0;  // second closest
    vec2 closest_id = vec2(0.0);

    // Check 3x3 neighborhood
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec2 neighbor = vec2(float(x), float(y));
            vec2 point = hash2(ip + neighbor);

            // Animate the seed points
            point = 0.5 + 0.5 * sin(t * 0.8 + 6.28318 * point);

            vec2 diff = neighbor + point - fp;
            float d = dot(diff, diff);

            if (d < d1) {
                d2 = d1;
                d1 = d;
                closest_id = ip + neighbor;
            } else if (d < d2) {
                d2 = d;
            }
        }
    }

    d1 = sqrt(d1);
    d2 = sqrt(d2);

    // Edge detection from distance difference
    float edge = smoothstep(0.0, edge_width, d2 - d1);

    // Cell color from cell ID
    vec2 id_hash = hash2(closest_id);
    float hue = id_hash.x + t * 0.1;
    float sat = 0.5 + id_hash.y * 0.4;
    float val = (0.4 + d1 * 0.4) * edge;

    // HSV to RGB
    vec3 c = vec3(hue, sat, val);
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 px = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    vec3 col = c.z * mix(K.xxx, clamp(px - K.xxx, 0.0, 1.0), c.y);

    col *= brightness;

    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
