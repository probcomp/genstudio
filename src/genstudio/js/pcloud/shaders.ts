export const MAX_DECORATIONS = 16; // Adjust based on needs

export const mainShaders = {
    vertex: `#version 300 es
        precision highp float;
        precision highp int;
        #define MAX_DECORATIONS ${MAX_DECORATIONS}

        uniform mat4 uProjectionMatrix;
        uniform mat4 uViewMatrix;
        uniform float uPointSize;
        uniform vec2 uCanvasSize;

        // Decoration data passed as individual uniforms for better compatibility
        uniform highp int uDecorationIndices[MAX_DECORATIONS];
        uniform float uDecorationScales[MAX_DECORATIONS];
        uniform int uDecorationCount;

        layout(location = 0) in vec3 position;
        layout(location = 1) in vec3 color;
        layout(location = 2) in float pointId;

        out vec3 vColor;
        flat out int vVertexID;

        void main() {
            vVertexID = int(pointId);
            vColor = color;

            vec4 viewPos = uViewMatrix * vec4(position, 1.0);
            float dist = -viewPos.z;

            // Calculate base point size with perspective
            float projectedSize = (uPointSize * uCanvasSize.y) / (2.0 * dist);
            float baseSize = clamp(projectedSize, 1.0, 20.0);

            // Apply decoration scaling if point is decorated
            float scale = 1.0;
            for (int i = 0; i < uDecorationCount; i++) {
                if (uDecorationIndices[i] == vVertexID) {
                    scale = uDecorationScales[i];
                    break;
                }
            }

            gl_Position = uProjectionMatrix * viewPos;
            gl_PointSize = baseSize * scale;
        }`,

    fragment: `#version 300 es
        precision highp float;
        precision highp int;
        #define MAX_DECORATIONS ${MAX_DECORATIONS}

        in vec3 vColor;
        flat in int vVertexID;
        out vec4 fragColor;

        // Decoration appearance uniforms
        uniform highp int uDecorationIndices[MAX_DECORATIONS];
        uniform vec3 uDecorationColors[MAX_DECORATIONS];
        uniform float uDecorationAlphas[MAX_DECORATIONS];
        uniform int uDecorationBlendModes[MAX_DECORATIONS];
        uniform float uDecorationBlendStrengths[MAX_DECORATIONS];
        uniform int uDecorationCount;

        vec3 applyBlend(vec3 base, vec3 blend, int mode, float strength) {
            vec3 result = base;
            if (mode == 0) { // replace
                result = blend;
            } else if (mode == 1) { // multiply
                result = base * blend;
            } else if (mode == 2) { // add
                result = min(base + blend, 1.0);
            } else if (mode == 3) { // screen
                result = 1.0 - (1.0 - base) * (1.0 - blend);
            }
            return mix(base, result, strength);
        }

        void main() {
            // Basic point shape
            vec2 coord = gl_PointCoord * 2.0 - 1.0;
            float dist = dot(coord, coord);
            if (dist > 1.0) {
                discard;
            }

            vec3 baseColor = vColor;
            float alpha = 1.0;

            // Apply decorations in order
            for (int i = 0; i < uDecorationCount; i++) {
                if (uDecorationIndices[i] == vVertexID) {
                    baseColor = applyBlend(
                        baseColor,
                        uDecorationColors[i],
                        uDecorationBlendModes[i],
                        uDecorationBlendStrengths[i]
                    );
                    alpha *= uDecorationAlphas[i];
                }
            }

            fragColor = vec4(baseColor, alpha);
        }`
};

export const pickingShaders = {
    vertex: `#version 300 es
        uniform mat4 uProjectionMatrix;
        uniform mat4 uViewMatrix;
        uniform float uPointSize;
        uniform vec2 uCanvasSize;
        uniform int uHighlightedPoint;

        layout(location = 0) in vec3 position;
        layout(location = 1) in float pointId;

        out float vPointId;

        void main() {
            vPointId = pointId;
            vec4 viewPos = uViewMatrix * vec4(position, 1.0);
            float dist = -viewPos.z;

            float projectedSize = (uPointSize * uCanvasSize.y) / (2.0 * dist);
            float baseSize = clamp(projectedSize, 1.0, 20.0);

            bool isHighlighted = (int(pointId) == uHighlightedPoint);
            float minHighlightSize = 8.0;
            float relativeHighlightSize = min(uCanvasSize.x, uCanvasSize.y) * 0.02;
            float sizeFromBase = baseSize * 2.0;
            float highlightSize = max(max(minHighlightSize, relativeHighlightSize), sizeFromBase);

            gl_Position = uProjectionMatrix * viewPos;
            gl_PointSize = isHighlighted ? highlightSize : baseSize;
        }`,

    fragment: `#version 300 es
        precision highp float;

        in float vPointId;
        out vec4 fragColor;

        void main() {
            vec2 coord = gl_PointCoord * 2.0 - 1.0;
            float dist = dot(coord, coord);
            if (dist > 1.0) {
                discard;
            }

            float id = vPointId;
            float blue = floor(id / (256.0 * 256.0));
            float green = floor((id - blue * 256.0 * 256.0) / 256.0);
            float red = id - blue * 256.0 * 256.0 - green * 256.0;

            fragColor = vec4(red / 255.0, green / 255.0, blue / 255.0, 1.0);
            gl_FragDepth = gl_FragCoord.z;
        }`
};
