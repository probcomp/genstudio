export const mainShaders = {
    vertex: `#version 300 es
        uniform mat4 uProjectionMatrix;
        uniform mat4 uViewMatrix;
        uniform float uPointSize;
        uniform int uHighlightedPoint;
        uniform vec3 uHighlightColor;
        uniform vec2 uCanvasSize;

        layout(location = 0) in vec3 position;
        layout(location = 1) in vec3 color;
        layout(location = 2) in float pointId;

        out vec3 vColor;

        void main() {
            bool isHighlighted = (int(pointId) == uHighlightedPoint);
            vColor = isHighlighted ? uHighlightColor : color;

            vec4 viewPos = uViewMatrix * vec4(position, 1.0);
            float dist = -viewPos.z;

            // Physical size scaling:
            // - uPointSize represents the desired size in world units
            // - Scale by canvas height to maintain size in screen space
            // - Multiply by 0.5 to convert from diameter to radius
            float projectedSize = (uPointSize * uCanvasSize.y) / (2.0 * dist);
            float baseSize = clamp(projectedSize, 1.0, 20.0);

            float minHighlightSize = 8.0;
            float relativeHighlightSize = min(uCanvasSize.x, uCanvasSize.y) * 0.02;
            float sizeFromBase = baseSize * 2.0;

            float highlightSize = max(max(minHighlightSize, relativeHighlightSize), sizeFromBase);

            gl_Position = uProjectionMatrix * viewPos;
            gl_PointSize = isHighlighted ? highlightSize : baseSize;
        }`,

    fragment: `#version 300 es
        precision highp float;

        in vec3 vColor;
        out vec4 fragColor;

        void main() {
            vec2 coord = gl_PointCoord * 2.0 - 1.0;
            float dist = dot(coord, coord);
            if (dist > 1.0) {
                discard;
            }
            fragColor = vec4(vColor, 1.0);
        }`
};

export const pickingShaders = {
    vertex: `#version 300 es
        uniform mat4 uProjectionMatrix;
        uniform mat4 uViewMatrix;
        uniform float uPointSize;
        uniform vec2 uCanvasSize;

        layout(location = 0) in vec3 position;
        layout(location = 1) in float pointId;

        out float vPointId;

        void main() {
            vPointId = pointId;
            vec4 viewPos = uViewMatrix * vec4(position, 1.0);
            float dist = -viewPos.z;

            float projectedSize = (uPointSize * uCanvasSize.y) / (2.0 * dist);
            float pointSize = clamp(projectedSize, 1.0, 20.0);

            gl_Position = uProjectionMatrix * viewPos;
            gl_PointSize = pointSize;
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
        }`
};
