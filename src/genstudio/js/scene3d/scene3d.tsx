import React, { useEffect, useRef, useCallback, useMemo } from 'react';
import { createProgram } from './webgl-utils';
import { PointCloudViewerProps, ShaderUniforms, CameraState } from './types';
import { useContainerWidth, useDeepMemo } from '../utils';
import { FPSCounter, useFPSCounter } from './fps';
import Camera from './camera'

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

        // Decoration property uniforms
        uniform float uDecorationScales[MAX_DECORATIONS];
        uniform float uDecorationMinSizes[MAX_DECORATIONS];

        // Decoration mapping texture
        uniform sampler2D uDecorationMap;
        uniform int uDecorationMapSize;

        layout(location = 0) in vec3 position;
        layout(location = 1) in vec3 color;
        layout(location = 2) in float pointId;

        out vec3 vColor;
        flat out int vVertexID;
        flat out int vDecorationIndex;

        uniform bool uRenderMode;  // false = normal, true = picking

        int getDecorationIndex(int pointId) {
            // Convert pointId to texture coordinates
            int texWidth = uDecorationMapSize;
            int x = pointId % texWidth;
            int y = pointId / texWidth;

            vec2 texCoord = (vec2(x, y) + 0.5) / float(texWidth);
            return int(texture(uDecorationMap, texCoord).r * 255.0) - 1; // -1 means no decoration
        }

        void main() {
            vVertexID = int(pointId);
            vColor = color;
            vDecorationIndex = getDecorationIndex(vVertexID);

            vec4 viewPos = uViewMatrix * vec4(position, 1.0);
            float dist = -viewPos.z;

            float projectedSize = (uPointSize * uCanvasSize.y) / (2.0 * dist);
            float baseSize = clamp(projectedSize, 1.0, 20.0);

            float scale = 1.0;
            float minSize = 0.0;
            if (vDecorationIndex >= 0) {
                scale = uDecorationScales[vDecorationIndex];
                minSize = uDecorationMinSizes[vDecorationIndex];
            }

            float finalSize = max(baseSize * scale, minSize);
            gl_PointSize = finalSize;

            gl_Position = uProjectionMatrix * viewPos;
        }`,

    fragment: `#version 300 es
        precision highp float;
        precision highp int;
        #define MAX_DECORATIONS ${MAX_DECORATIONS}

        // Decoration property uniforms
        uniform vec3 uDecorationColors[MAX_DECORATIONS];
        uniform float uDecorationAlphas[MAX_DECORATIONS];

        // Decoration mapping texture
        uniform sampler2D uDecorationMap;
        uniform int uDecorationMapSize;

        in vec3 vColor;
        flat in int vVertexID;
        flat in int vDecorationIndex;
        out vec4 fragColor;

        uniform bool uRenderMode;  // false = normal, true = picking

        void main() {
            vec2 coord = gl_PointCoord * 2.0 - 1.0;
            float dist = dot(coord, coord);
            if (dist > 1.0) {
                discard;
            }

            if (uRenderMode) {
                // Just output ID, ignore decorations:
                float id = float(vVertexID);
                float blue = floor(id / (256.0 * 256.0));
                float green = floor((id - blue * 256.0 * 256.0) / 256.0);
                float red = id - blue * 256.0 * 256.0 - green * 256.0;
                fragColor = vec4(red / 255.0, green / 255.0, blue / 255.0, 1.0);
                return;
            }

            // Normal rendering mode
            vec3 baseColor = vColor;
            float alpha = 1.0;

            if (vDecorationIndex >= 0) {
                vec3 decorationColor = uDecorationColors[vDecorationIndex];
                if (decorationColor.r >= 0.0) {
                    baseColor = decorationColor;
                }
                alpha *= uDecorationAlphas[vDecorationIndex];
            }

            fragColor = vec4(baseColor, alpha);
        }`
};

    // Helper to save and restore GL state
const withGLState = (gl: WebGL2RenderingContext, uniformsRef, callback: () => void) => {
    const fbo = gl.getParameter(gl.FRAMEBUFFER_BINDING);
    const viewport = gl.getParameter(gl.VIEWPORT);
    const activeTexture = gl.getParameter(gl.ACTIVE_TEXTURE);
    const texture = gl.getParameter(gl.TEXTURE_BINDING_2D);
    const renderMode = 0; // Normal rendering mode
    const program = gl.getParameter(gl.CURRENT_PROGRAM);

    try {
        callback();
    } finally {
        // Restore state
        gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
        gl.viewport(viewport[0], viewport[1], viewport[2], viewport[3]);
        gl.activeTexture(activeTexture);
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.useProgram(program);
        gl.uniform1i(uniformsRef.current.renderMode, renderMode);
    }
};

function usePicking(pointSize: number, programRef, uniformsRef, vaoRef) {
    const pickingFbRef = useRef<WebGLFramebuffer | null>(null);
    const pickingTextureRef = useRef<WebGLTexture | null>(null);
    const numPointsRef = useRef<number>(0);
    const PICK_RADIUS = 10;
    const pickPoint = useCallback((gl: WebGL2RenderingContext, camera: CameraState, pixelCoords: [number, number]): number | null => {
        if (!gl || !programRef.current || !pickingFbRef.current || !vaoRef.current) {
            return null;
        }

        const [pixelX, pixelY] = pixelCoords;
        let closestId: number | null = null;

        withGLState(gl, uniformsRef, () => {
            // Set up picking framebuffer
            gl.bindFramebuffer(gl.FRAMEBUFFER, pickingFbRef.current);
            gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
            gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

            // Configure for picking render
            gl.useProgram(programRef.current);
            gl.uniform1i(uniformsRef.current.renderMode, 1); // Picking mode ON

            // Set uniforms
            const perspectiveMatrix = Camera.getPerspectiveMatrix(camera, gl);
            const orientationMatrix = Camera.getOrientationMatrix(camera);
            gl.uniformMatrix4fv(uniformsRef.current.projection, false, perspectiveMatrix);
            gl.uniformMatrix4fv(uniformsRef.current.view, false, orientationMatrix);
            gl.uniform1f(uniformsRef.current.pointSize, pointSize);
            gl.uniform2f(uniformsRef.current.canvasSize, gl.canvas.width, gl.canvas.height);

            // Draw points
            gl.bindVertexArray(vaoRef.current);
            gl.bindTexture(gl.TEXTURE_2D, null);
            gl.drawArrays(gl.POINTS, 0, numPointsRef.current);

            // Read pixels
            const size = PICK_RADIUS * 2 + 1;
            const startX = Math.max(0, pixelX - PICK_RADIUS);
            const startY = Math.max(0, gl.canvas.height - pixelY - PICK_RADIUS);
            const pixels = new Uint8Array(size * size * 4);
            gl.readPixels(startX, startY, size, size, gl.RGBA, gl.UNSIGNED_BYTE, pixels);

            // Find closest point
            const centerX = PICK_RADIUS;
            const centerY = PICK_RADIUS;
            let minDist = Infinity;

            for (let y = 0; y < size; y++) {
                for (let x = 0; x < size; x++) {
                    const i = (y * size + x) * 4;
                    if (pixels[i + 3] > 0) { // Found a point
                        const dist = Math.hypot(x - centerX, y - centerY);
                        if (dist < minDist) {
                            minDist = dist;
                            closestId = pixels[i] +
                                pixels[i + 1] * 256 +
                                pixels[i + 2] * 256 * 256;
                        }
                    }
                }
            }
        });

        return closestId;
    }, [pointSize]);

    function initPicking(gl: WebGL2RenderingContext, numPoints) {
        numPointsRef.current = numPoints;

        // Create framebuffer and texture for picking
        const pickingFb = gl.createFramebuffer();
        const pickingTexture = gl.createTexture();
        pickingFbRef.current = pickingFb;
        pickingTextureRef.current = pickingTexture;

        // Initialize texture
        gl.bindTexture(gl.TEXTURE_2D, pickingTexture);
        gl.texImage2D(
            gl.TEXTURE_2D, 0, gl.RGBA,
            gl.canvas.width, gl.canvas.height, 0,
            gl.RGBA, gl.UNSIGNED_BYTE, null
        );
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

        // Create and attach depth buffer
        const depthBuffer = gl.createRenderbuffer();
        gl.bindRenderbuffer(gl.RENDERBUFFER, depthBuffer);
        gl.renderbufferStorage(
            gl.RENDERBUFFER,
            gl.DEPTH_COMPONENT24,
            gl.canvas.width,
            gl.canvas.height
        );

        // Set up framebuffer
        gl.bindFramebuffer(gl.FRAMEBUFFER, pickingFb);
        gl.framebufferTexture2D(
            gl.FRAMEBUFFER,
            gl.COLOR_ATTACHMENT0,
            gl.TEXTURE_2D,
            pickingTexture,
            0
        );
        gl.framebufferRenderbuffer(
            gl.FRAMEBUFFER,
            gl.DEPTH_ATTACHMENT,
            gl.RENDERBUFFER,
            depthBuffer
        );

        // Verify framebuffer is complete
        const fbStatus = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
        if (fbStatus !== gl.FRAMEBUFFER_COMPLETE) {
            console.error('Picking framebuffer is incomplete');
            return;
        }

        // Restore default framebuffer
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);

        return () => {
            if (pickingFb) gl.deleteFramebuffer(pickingFb);
            if (pickingTexture) gl.deleteTexture(pickingTexture);
            if (depthBuffer) gl.deleteRenderbuffer(depthBuffer);
        };
    }

    return {
        initPicking,
        pickPoint
    };
}

function cacheUniformLocations(
    gl: WebGL2RenderingContext,
    program: WebGLProgram
): ShaderUniforms {
    return {
        projection: gl.getUniformLocation(program, 'uProjectionMatrix'),
        view: gl.getUniformLocation(program, 'uViewMatrix'),
        pointSize: gl.getUniformLocation(program, 'uPointSize'),
        canvasSize: gl.getUniformLocation(program, 'uCanvasSize'),

        // Decoration uniforms
        decorationScales: gl.getUniformLocation(program, 'uDecorationScales'),
        decorationColors: gl.getUniformLocation(program, 'uDecorationColors'),
        decorationAlphas: gl.getUniformLocation(program, 'uDecorationAlphas'),

        // Decoration map uniforms
        decorationMap: gl.getUniformLocation(program, 'uDecorationMap'),
        decorationMapSize: gl.getUniformLocation(program, 'uDecorationMapSize'),

        // Decoration min sizes
        decorationMinSizes: gl.getUniformLocation(program, 'uDecorationMinSizes'),

        // Render mode
        renderMode: gl.getUniformLocation(program, 'uRenderMode'),
    };
}

function computeCanvasDimensions(containerWidth, width, height, aspectRatio = 1) {
    if (!containerWidth && !width) return;

    // Determine final width from explicit width or container width
    const finalWidth = width || containerWidth;

    // Determine final height from explicit height or aspect ratio
    const finalHeight = height || finalWidth / aspectRatio;

    return {
        width: finalWidth,
        height: finalHeight,
        style: {
            width: width ? `${width}px` : '100%',
            height: `${finalHeight}px`
        }
    };
}

function devicePixels(gl, clientX, clientY) {
    const rect = gl.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;

    const pixelX = Math.floor((clientX - rect.left) * dpr);
    const pixelY = Math.floor((clientY - rect.top) * dpr);
    return [pixelX, pixelY]
}

// Define interaction modes
type InteractionMode = 'none' | 'orbit' | 'pan';
type InteractionState = {
    mode: InteractionMode;
    startX: number;
    startY: number;
};

export function Scene({
    points,
    camera,
    defaultCamera,
    onCameraChange,
    backgroundColor = [0.1, 0.1, 0.1],
    className,
    pointSize = 4.0,
    decorations = {},
    onPointClick,
    onPointHover,
    width,
    height,
    aspectRatio
}: PointCloudViewerProps) {

    points = useDeepMemo(points)
    decorations = useDeepMemo(decorations)
    backgroundColor = useDeepMemo(backgroundColor)


    const callbacksRef = useRef({})
    useEffect(() => {
        callbacksRef.current = {onPointHover, onPointClick, onCameraChange}
    },[onPointHover, onPointClick])

    const [containerRef, containerWidth] = useContainerWidth(1);
    const dimensions = useMemo(() => computeCanvasDimensions(containerWidth, width, height, aspectRatio), [containerWidth, width, height, aspectRatio])

    const renderFunctionRef = useRef<(() => void) | null>(null);
    const renderRAFRef = useRef<number | null>(null);
    const lastRenderTime = useRef<number | null>(null);

    const requestRender = useCallback(() => {
        if (!renderFunctionRef.current) return;

        cancelAnimationFrame(renderRAFRef.current);
        renderRAFRef.current = requestAnimationFrame(() => {
            renderFunctionRef.current();
            updateDisplay(performance.now() - (lastRenderTime.current ?? performance.now()));
            lastRenderTime.current = performance.now();
        });
    }, []);

    const {
        cameraRef,
        handleCameraMove
    } = Camera.useCamera(requestRender, camera, defaultCamera, callbacksRef);

    const canvasRef = useRef<HTMLCanvasElement>(null);
    const glRef = useRef<WebGL2RenderingContext>(null);
    const programRef = useRef<WebGLProgram>(null);
    const interactionState = useRef<InteractionState>({
        mode: 'none',
        startX: 0,
        startY: 0
    });
    const animationFrameRef = useRef<number>();
    const vaoRef = useRef(null);
    const uniformsRef = useRef<ShaderUniforms | null>(null);
    const mouseDownPositionRef = useRef<{x: number, y: number} | null>(null);
    const CLICK_THRESHOLD = 3; // Pixels of movement allowed before considering it a drag

    const { fpsDisplayRef, updateDisplay } = useFPSCounter();

    const pickingSystem = usePicking(pointSize, programRef, uniformsRef, vaoRef);

    // Add refs for decoration texture
    const decorationMapRef = useRef<WebGLTexture | null>(null);
    const decorationMapSizeRef = useRef<number>(0);
    const decorationsRef = useRef(decorations);

    // Update the ref when decorations change
    useEffect(() => {
        decorationsRef.current = decorations;
        requestRender();
    }, [decorations]);

    // Helper to create/update decoration map texture
    const updateDecorationMap = useCallback((gl: WebGL2RenderingContext, numPoints: number) => {
        // Calculate texture size (power of 2 that can fit all points)
        const texSize = Math.ceil(Math.sqrt(numPoints));
        const size = Math.pow(2, Math.ceil(Math.log2(texSize)));
        decorationMapSizeRef.current = size;

        // Create mapping array (default to 0 = no decoration)
        const mapping = new Uint8Array(size * size).fill(0);

        // Fill in decoration mappings - make sure we're using the correct point indices
        Object.values(decorationsRef.current).forEach((decoration, decorationIndex) => {
            decoration.indexes.forEach(pointIndex => {
                if (pointIndex < numPoints) {
                    // The mapping array should be indexed by the actual point index
                    mapping[pointIndex] = decorationIndex + 1;
                }
            });
        });

        // Create/update texture
        if (!decorationMapRef.current) {
            decorationMapRef.current = gl.createTexture();
        }

        gl.bindTexture(gl.TEXTURE_2D, decorationMapRef.current);
        gl.texImage2D(
            gl.TEXTURE_2D,
            0,
            gl.R8,
            size,
            size,
            0,
            gl.RED,
            gl.UNSIGNED_BYTE,
            mapping
        );
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    }, [decorationsRef]);

    // Simplified mouse handlers
    const handleMouseDown = useCallback((e: MouseEvent) => {
        const mode: InteractionMode = e.button === 0
            ? (e.shiftKey ? 'pan' : 'orbit')
            : e.button === 1 ? 'pan' : 'none';

        if (mode !== 'none') {
            interactionState.current = {
                mode,
                startX: e.clientX,
                startY: e.clientY
            };
        }
    }, []);

    const handleMouseMove = useCallback((e: MouseEvent) => {
        if (!cameraRef.current) return;
        const onHover = callbacksRef.current.onPointHover;
        const { mode, startX, startY } = interactionState.current;

        switch (mode) {
            case 'orbit':
                onHover?.(null);
                handleCameraMove(camera => Camera.orbit(camera, e.movementX, e.movementY));
                break;
            case 'pan':
                onHover?.(null);
                handleCameraMove(camera => Camera.pan(camera, e.movementX, e.movementY));
                break;
            case 'none':
                if (onHover) {
                    const pointIndex = pickingSystem.pickPoint(
                        glRef.current,
                        cameraRef.current,
                        devicePixels(glRef.current, e.clientX, e.clientY)
                    );
                    onHover(pointIndex);
                }
                break;
        }
    }, [handleCameraMove, pickingSystem.pickPoint]);

    const handleMouseUp = useCallback((e: MouseEvent) => {
        const { mode, startX, startY } = interactionState.current;
        const onClick = callbacksRef.current.onPointClick;

        if (mode === 'orbit' && onClick) {
            const dx = e.clientX - startX;
            const dy = e.clientY - startY;
            const distance = Math.sqrt(dx * dx + dy * dy);

            if (distance < CLICK_THRESHOLD) {
                const pointIndex = pickingSystem.pickPoint(
                    glRef.current,
                    cameraRef.current,
                    devicePixels(glRef.current, e.clientX, e.clientY)
                );
                if (pointIndex !== null) {
                    onClick(pointIndex, e);
                }
            }
        }

        interactionState.current = {
            mode: 'none',
            startX: 0,
            startY: 0
        };
    }, [pickingSystem.pickPoint]);

    const handleWheel = useCallback((e: WheelEvent) => {
        e.preventDefault();
        handleCameraMove(camera => Camera.zoom(camera, e.deltaY));
    }, [handleCameraMove]);

    // Add mouseLeave handler to clear hover state when leaving canvas
    useEffect(() => {
        if (!canvasRef.current || !dimensions) return;

        const handleMouseLeave = () => {
            callbacksRef.current.onPointHover?.(null);
        };

        canvasRef.current.addEventListener('mouseleave', handleMouseLeave);

        return () => {
            if (canvasRef.current) {
                canvasRef.current.removeEventListener('mouseleave', handleMouseLeave);
            }
        };
    }, []);



    // Effect for WebGL initialization
    useEffect(() => {
        if (!canvasRef.current) return;
        const disposeFns = []
        const gl = canvasRef.current.getContext('webgl2');
        if (!gl) {
            return console.error('WebGL2 not supported');
        }

        glRef.current = gl;

        // Create program and get uniforms
        const program = createProgram(gl, mainShaders.vertex, mainShaders.fragment);
        programRef.current = program;

        // Cache uniform locations
        uniformsRef.current = cacheUniformLocations(gl, program);

        // Create buffers
        const positionBuffer = gl.createBuffer();
        const colorBuffer = gl.createBuffer();
        const numPoints = points.position.length / 3;

        // Create point ID buffer with verified sequential IDs
        const pointIdBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, pointIdBuffer);
        const pointIds = new Float32Array(numPoints);
        for (let i = 0; i < numPoints; i++) {
            pointIds[i] = i;
        }
        gl.bufferData(gl.ARRAY_BUFFER, pointIds, gl.STATIC_DRAW);

        // Position buffer
        gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, points.position, gl.STATIC_DRAW);

        // Color buffer
        gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
        if (points.color) {
            const normalizedColors = new Float32Array(points.color.length);
            for (let i = 0; i < points.color.length; i++) {
                normalizedColors[i] = points.color[i] / 255.0;
            }
            gl.bufferData(gl.ARRAY_BUFFER, normalizedColors, gl.STATIC_DRAW);
        } else {
            const defaultColors = new Float32Array(points.position.length);
            defaultColors.fill(0.7);
            gl.bufferData(gl.ARRAY_BUFFER, defaultColors, gl.STATIC_DRAW);
        }

        // Set up VAO with consistent attribute locations
        const vao = gl.createVertexArray();
        gl.bindVertexArray(vao);
        vaoRef.current = vao;

        // Position attribute (location 0)
        gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

        // Color attribute (location 1)
        gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
        gl.enableVertexAttribArray(1);
        gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);

        // Point ID attribute (location 2)
        gl.bindBuffer(gl.ARRAY_BUFFER, pointIdBuffer);
        gl.enableVertexAttribArray(2);
        gl.vertexAttribPointer(2, 1, gl.FLOAT, false, 0, 0);

        // Store numPoints in ref for picking system
        // numPointsRef.current = numPoints;

        // Initialize picking system after VAO setup is complete
        disposeFns.push(pickingSystem.initPicking(gl, points.position.length / 3));

        canvasRef.current.addEventListener('mousedown', handleMouseDown);
        canvasRef.current.addEventListener('mousemove', handleMouseMove);
        canvasRef.current.addEventListener('mouseup', handleMouseUp);
        canvasRef.current.addEventListener('wheel', handleWheel, { passive: false });

        return () => {
            if (animationFrameRef.current) {
                cancelAnimationFrame(animationFrameRef.current);
            }
            if (gl) {
                if (programRef.current) {
                    gl.deleteProgram(programRef.current);
                    programRef.current = null;
                }
                if (vao) {
                    gl.deleteVertexArray(vao);
                }
                if (positionBuffer) {
                    gl.deleteBuffer(positionBuffer);
                }
                if (colorBuffer) {
                    gl.deleteBuffer(colorBuffer);
                }
                if (pointIdBuffer) {
                    gl.deleteBuffer(pointIdBuffer);
                }
                disposeFns.forEach(fn => fn?.());
            }
            if (canvasRef.current) {
                canvasRef.current.removeEventListener('mousedown', handleMouseDown);
                canvasRef.current.removeEventListener('mousemove', handleMouseMove);
                canvasRef.current.removeEventListener('mouseup', handleMouseUp);
                canvasRef.current.removeEventListener('wheel', handleWheel);
            }
        };
    }, [points, handleMouseMove, handleMouseUp, handleWheel, canvasRef.current?.width, canvasRef.current?.height]);

    // Effect for per-frame rendering and picking updates
    useEffect(() => {
        if (!glRef.current || !programRef.current || !cameraRef.current) return;

        const gl = glRef.current;

        renderFunctionRef.current = function render() {
            gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
            gl.clearColor(backgroundColor[0], backgroundColor[1], backgroundColor[2], 1.0);
            gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
            gl.enable(gl.DEPTH_TEST);
            gl.enable(gl.BLEND);
            gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

            gl.useProgram(programRef.current);

            // Set up matrices
            const perspectiveMatrix = Camera.getPerspectiveMatrix(cameraRef.current, gl);
            const orientationMatrix = Camera.getOrientationMatrix(cameraRef.current)

            // Set all uniforms in one place
            gl.uniformMatrix4fv(uniformsRef.current.projection, false, perspectiveMatrix);
            gl.uniformMatrix4fv(uniformsRef.current.view, false, orientationMatrix);
            gl.uniform1f(uniformsRef.current.pointSize, pointSize);
            gl.uniform2f(uniformsRef.current.canvasSize, gl.canvas.width, gl.canvas.height);

            // Update decoration map
            updateDecorationMap(gl, points.position.length / 3);

            // Set decoration map uniforms
            gl.activeTexture(gl.TEXTURE0);
            gl.bindTexture(gl.TEXTURE_2D, decorationMapRef.current);
            gl.uniform1i(uniformsRef.current.decorationMap, 0);
            gl.uniform1i(uniformsRef.current.decorationMapSize, decorationMapSizeRef.current);

            const currentDecorations = decorationsRef.current;

            // Prepare decoration data
            const scales = new Float32Array(MAX_DECORATIONS).fill(1.0);
            const colors = new Float32Array(MAX_DECORATIONS * 3).fill(-1);
            const alphas = new Float32Array(MAX_DECORATIONS).fill(1.0);
            const minSizes = new Float32Array(MAX_DECORATIONS).fill(0.0);

            // Fill arrays with decoration data

            Object.values(currentDecorations).slice(0, MAX_DECORATIONS).forEach((decoration, i) => {
                scales[i] = decoration.scale ?? 1.0;

                if (decoration.color) {
                    const baseIdx = i * 3;
                    colors[baseIdx] = decoration.color[0];
                    colors[baseIdx + 1] = decoration.color[1];
                    colors[baseIdx + 2] = decoration.color[2];
                }

                alphas[i] = decoration.alpha ?? 1.0;
                minSizes[i] = decoration.minSize ?? 0.0;
            });

            // Set uniforms
            gl.uniform1fv(uniformsRef.current.decorationScales, scales);
            gl.uniform3fv(uniformsRef.current.decorationColors, colors);
            gl.uniform1fv(uniformsRef.current.decorationAlphas, alphas);
            gl.uniform1fv(uniformsRef.current.decorationMinSizes, minSizes);

            // Ensure correct VAO is bound
            gl.bindVertexArray(vaoRef.current);
            gl.drawArrays(gl.POINTS, 0, points.position.length / 3);
        }

    }, [points, backgroundColor, pointSize, canvasRef.current?.width, canvasRef.current?.height]);


    // Set up resize observer
    useEffect(() => {
        if (!canvasRef.current || !glRef.current || !dimensions) return;

        const gl = glRef.current;
        const canvas = canvasRef.current;

        const dpr = window.devicePixelRatio || 1;
        const displayWidth = Math.floor(dimensions.width * dpr);
        const displayHeight = Math.floor(dimensions.height * dpr);

        if (canvas.width !== displayWidth || canvas.height !== displayHeight) {
            canvas.width = displayWidth;
            canvas.height = displayHeight;
            canvas.style.width = dimensions.style.width;
            canvas.style.height = dimensions.style.height;
            gl.viewport(0, 0, canvas.width, canvas.height);
            requestRender();
        }
    }, [dimensions]);


    // Clean up
    useEffect(() => {
        const gl = glRef.current;  // Get gl from ref
        return () => {
            if (gl && decorationMapRef.current) {
                gl.deleteTexture(decorationMapRef.current);
            }
        };
    }, []);  // Remove gl from deps since we're getting it from ref

    return (
        <div
            ref={containerRef}
            style={{
                position: 'relative'
            }}
        >
            <canvas
                ref={canvasRef}
                className={className}
                style={dimensions?.style}
            />
            <FPSCounter fpsRef={fpsDisplayRef} />
        </div>
    );
}
