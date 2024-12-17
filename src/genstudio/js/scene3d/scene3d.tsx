import React, { useEffect, useRef, useCallback, useMemo, useState } from 'react';
import { mat4, vec3 } from 'gl-matrix';
import { createProgram, createPointIdBuffer } from './webgl-utils';
import { CameraParams, PointCloudViewerProps, ShaderUniforms, PickingUniforms } from './types';
import { useContainerWidth, useDeepMemo } from '../utils';
import { FPSCounter, useFPSCounter } from './fps';

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
        uniform int uDecorationBlendModes[MAX_DECORATIONS];
        uniform float uDecorationBlendStrengths[MAX_DECORATIONS];

        // Decoration mapping texture
        uniform sampler2D uDecorationMap;
        uniform int uDecorationMapSize;

        in vec3 vColor;
        flat in int vVertexID;
        flat in int vDecorationIndex;
        out vec4 fragColor;

        uniform bool uRenderMode;  // false = normal, true = picking

        vec3 applyBlend(vec3 base, vec3 blend, int mode, float strength) {
            if (blend.r < 0.0) return base;  // No color override

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
                    baseColor = applyBlend(
                        baseColor,
                        decorationColor,
                        uDecorationBlendModes[vDecorationIndex],
                        uDecorationBlendStrengths[vDecorationIndex]
                    );
                }
                alpha *= uDecorationAlphas[vDecorationIndex];
            }

            fragColor = vec4(baseColor, alpha);
        }`
};
export class OrbitCamera {
    position: vec3;
    target: vec3;
    up: vec3;
    radius: number;
    phi: number;
    theta: number;
    fov: number;
    near: number;
    far: number;

    constructor(orientation: {
        position: vec3,
        target: vec3,
        up: vec3
    }, perspective: {
        fov: number,
        near: number,
        far: number
    }) {
        this.setPerspective(perspective);
        this.setOrientation(orientation);
    }

    setOrientation(orientation: {
        position: vec3,
        target: vec3,
        up: vec3
    }): void {
        this.position = vec3.clone(orientation.position);
        this.target = vec3.clone(orientation.target);
        this.up = vec3.clone(orientation.up);

        // Initialize orbit parameters
        this.radius = vec3.distance(orientation.position, orientation.target);

        // Calculate relative position from target
        const relativePos = vec3.sub(vec3.create(), orientation.position, orientation.target);

        // Calculate angles
        this.phi = Math.acos(relativePos[2] / this.radius);  // Changed from position[1]
        this.theta = Math.atan2(relativePos[0], relativePos[1]);  // Changed from position[0], position[2]
    }

    setPerspective(perspective: {
        fov: number,
        near: number,
        far: number
    }): void {
        this.fov = perspective.fov;
        this.near = perspective.near;
        this.far = perspective.far;
    }

    getOrientationMatrix(): mat4 {
        return mat4.lookAt(mat4.create(), this.position, this.target, this.up);
    }
    getPerspectiveMatrix(gl): mat4 {
        return mat4.perspective(
            mat4.create(),
            this.fov * Math.PI / 180,
            gl.canvas.width / gl.canvas.height,
            this.near,
            this.far
        )
    }

    orbit(deltaX: number, deltaY: number): void {
        // Update angles - note we swap the relationship here
        this.theta += deltaX * 0.01;  // Left/right movement affects azimuthal angle
        this.phi -= deltaY * 0.01;    // Up/down movement affects polar angle (note: + instead of -)

        // Clamp phi to avoid flipping and keep camera above ground
        this.phi = Math.max(0.01, Math.min(Math.PI - 0.01, this.phi));

        // Calculate new position using spherical coordinates
        const sinPhi = Math.sin(this.phi);
        const cosPhi = Math.cos(this.phi);
        const sinTheta = Math.sin(this.theta);
        const cosTheta = Math.cos(this.theta);

        // Update position in world space
        // Note: Changed coordinate mapping to match expected behavior
        this.position[0] = this.target[0] + this.radius * sinPhi * sinTheta;
        this.position[1] = this.target[1] + this.radius * sinPhi * cosTheta;
        this.position[2] = this.target[2] + this.radius * cosPhi;
    }

    zoom(delta: number): void {
        // Update radius with limits
        this.radius = Math.max(0.1, Math.min(1000, this.radius + delta * 0.1));

        // Update position based on new radius
        const direction = vec3.sub(vec3.create(), this.target, this.position);
        vec3.normalize(direction, direction);
        vec3.scaleAndAdd(this.position, this.target, direction, -this.radius);
    }

    pan(deltaX: number, deltaY: number): void {
        // Calculate right vector
        const forward = vec3.sub(vec3.create(), this.target, this.position);
        const right = vec3.cross(vec3.create(), forward, this.up);
        vec3.normalize(right, right);

        // Calculate actual up vector (not world up)
        const actualUp = vec3.cross(vec3.create(), right, forward);
        vec3.normalize(actualUp, actualUp);

        // Scale the movement based on distance
        const scale = this.radius * 0.002;

        // Move both position and target
        const movement = vec3.create();
        vec3.scaleAndAdd(movement, movement, right, -deltaX * scale);
        vec3.scaleAndAdd(movement, movement, actualUp, deltaY * scale);

        vec3.add(this.position, this.position, movement);
        vec3.add(this.target, this.target, movement);
    }
}

function useCamera(
    requestRender: () => void,
    camera: CameraParams | undefined,
    defaultCamera: CameraParams | undefined,
    callbacksRef
) {
    const isControlled = camera !== undefined;
    const initialCamera = isControlled ? camera : defaultCamera;

    if (!initialCamera) {
        throw new Error('Either camera or defaultCamera must be provided');
    }

    // Create camera instance once
    const cameraRef = useRef<OrbitCamera | null>(null);

    // Initialize camera once with default values
    useMemo(() => {
        const orientation = {
            position: Array.isArray(initialCamera.position)
                ? vec3.fromValues(...initialCamera.position)
                : vec3.clone(initialCamera.position),
            target: Array.isArray(initialCamera.target)
                ? vec3.fromValues(...initialCamera.target)
                : vec3.clone(initialCamera.target),
            up: Array.isArray(initialCamera.up)
                ? vec3.fromValues(...initialCamera.up)
                : vec3.clone(initialCamera.up)
        };

        const perspective = {
            fov: initialCamera.fov,
            near: initialCamera.near,
            far: initialCamera.far
        };

        cameraRef.current = new OrbitCamera(orientation, perspective);
    }, []); // Empty deps since we only want this on mount

    // Update camera when controlled props change
    useEffect(() => {
        if (!isControlled || !cameraRef.current) return;

        const orientation = {
            position: Array.isArray(camera.position)
                ? vec3.fromValues(...camera.position)
                : vec3.clone(camera.position),
            target: Array.isArray(camera.target)
                ? vec3.fromValues(...camera.target)
                : vec3.clone(camera.target),
            up: Array.isArray(camera.up)
                ? vec3.fromValues(...camera.up)
                : vec3.clone(camera.up)
        };

        cameraRef.current.setOrientation(orientation);
        cameraRef.current.setPerspective({
            fov: camera.fov,
            near: camera.near,
            far: camera.far
        });

        requestRender();
    }, [isControlled, camera]);

    const notifyCameraChange = useCallback(() => {
        const onCameraChange = callbacksRef.current.onCameraChange;
        if (!cameraRef.current || !onCameraChange) return;

        const camera = cameraRef.current;
        onCameraChange({
            position: [...camera.position] as [number, number, number],
            target: [...camera.target] as [number, number, number],
            up: [...camera.up] as [number, number, number],
            fov: camera.fov,
            near: camera.near,
            far: camera.far
        });
    }, []);

    const handleCameraMove = useCallback((action: (camera: OrbitCamera) => void) => {
        if (!cameraRef.current) return;
        action(cameraRef.current);

        if (isControlled) {
            notifyCameraChange();
        } else {
            requestRender();
        }
    }, [isControlled, notifyCameraChange, requestRender]);

    return {
        cameraRef,
        handleCameraMove
    };
}

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
    const pickPoint = useCallback((gl: WebGL2RenderingContext, camera: OrbitCamera, pixelCoords: [number, number]): number | null => {
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
            const perspectiveMatrix = camera.getPerspectiveMatrix(gl);
            const orientationMatrix = camera.getOrientationMatrix();
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
        decorationBlendModes: gl.getUniformLocation(program, 'uDecorationBlendModes'),
        decorationBlendStrengths: gl.getUniformLocation(program, 'uDecorationBlendStrengths'),

        // Decoration map uniforms
        decorationMap: gl.getUniformLocation(program, 'uDecorationMap'),
        decorationMapSize: gl.getUniformLocation(program, 'uDecorationMapSize'),

        // Decoration min sizes
        decorationMinSizes: gl.getUniformLocation(program, 'uDecorationMinSizes'),

        // Render mode
        renderMode: gl.getUniformLocation(program, 'uRenderMode'),
    };
}

// Add helper to convert blend mode string to int
function blendModeToInt(mode: DecorationGroup['blendMode']): number {
    switch (mode) {
        case 'replace': return 0;
        case 'multiply': return 1;
        case 'add': return 2;
        case 'screen': return 3;
        default: return 0;
    }
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
        if (renderFunctionRef.current) {
            // Cancel any pending render
            if (renderRAFRef.current) {
                cancelAnimationFrame(renderRAFRef.current);
            }

            renderRAFRef.current = requestAnimationFrame(() => {
                renderFunctionRef.current();

                const now = performance.now();
                if (lastRenderTime.current) {
                    const timeBetweenRenders = now - lastRenderTime.current;
                    updateDisplay(timeBetweenRenders);
                }
                lastRenderTime.current = now;

                renderRAFRef.current = null;
            });
        }
    }, []);

    const {
        cameraRef,
        handleCameraMove
    } = useCamera(requestRender, camera, defaultCamera, callbacksRef);

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
                handleCameraMove(camera => camera.orbit(e.movementX, e.movementY));
                break;
            case 'pan':
                onHover?.(null);
                handleCameraMove(camera => camera.pan(e.movementX, e.movementY));
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
        handleCameraMove(camera => camera.zoom(e.deltaY));
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
        const numPoints = points.xyz.length / 3;

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
        gl.bufferData(gl.ARRAY_BUFFER, points.xyz, gl.STATIC_DRAW);

        // Color buffer
        gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
        if (points.rgb) {
            const normalizedColors = new Float32Array(points.rgb.length);
            for (let i = 0; i < points.rgb.length; i++) {
                normalizedColors[i] = points.rgb[i] / 255.0;
            }
            gl.bufferData(gl.ARRAY_BUFFER, normalizedColors, gl.STATIC_DRAW);
        } else {
            const defaultColors = new Float32Array(points.xyz.length);
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
        disposeFns.push(pickingSystem.initPicking(gl, points.xyz.length / 3));

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
            const perspectiveMatrix = cameraRef.current.getPerspectiveMatrix(gl);
            const orientationMatrix = cameraRef.current.getOrientationMatrix()

            // Set all uniforms in one place
            gl.uniformMatrix4fv(uniformsRef.current.projection, false, perspectiveMatrix);
            gl.uniformMatrix4fv(uniformsRef.current.view, false, orientationMatrix);
            gl.uniform1f(uniformsRef.current.pointSize, pointSize);
            gl.uniform2f(uniformsRef.current.canvasSize, gl.canvas.width, gl.canvas.height);

            // Update decoration map
            updateDecorationMap(gl, points.xyz.length / 3);

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
            const blendModes = new Int32Array(MAX_DECORATIONS).fill(0);
            const blendStrengths = new Float32Array(MAX_DECORATIONS).fill(1.0);
            const minSizes = new Float32Array(MAX_DECORATIONS).fill(0.0);

            // Fill arrays with decoration data
            const numDecorations = Math.min(Object.keys(currentDecorations).length, MAX_DECORATIONS);
            Object.values(currentDecorations).slice(0, MAX_DECORATIONS).forEach((decoration, i) => {
                scales[i] = decoration.scale ?? 1.0;

                if (decoration.color) {
                    const baseIdx = i * 3;
                    colors[baseIdx] = decoration.color[0];
                    colors[baseIdx + 1] = decoration.color[1];
                    colors[baseIdx + 2] = decoration.color[2];
                }

                alphas[i] = decoration.alpha ?? 1.0;
                blendModes[i] = blendModeToInt(decoration.blendMode);
                blendStrengths[i] = decoration.blendStrength ?? 1.0;
                minSizes[i] = decoration.minSize ?? 0.0;
            });

            // Set uniforms

            gl.uniform1fv(uniformsRef.current.decorationScales, scales);
            gl.uniform3fv(uniformsRef.current.decorationColors, colors);
            gl.uniform1fv(uniformsRef.current.decorationAlphas, alphas);
            gl.uniform1iv(uniformsRef.current.decorationBlendModes, blendModes);
            gl.uniform1fv(uniformsRef.current.decorationBlendStrengths, blendStrengths);
            gl.uniform1i(uniformsRef.current.decorationCount, numDecorations);
            gl.uniform1fv(uniformsRef.current.decorationMinSizes, minSizes);

            // Ensure correct VAO is bound
            gl.bindVertexArray(vaoRef.current);
            gl.drawArrays(gl.POINTS, 0, points.xyz.length / 3);
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
