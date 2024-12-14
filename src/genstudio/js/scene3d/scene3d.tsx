import React, { useEffect, useRef, useCallback, useMemo, useState } from 'react';
import { mat4, vec3 } from 'gl-matrix';
import { createProgram, createPointIdBuffer } from './webgl-utils';
import { PointCloudData, CameraParams, PointCloudViewerProps, ShaderUniforms, PickingUniforms } from './types';
import { mainShaders, pickingShaders, MAX_DECORATIONS } from './shaders';
import { OrbitCamera } from './orbit-camera';
import { useContainerWidth, useDeepMemo } from '../utils';
import { FPSCounter, useFPSCounter } from './fps';


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

    const cameraParams = useMemo(() => ({
        position: Array.isArray(initialCamera.position)
            ? vec3.fromValues(...initialCamera.position)
            : vec3.clone(initialCamera.position),
        target: Array.isArray(initialCamera.target)
            ? vec3.fromValues(...initialCamera.target)
            : vec3.clone(initialCamera.target),
        up: Array.isArray(initialCamera.up)
            ? vec3.fromValues(...initialCamera.up)
            : vec3.clone(initialCamera.up),
        fov: initialCamera.fov,
        near: initialCamera.near,
        far: initialCamera.far
    }), [initialCamera]);

    const cameraRef = useRef<OrbitCamera | null>(null);

    // Initialize camera only once for uncontrolled mode
    useEffect(() => {
        if (!isControlled && !cameraRef.current) {
            cameraRef.current = new OrbitCamera(
                cameraParams.position,
                cameraParams.target,
                cameraParams.up
            );
        }
    }, []); // Empty deps since we only want this on mount for uncontrolled mode

    // Update camera only in controlled mode
    useEffect(() => {
        if (isControlled) {
            cameraRef.current = new OrbitCamera(
                cameraParams.position,
                cameraParams.target,
                cameraParams.up
            );
        }
    }, [isControlled, cameraParams]);

    const setupMatrices = useCallback((gl: WebGL2RenderingContext) => {
        const projectionMatrix = mat4.perspective(
            mat4.create(),
            cameraParams.fov * Math.PI / 180,
            gl.canvas.width / gl.canvas.height,
            cameraParams.near,
            cameraParams.far
        );

        const viewMatrix = cameraRef.current?.getViewMatrix() || mat4.create();

        return { projectionMatrix, viewMatrix };
    }, [cameraParams]);

    const notifyCameraChange = useCallback(() => {
        const onCameraChange = callbacksRef.current.onCameraChange
        if (!cameraRef.current || !onCameraChange) return;

        const camera = cameraRef.current;
        tempCamera.position = [...camera.position] as [number, number, number];
        tempCamera.target = [...camera.target] as [number, number, number];
        tempCamera.up = [...camera.up] as [number, number, number];
        tempCamera.fov = cameraParams.fov;
        tempCamera.near = cameraParams.near;
        tempCamera.far = cameraParams.far;

        onCameraChange(tempCamera);
    }, [cameraParams]);

    const handleCameraUpdate = useCallback((action: (camera: OrbitCamera) => void) => {
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
        setupMatrices,
        cameraParams,
        handleCameraUpdate
    };
}

function usePicking(
    gl: WebGL2RenderingContext | null,
    points: PointCloudData,
    pointSize: number,
    setupMatrices: (gl: WebGL2RenderingContext) => { projectionMatrix: mat4, viewMatrix: mat4 }
) {
    const pickingProgramRef = useRef<WebGLProgram | null>(null);
    const pickingVaoRef = useRef(null);
    const pickingFbRef = useRef<WebGLFramebuffer | null>(null);
    const pickingTextureRef = useRef<WebGLTexture | null>(null);
    const pickingUniformsRef = useRef<PickingUniforms | null>(null);

    // Initialize picking system
    useEffect(() => {
        if (!gl) return;

        // Create picking program
        const pickingProgram = createProgram(gl, pickingShaders.vertex, pickingShaders.fragment);
        if (!pickingProgram) {
            console.error('Failed to create picking program');
            return;
        }
        pickingProgramRef.current = pickingProgram;

        // Cache picking uniforms
        pickingUniformsRef.current = {
            projection: gl.getUniformLocation(pickingProgram, 'uProjectionMatrix'),
            view: gl.getUniformLocation(pickingProgram, 'uViewMatrix'),
            pointSize: gl.getUniformLocation(pickingProgram, 'uPointSize'),
            canvasSize: gl.getUniformLocation(pickingProgram, 'uCanvasSize')
        };

        // Create framebuffer and texture
        const { pickingFb, pickingTexture, depthBuffer } = setupPickingFramebuffer(gl);
        if (!pickingFb || !pickingTexture) return;

        pickingFbRef.current = pickingFb;
        pickingTextureRef.current = pickingTexture;

        return () => {
            gl.deleteProgram(pickingProgram);
            gl.deleteFramebuffer(pickingFb);
            gl.deleteTexture(pickingTexture);
            gl.deleteRenderbuffer(depthBuffer);
        };
    }, [gl]);

    const pickPoint = useCallback((x: number, y: number, radius: number = 5): number | null => {
        if (!gl || !pickingProgramRef.current || !pickingVaoRef.current || !pickingFbRef.current) {
            return null;
        }

        if (!(gl.canvas instanceof HTMLCanvasElement)) {
            console.error('Canvas must be an HTMLCanvasElement for picking');
            return null;
        }

        // Convert mouse coordinates to device pixels
        const rect = gl.canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        const pixelX = Math.floor((x - rect.left) * dpr);
        const pixelY = Math.floor((y - rect.top) * dpr);

        // Save WebGL state
        const currentFBO = gl.getParameter(gl.FRAMEBUFFER_BINDING);
        const currentViewport = gl.getParameter(gl.VIEWPORT);

        // Render to picking framebuffer
        gl.bindFramebuffer(gl.FRAMEBUFFER, pickingFbRef.current);
        gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        gl.enable(gl.DEPTH_TEST);

        // Set up picking shader
        gl.useProgram(pickingProgramRef.current);
        const { projectionMatrix, viewMatrix } = setupMatrices(gl);
        gl.uniformMatrix4fv(pickingUniformsRef.current.projection, false, projectionMatrix);
        gl.uniformMatrix4fv(pickingUniformsRef.current.view, false, viewMatrix);
        gl.uniform1f(pickingUniformsRef.current.pointSize, pointSize);
        gl.uniform2f(pickingUniformsRef.current.canvasSize, gl.canvas.width, gl.canvas.height);

        // Draw points
        gl.bindVertexArray(pickingVaoRef.current);
        gl.drawArrays(gl.POINTS, 0, points.xyz.length / 3);

        // Read pixels in the region around the cursor
        const size = radius * 2 + 1;
        const startX = Math.max(0, pixelX - radius);
        const startY = Math.max(0, gl.canvas.height - pixelY - radius);
        const pixels = new Uint8Array(size * size * 4);
        gl.readPixels(
            startX,
            startY,
            size,
            size,
            gl.RGBA,
            gl.UNSIGNED_BYTE,
            pixels
        );

        // Find closest point in the region
        let closestPoint: number | null = null;
        let minDistance = Infinity;
        const centerX = radius;
        const centerY = radius;

        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const i = (y * size + x) * 4;
                if (pixels[i + 3] > 0) { // If alpha > 0, there's a point here
                    const dx = x - centerX;
                    const dy = y - centerY;
                    const distance = dx * dx + dy * dy;

                    if (distance < minDistance) {
                        minDistance = distance;
                        closestPoint = pixels[i] +
                                     pixels[i + 1] * 256 +
                                     pixels[i + 2] * 256 * 256;
                    }
                }
            }
        }

        // Restore WebGL state
        gl.bindFramebuffer(gl.FRAMEBUFFER, currentFBO);
        gl.viewport(...currentViewport);


        return closestPoint;
    }, [gl, points.xyz.length, pointSize, setupMatrices]);

    return {
        pickingProgramRef,
        pickingVaoRef,
        pickingFbRef,
        pickingTextureRef,
        pickingUniformsRef,
        pickPoint
    };
}

// Helper function to set up picking framebuffer
function setupPickingFramebuffer(gl: WebGL2RenderingContext) {
    const pickingFb = gl.createFramebuffer();
    const pickingTexture = gl.createTexture();
    if (!pickingFb || !pickingTexture) return {};

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

    const depthBuffer = gl.createRenderbuffer();
    gl.bindRenderbuffer(gl.RENDERBUFFER, depthBuffer);
    gl.renderbufferStorage(
        gl.RENDERBUFFER,
        gl.DEPTH_COMPONENT24,
        gl.canvas.width,
        gl.canvas.height
    );

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

    return { pickingFb, pickingTexture, depthBuffer };
}

function setupBuffers(
    gl: WebGL2RenderingContext,
    points: PointCloudData
): { positionBuffer: WebGLBuffer, colorBuffer: WebGLBuffer } | null {
    const positionBuffer = gl.createBuffer();
    const colorBuffer = gl.createBuffer();

    if (!positionBuffer || !colorBuffer) {
        console.error('Failed to create buffers');
        return null;
    }

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

    return { positionBuffer, colorBuffer };
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
        decorationIndices: gl.getUniformLocation(program, 'uDecorationIndices'),
        decorationScales: gl.getUniformLocation(program, 'uDecorationScales'),
        decorationColors: gl.getUniformLocation(program, 'uDecorationColors'),
        decorationAlphas: gl.getUniformLocation(program, 'uDecorationAlphas'),
        decorationBlendModes: gl.getUniformLocation(program, 'uDecorationBlendModes'),
        decorationBlendStrengths: gl.getUniformLocation(program, 'uDecorationBlendStrengths'),
        decorationCount: gl.getUniformLocation(program, 'uDecorationCount'),

        // Decoration map uniforms
        decorationMap: gl.getUniformLocation(program, 'uDecorationMap'),
        decorationMapSize: gl.getUniformLocation(program, 'uDecorationMapSize'),

        // Decoration min sizes
        decorationMinSizes: gl.getUniformLocation(program, 'uDecorationMinSizes'),
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

    let finalWidth, finalHeight;

    // Case 1: Only height specified
    if (height && !width) {
      finalHeight = height;
      finalWidth = containerWidth;
    }

    // Case 2: Only width specified
    else if (width && !height) {
      finalWidth = width;
      finalHeight = width / aspectRatio;
    }

    // Case 3: Both dimensions specified
    else if (width && height) {
      finalWidth = width;
      finalHeight = height;
    }

    // Case 4: Neither dimension specified
    else {
      finalWidth = containerWidth;
      finalHeight = containerWidth / aspectRatio;
    }

    return {
      width: finalWidth,
      height: finalHeight,
      style: {
        // Only set explicit width if user provided it
        width: width ? `${width}px` : '100%',
        height: `${finalHeight}px`
      }
    };
  }

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


    const callbacksRef = useRef({})
    useEffect(() => {
        callbacksRef.current = {onPointHover, onPointClick, onCameraChange}
    },[onPointHover, onPointClick])

    const [containerRef, containerWidth] = useContainerWidth(1);
    const dimensions = useMemo(() => computeCanvasDimensions(containerWidth, width, height, aspectRatio), [containerWidth, width, height, aspectRatio])

    camera = useDeepMemo(camera)
    defaultCamera = useDeepMemo(defaultCamera)
    // decorations = useDeepMemo(decorations)

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

    useEffect(() => requestRender(), [points.xyz, points.rgb, camera, defaultCamera, decorations])

    const {
        cameraRef,
        setupMatrices,
        cameraParams,
        handleCameraUpdate
    } = useCamera(requestRender, camera, defaultCamera, callbacksRef);

    const canvasRef = useRef<HTMLCanvasElement>(null);
    const glRef = useRef<WebGL2RenderingContext>(null);
    const programRef = useRef<WebGLProgram>(null);
    const interactionState = useRef({
        isDragging: false,
        isPanning: false
    });
    const animationFrameRef = useRef<number>();
    const vaoRef = useRef(null);
    const uniformsRef = useRef<ShaderUniforms | null>(null);
    const mouseDownPositionRef = useRef<{x: number, y: number} | null>(null);
    const CLICK_THRESHOLD = 3; // Pixels of movement allowed before considering it a drag

    const { fpsDisplayRef, updateDisplay } = useFPSCounter();

    const {
        pickingProgramRef,
        pickingVaoRef,
        pickingFbRef,
        pickingTextureRef,
        pickingUniformsRef,
        pickPoint
    } = usePicking(glRef.current, points, pointSize, setupMatrices);

    // Add refs for decoration texture
    const decorationMapRef = useRef<WebGLTexture | null>(null);
    const decorationMapSizeRef = useRef<number>(0);

    // Helper to create/update decoration map texture
    const updateDecorationMap = useCallback((gl: WebGL2RenderingContext, numPoints: number) => {
        // Calculate texture size (power of 2 that can fit all points)
        const texSize = Math.ceil(Math.sqrt(numPoints));
        const size = Math.pow(2, Math.ceil(Math.log2(texSize)));
        decorationMapSizeRef.current = size;

        // Create mapping array (default to 0 = no decoration)
        const mapping = new Uint8Array(size * size).fill(0);

        // Fill in decoration mappings - make sure we're using the correct point indices
        Object.values(decorations).forEach((decoration, decorationIndex) => {
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
    }, [decorations]);

    // Update handleMouseMove to properly clear hover state and handle all cases
    const handleMouseMove = useCallback((e: MouseEvent) => {
        if (!cameraRef.current) return;
        const onHover = callbacksRef.current.onPointHover

        if (interactionState.current.isDragging) {
            onHover?.(null);
            handleCameraUpdate(camera => camera.orbit(e.movementX, e.movementY));
        } else if (interactionState.current.isPanning) {
            onHover?.(null);
            handleCameraUpdate(camera => camera.pan(e.movementX, e.movementY));
        } else if (onHover) {
            const pointIndex = pickPoint(e.clientX, e.clientY, 4); // Use consistent radius
            onHover?.(pointIndex);
        }
    }, [handleCameraUpdate, pickPoint, requestRender]);

    // Update handleMouseUp to use the same radius
    const handleMouseUp = useCallback((e: MouseEvent) => {
        const wasDragging = interactionState.current.isDragging;
        const wasPanning = interactionState.current.isPanning;

        const onClick = callbacksRef.current.onPointClick

        interactionState.current.isDragging = false;
        interactionState.current.isPanning = false;

        if (wasDragging && !wasPanning && mouseDownPositionRef.current && onClick) {
            const dx = e.clientX - mouseDownPositionRef.current.x;
            const dy = e.clientY - mouseDownPositionRef.current.y;
            const distance = Math.sqrt(dx * dx + dy * dy);

            if (distance < CLICK_THRESHOLD) {
                const pointIndex = pickPoint(e.clientX, e.clientY, 4); // Same radius as hover
                if (pointIndex !== null) {
                    onClick(pointIndex, e);
                }
            }
        }

        mouseDownPositionRef.current = null;
    }, [pickPoint]);

    const handleWheel = useCallback((e: WheelEvent) => {
        e.preventDefault();
        handleCameraUpdate(camera => camera.zoom(e.deltaY));
    }, [handleCameraUpdate]);

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

    useEffect(() => {
        // NOTE
        // The picking framebuffer updates should be decoupled from the main render setup. This effect is doing too much - it's both initializing WebGL and handling per-frame picking updates. These concerns should be separated into different effects or functions.

        if (!canvasRef.current) return;

        const gl = canvasRef.current.getContext('webgl2');
        if (!gl) {
            console.error('WebGL2 not supported');
            return null;
        }
        glRef.current = gl;

        // Create program and get uniforms
        const program = createProgram(gl, mainShaders.vertex, mainShaders.fragment);
        if (!program) {
            console.error('Failed to create shader program');
            return;
        }
        programRef.current = program;

        // Cache uniform locations
        uniformsRef.current = cacheUniformLocations(gl, program);

        // Set up buffers
        const buffers = setupBuffers(gl, points);
        if (!buffers) return;

        // Set up VAO
        const vao = gl.createVertexArray();
        gl.bindVertexArray(vao);
        vaoRef.current = vao;

        gl.bindBuffer(gl.ARRAY_BUFFER, buffers.positionBuffer);
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

        gl.bindBuffer(gl.ARRAY_BUFFER, buffers.colorBuffer);
        gl.enableVertexAttribArray(1);
        gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);

        // Point ID buffer for main VAO
        const mainPointIdBuffer = createPointIdBuffer(gl, points.xyz.length / 3, 2);

        // Create picking program
        const pickingProgram = createProgram(gl, pickingShaders.vertex, pickingShaders.fragment);
        if (!pickingProgram) {
            console.error('Failed to create picking program');
            return;
        }
        pickingProgramRef.current = pickingProgram;

        // Cache picking uniforms
        pickingUniformsRef.current = {
            projection: gl.getUniformLocation(pickingProgram, 'uProjectionMatrix'),
            view: gl.getUniformLocation(pickingProgram, 'uViewMatrix'),
            pointSize: gl.getUniformLocation(pickingProgram, 'uPointSize'),
            canvasSize: gl.getUniformLocation(pickingProgram, 'uCanvasSize')
        };

        // After setting up the main VAO, set up picking VAO:
        const pickingVao = gl.createVertexArray();
        gl.bindVertexArray(pickingVao);
        pickingVaoRef.current = pickingVao;

        // Position buffer (reuse the same buffer)
        gl.bindBuffer(gl.ARRAY_BUFFER, buffers.positionBuffer);
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

        // Point ID buffer
        const pickingPointIdBuffer = createPointIdBuffer(gl, points.xyz.length / 3, 1);

        // Restore main VAO binding
        gl.bindVertexArray(vao);

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
            gl.DEPTH_COMPONENT24, // Use 24-bit depth buffer for better precision
            gl.canvas.width,
            gl.canvas.height
        );

        // Attach both color and depth buffers to the framebuffer
        gl.bindFramebuffer(gl.FRAMEBUFFER, pickingFbRef.current);
        gl.framebufferTexture2D(
            gl.FRAMEBUFFER,
            gl.COLOR_ATTACHMENT0,
            gl.TEXTURE_2D,
            pickingTextureRef.current,
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

        // Render function
        function render() {
            if (!gl || !programRef.current || !cameraRef.current) return;



            gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
            gl.clearColor(backgroundColor[0], backgroundColor[1], backgroundColor[2], 1.0);
            gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
            gl.enable(gl.DEPTH_TEST);
            gl.enable(gl.BLEND);
            gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);  // Standard alpha blending

            gl.useProgram(programRef.current);

            // Set up matrices
            const { projectionMatrix, viewMatrix } = setupMatrices(gl);


            // Set all uniforms in one place
            gl.uniformMatrix4fv(uniformsRef.current.projection, false, projectionMatrix);
            gl.uniformMatrix4fv(uniformsRef.current.view, false, viewMatrix);
            gl.uniform1f(uniformsRef.current.pointSize, pointSize);
            gl.uniform2f(uniformsRef.current.canvasSize, gl.canvas.width, gl.canvas.height);

            // Update decoration map
            updateDecorationMap(gl, points.xyz.length / 3);

            // Set decoration map uniforms
            gl.activeTexture(gl.TEXTURE0);
            gl.bindTexture(gl.TEXTURE_2D, decorationMapRef.current);
            gl.uniform1i(uniformsRef.current.decorationMap, 0);
            gl.uniform1i(uniformsRef.current.decorationMapSize, decorationMapSizeRef.current);

            // Prepare decoration data
            const indices = new Int32Array(MAX_DECORATIONS).fill(-1);
            const scales = new Float32Array(MAX_DECORATIONS).fill(1.0);
            const colors = new Float32Array(MAX_DECORATIONS * 3).fill(-1);  // Use -1 to indicate no color override
            const alphas = new Float32Array(MAX_DECORATIONS).fill(1.0);
            const blendModes = new Int32Array(MAX_DECORATIONS).fill(0);
            const blendStrengths = new Float32Array(MAX_DECORATIONS).fill(1.0);
            const minSizes = new Float32Array(MAX_DECORATIONS).fill(0.0);

            // Fill arrays with decoration data
            const numDecorations = Math.min(Object.keys(decorations).length, MAX_DECORATIONS);
            Object.values(decorations).slice(0, MAX_DECORATIONS).forEach((decoration, i) => {
                // Set index (for now just use first index)
                indices[i] = decoration.indexes[0];

                // Set scale (default to 1.0)
                scales[i] = decoration.scale ?? 1.0;

                // Only set color if specified
                if (decoration.color) {
                    const baseIdx = i * 3;
                    colors[baseIdx] = decoration.color[0];
                    colors[baseIdx + 1] = decoration.color[1];
                    colors[baseIdx + 2] = decoration.color[2];
                }

                // Set alpha (default to 1.0)
                alphas[i] = decoration.alpha ?? 1.0;

                // Set blend mode and strength
                blendModes[i] = blendModeToInt(decoration.blendMode);
                blendStrengths[i] = decoration.blendStrength ?? 1.0;

                // Set minimum size (default to 0 = no minimum)
                minSizes[i] = decoration.minSize ?? 0.0;
            });

            // Set uniforms
            gl.uniform1iv(uniformsRef.current.decorationIndices, indices);
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

        renderFunctionRef.current = render
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
                if (pickingProgramRef.current) {
                    gl.deleteProgram(pickingProgramRef.current);
                    pickingProgramRef.current = null;
                }
                if (vao) {
                    gl.deleteVertexArray(vao);
                }
                if (pickingVao) {
                    gl.deleteVertexArray(pickingVao);
                }
                if (buffers.positionBuffer) {
                    gl.deleteBuffer(buffers.positionBuffer);
                }
                if (buffers.colorBuffer) {
                    gl.deleteBuffer(buffers.colorBuffer);
                }
                if (mainPointIdBuffer) {
                    gl.deleteBuffer(mainPointIdBuffer);
                }
                if (pickingPointIdBuffer) {
                    gl.deleteBuffer(pickingPointIdBuffer);
                }
                if (pickingFb) {
                    gl.deleteFramebuffer(pickingFb);
                }
                if (pickingTexture) {
                    gl.deleteTexture(pickingTexture);
                }
                if (depthBuffer) {
                    gl.deleteRenderbuffer(depthBuffer);
                }
            }
            if (canvasRef.current) {
                canvasRef.current.removeEventListener('mousedown', handleMouseDown);
                canvasRef.current.removeEventListener('mousemove', handleMouseMove);
                canvasRef.current.removeEventListener('mouseup', handleMouseUp);
                canvasRef.current.removeEventListener('wheel', handleWheel);
            }
        };

    }, [points.xyz,
        points.rgb,
        cameraParams,
        // IMPORTANT
        // this currently needs to be invalidated in order for 'picking' to work.
        // backgroundColor is an array whose identity always changes, which causes
        // this to invalidate.
        // this _should_ be ...backgroundColor (potentially) IF we can
        // figure out how to get picking to update when it needs to.
        backgroundColor,
        handleMouseMove,
        requestRender,
        pointSize]);

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

    // Add back handleMouseDown
    const handleMouseDown = useCallback((e: MouseEvent) => {
        if (e.button === 0 && !e.shiftKey) {  // Left click without shift
            mouseDownPositionRef.current = { x: e.clientX, y: e.clientY };
            interactionState.current.isDragging = true;
        } else if (e.button === 1 || (e.button === 0 && e.shiftKey)) {  // Middle click or shift+left click
            interactionState.current.isPanning = true;
        }
    }, []);

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

// Reuse this object to avoid allocations
const tempCamera: CameraParams = {
    position: [0, 0, 0],
    target: [0, 0, 0],
    up: [0, 1, 0],
    fov: 45,
    near: 0.1,
    far: 1000
};
