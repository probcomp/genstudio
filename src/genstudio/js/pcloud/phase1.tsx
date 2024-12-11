import React, { useEffect, useRef, useCallback, useMemo } from 'react';
import { mat4, vec3 } from 'gl-matrix';
import { createProgram, createPointIdBuffer } from './webgl-utils';
import { PointCloudData, CameraParams, PointCloudViewerProps, ShaderUniforms } from './types';
import { mainShaders } from './shaders';
import { OrbitCamera } from './orbit-camera';
import { pickingShaders } from './shaders';

export function PointCloudViewer({
    points,
    camera,
    defaultCamera,
    onCameraChange,
    backgroundColor = [0.1, 0.1, 0.1],
    className,
    pointSize = 4.0,
    highlights = [],
    onPointClick,
    onPointHover,
    highlightColor = [1.0, 0.3, 0.0],
    hoveredHighlightColor = [1.0, 0.5, 0.0],
    pickingRadius = 5.0
}: PointCloudViewerProps) {
    // Track whether we're in controlled mode
    const isControlled = camera !== undefined;

    // Use defaultCamera for initial setup only
    const initialCamera = isControlled ? camera : defaultCamera;

    if (!initialCamera) {
        throw new Error('Either camera or defaultCamera must be provided');
    }

    const cameraParams = useMemo(() => {
        return {
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
        }
    }, [initialCamera]);

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

    const canvasRef = useRef<HTMLCanvasElement>(null);
    const glRef = useRef<WebGL2RenderingContext>(null);
    const programRef = useRef<WebGLProgram>(null);
    const cameraRef = useRef<OrbitCamera>(null);
    const interactionState = useRef({
        isDragging: false,
        isPanning: false
    });
    const fpsRef = useRef<HTMLDivElement>(null);
    const lastFrameTimeRef = useRef<number>(0);
    const frameCountRef = useRef<number>(0);
    const lastFpsUpdateRef = useRef<number>(0);
    const animationFrameRef = useRef<number>();
    const needsRenderRef = useRef<boolean>(true);
    const lastFrameTimesRef = useRef<number[]>([]);
    const MAX_FRAME_SAMPLES = 10;  // Keep last 10 frames for averaging
    const vaoRef = useRef(null);
    const pickingProgramRef = useRef<WebGLProgram | null>(null);
    const pickingVaoRef = useRef(null);
    const pickingFbRef = useRef<WebGLFramebuffer | null>(null);
    const pickingTextureRef = useRef<WebGLTexture | null>(null);
    const hoveredPointRef = useRef<number | null>(null);
    const uniformsRef = useRef<ShaderUniforms | null>(null);

    // Move requestRender before handleCameraUpdate
    const requestRender = useCallback(() => {
        needsRenderRef.current = true;
    }, []);

    // Optimize notification by reusing arrays
    const notifyCameraChange = useCallback(() => {
        if (!cameraRef.current || !onCameraChange) return;

        // Reuse arrays to avoid allocations
        const camera = cameraRef.current;
        tempCamera.position = [...camera.position] as [number, number, number];
        tempCamera.target = [...camera.target] as [number, number, number];
        tempCamera.up = [...camera.up] as [number, number, number];
        tempCamera.fov = cameraParams.fov;
        tempCamera.near = cameraParams.near;
        tempCamera.far = cameraParams.far;

        onCameraChange(tempCamera);
    }, [onCameraChange, cameraParams]);

    // Add back handleCameraUpdate
    const handleCameraUpdate = useCallback((action: (camera: OrbitCamera) => void) => {
        if (!cameraRef.current) return;
        action(cameraRef.current);
        requestRender();
        if (isControlled) {
            notifyCameraChange();
        }
    }, [isControlled, notifyCameraChange, requestRender]);

    // Add this function inside the component, before the useEffect:
    const pickPoint = useCallback((x: number, y: number): number | null => {
        const gl = glRef.current;
        if (!gl || !pickingProgramRef.current || !pickingVaoRef.current || !pickingFbRef.current || !cameraRef.current) {
            return null;
        }

        // Save current WebGL state
        const currentFBO = gl.getParameter(gl.FRAMEBUFFER_BINDING);

        // Switch to picking framebuffer
        gl.bindFramebuffer(gl.FRAMEBUFFER, pickingFbRef.current);
        gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        gl.enable(gl.DEPTH_TEST);
        gl.depthFunc(gl.LESS);
        gl.depthMask(true);

        // Use picking shader
        gl.useProgram(pickingProgramRef.current);

        // Use shared matrix setup
        const { projectionMatrix, viewMatrix } = setupMatrices(gl);

        // Set uniforms for picking shader
        gl.uniformMatrix4fv(gl.getUniformLocation(pickingProgramRef.current, 'uProjectionMatrix'), false, projectionMatrix);
        gl.uniformMatrix4fv(gl.getUniformLocation(pickingProgramRef.current, 'uViewMatrix'), false, viewMatrix);
        gl.uniform1f(gl.getUniformLocation(pickingProgramRef.current, 'uPointSize'), pointSize);
        gl.uniform2f(gl.getUniformLocation(pickingProgramRef.current, 'uCanvasSize'), gl.canvas.width, gl.canvas.height);

        // Draw points for picking
        gl.bindVertexArray(pickingVaoRef.current);
        gl.drawArrays(gl.POINTS, 0, points.xyz.length / 3);

        // Read pixel
        const rect = gl.canvas.getBoundingClientRect();
        const pixelX = Math.floor((x - rect.left) * gl.canvas.width / rect.width);
        const pixelY = Math.floor((rect.height - (y - rect.top)) * gl.canvas.height / rect.height);
        const pixel = new Uint8Array(4);
        gl.readPixels(pixelX, pixelY, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixel);

        // Restore previous state
        gl.bindFramebuffer(gl.FRAMEBUFFER, currentFBO);
        gl.useProgram(programRef.current);
        gl.bindVertexArray(vaoRef.current);
        requestRender();

        if (pixel[3] === 0) return null;
        return pixel[0] + pixel[1] * 256 + pixel[2] * 256 * 256;
    }, [points.xyz.length, pointSize, setupMatrices]);

    // Update the mouse handlers to properly handle clicks
    const handleMouseDown = useCallback((e: MouseEvent) => {
        if (e.button === 0 && !e.shiftKey) {  // Left click without shift
            if (onPointClick) {
                const pointIndex = pickPoint(e.clientX, e.clientY);
                if (pointIndex !== null) {
                    onPointClick(pointIndex, e);
                }
            }
            interactionState.current.isDragging = true;
        } else if (e.button === 1 || (e.button === 0 && e.shiftKey)) {  // Middle click or shift+left click
            interactionState.current.isPanning = true;
        }
    }, [pickPoint, onPointClick]);

    // Update handleMouseMove to include hover:
    const handleMouseMove = useCallback((e: MouseEvent) => {
        if (!cameraRef.current) return;

        if (interactionState.current.isDragging) {
            handleCameraUpdate(camera => camera.orbit(e.movementX, e.movementY));
        } else if (interactionState.current.isPanning) {
            handleCameraUpdate(camera => camera.pan(e.movementX, e.movementY));
        } else if (onPointHover) {
            const pointIndex = pickPoint(e.clientX, e.clientY);
            hoveredPointRef.current = pointIndex;
            onPointHover(pointIndex);
            requestRender();
        }
    }, [handleCameraUpdate, pickPoint, onPointHover, requestRender]);

    const handleWheel = useCallback((e: WheelEvent) => {
        e.preventDefault();
        handleCameraUpdate(camera => camera.zoom(e.deltaY));
    }, [handleCameraUpdate]);

    // Update handleMouseUp to properly reset state
    const handleMouseUp = useCallback((e: MouseEvent) => {
        // Only consider it a click if we didn't drag much
        const wasDragging = interactionState.current.isDragging;
        const wasPanning = interactionState.current.isPanning;

        interactionState.current.isDragging = false;
        interactionState.current.isPanning = false;

        // If we were dragging, don't trigger click
        if (wasDragging || wasPanning) {
            return;
        }
    }, []);

    // Add this function before useEffect
    const updateFPS = useCallback((timestamp: number) => {
        const frameTime = timestamp - lastFrameTimeRef.current;
        lastFrameTimeRef.current = timestamp;

        if (frameTime > 0) {
            lastFrameTimesRef.current.push(frameTime);
            if (lastFrameTimesRef.current.length > MAX_FRAME_SAMPLES) {
                lastFrameTimesRef.current.shift();
            }

            const avgFrameTime = lastFrameTimesRef.current.reduce((a, b) => a + b, 0) /
                lastFrameTimesRef.current.length;
            const fps = Math.round(1000 / avgFrameTime);

            if (fpsRef.current) {
                fpsRef.current.textContent = `${fps} FPS`;
            }
        }
    }, []);


    useEffect(() => {
        if (!canvasRef.current) return;

        // Initialize WebGL2 context
        const gl = canvasRef.current.getContext('webgl2');
        if (!gl) {
            console.error('WebGL2 not supported');
            return;
        }
        glRef.current = gl;

        // Ensure canvas and framebuffer are the same size
        const dpr = window.devicePixelRatio || 1;
        gl.canvas.width = gl.canvas.clientWidth * dpr;
        gl.canvas.height = gl.canvas.clientHeight * dpr;

        // Replace the manual shader compilation with createProgram
        const program = createProgram(gl, mainShaders.vertex, mainShaders.fragment);
        if (!program) {
            console.error('Failed to create shader program');
            return;
        }
        programRef.current = program;

        // Cache uniform locations during setup
        uniformsRef.current = {
            projection: gl.getUniformLocation(program, 'uProjectionMatrix'),
            view: gl.getUniformLocation(program, 'uViewMatrix'),
            pointSize: gl.getUniformLocation(program, 'uPointSize'),
            highlightedPoint: gl.getUniformLocation(program, 'uHighlightedPoint'),
            highlightColor: gl.getUniformLocation(program, 'uHighlightColor'),
            canvasSize: gl.getUniformLocation(program, 'uCanvasSize'),
            highlightedPoints: gl.getUniformLocation(program, 'uHighlightedPoints'),
            highlightCount: gl.getUniformLocation(program, 'uHighlightCount'),
            hoveredPoint: gl.getUniformLocation(program, 'uHoveredPoint'),
            hoveredHighlightColor: gl.getUniformLocation(program, 'uHoveredHighlightColor')
        };

        // Set up buffers
        const positionBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, points.xyz, gl.STATIC_DRAW);

        const colorBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
        if (points.rgb) {
            const normalizedColors = new Float32Array(points.rgb.length);
            for (let i = 0; i < points.rgb.length; i++) {
                normalizedColors[i] = points.rgb[i] / 255.0;
            }
            gl.bufferData(gl.ARRAY_BUFFER, normalizedColors, gl.STATIC_DRAW);
        } else {
            // Default color if none provided
            const defaultColors = new Float32Array(points.xyz.length);
            defaultColors.fill(0.7);
            gl.bufferData(gl.ARRAY_BUFFER, defaultColors, gl.STATIC_DRAW);
        }

        // Set up vertex attributes
        const vao = gl.createVertexArray();
        gl.bindVertexArray(vao);
        vaoRef.current = vao;

        gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

        gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
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

        // After setting up the main VAO, set up picking VAO:
        const pickingVao = gl.createVertexArray();
        gl.bindVertexArray(pickingVao);
        pickingVaoRef.current = pickingVao;

        // Position buffer (reuse the same buffer)
        gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

        // Point ID buffer
        const pickingPointIdBuffer = createPointIdBuffer(gl, points.xyz.length / 3, 1);

        // Restore main VAO binding
        gl.bindVertexArray(vao);

        // Create framebuffer and texture for picking
        const pickingFb = gl.createFramebuffer();
        const pickingTexture = gl.createTexture();
        if (!pickingFb || !pickingTexture) {
            console.error('Failed to create picking framebuffer');
            return;
        }
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
        function render(timestamp: number) {
            if (!gl || !programRef.current || !cameraRef.current) return;

            // Only render if needed
            if (!needsRenderRef.current) {
                animationFrameRef.current = requestAnimationFrame(render);
                return;
            }

            needsRenderRef.current = false;
            updateFPS(timestamp);

            gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
            gl.clearColor(backgroundColor[0], backgroundColor[1], backgroundColor[2], 1.0);
            gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
            gl.enable(gl.DEPTH_TEST);

            gl.useProgram(programRef.current);

            // Set up matrices
            const { projectionMatrix, viewMatrix } = setupMatrices(gl);


            // Set all uniforms in one place
            gl.uniformMatrix4fv(uniformsRef.current.projection, false, projectionMatrix);
            gl.uniformMatrix4fv(uniformsRef.current.view, false, viewMatrix);
            gl.uniform1f(uniformsRef.current.pointSize, pointSize);
            gl.uniform2f(uniformsRef.current.canvasSize, gl.canvas.width, gl.canvas.height);

            // Handle all highlight-related uniforms together
            const highlightArray = new Int32Array(100).fill(-1);
            highlights.slice(0, 100).forEach((idx, i) => {
                highlightArray[i] = idx;
            });

            gl.uniform1iv(uniformsRef.current.highlightedPoints, highlightArray);
            gl.uniform1i(uniformsRef.current.highlightCount, Math.min(highlights.length, 100));
            gl.uniform1i(uniformsRef.current.hoveredPoint, hoveredPointRef.current ?? -1);
            gl.uniform3fv(uniformsRef.current.highlightColor, highlightColor);
            gl.uniform3fv(uniformsRef.current.hoveredHighlightColor, hoveredHighlightColor);

            // Ensure correct VAO is bound
            gl.bindVertexArray(vaoRef.current);
            gl.drawArrays(gl.POINTS, 0, points.xyz.length / 3);

            animationFrameRef.current = requestAnimationFrame(render);
        }

        // Start the render loop
        animationFrameRef.current = requestAnimationFrame(render);

        canvasRef.current.addEventListener('mousedown', handleMouseDown);
        canvasRef.current.addEventListener('mousemove', handleMouseMove);
        canvasRef.current.addEventListener('mouseup', handleMouseUp);
        canvasRef.current.addEventListener('wheel', handleWheel, { passive: false });

        requestRender(); // Request initial render

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
                if (positionBuffer) {
                    gl.deleteBuffer(positionBuffer);
                }
                if (colorBuffer) {
                    gl.deleteBuffer(colorBuffer);
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
    }, [points, cameraParams, backgroundColor, handleCameraUpdate, handleMouseMove, handleWheel, requestRender, pointSize]);

    useEffect(() => {
        if (!canvasRef.current) return;

        const resizeObserver = new ResizeObserver(() => {
            requestRender();
        });

        resizeObserver.observe(canvasRef.current);

        return () => resizeObserver.disconnect();
    }, [requestRender]);

    return (
        <div style={{ position: 'relative' }}>
            <canvas
                ref={canvasRef}
                className={className}
                width={600}
                height={600}
            />
            <div
                ref={fpsRef}
                style={{
                    position: 'absolute',
                    top: '10px',
                    left: '10px',
                    color: 'white',
                    backgroundColor: 'rgba(0, 0, 0, 0.5)',
                    padding: '5px',
                    borderRadius: '3px',
                    fontSize: '14px'
                }}
            >
                0 FPS
            </div>
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
