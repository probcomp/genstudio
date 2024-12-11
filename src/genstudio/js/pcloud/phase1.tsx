import React, { useEffect, useRef, useCallback, useMemo } from 'react';
import { mat4, vec3 } from 'gl-matrix';

// Basic vertex shader for point cloud rendering
const vertexShader = `#version 300 es
uniform mat4 uProjectionMatrix;
uniform mat4 uViewMatrix;
uniform float uPointSize;

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;

out vec3 vColor;

void main() {
    vColor = color;
    gl_Position = uProjectionMatrix * uViewMatrix * vec4(position, 1.0);
    gl_PointSize = uPointSize;
}`;

// Fragment shader for point rendering
const fragmentShader = `#version 300 es
precision highp float;

in vec3 vColor;
out vec4 fragColor;

void main() {
    // Create a circular point with soft anti-aliasing
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    float r = dot(coord, coord);
    if (r > 1.0) {
        discard;
    }
    fragColor = vec4(vColor, 1.0);
}`;

interface PointCloudData {
    xyz: Float32Array;
    rgb?: Uint8Array;
}

interface CameraParams {
    position: vec3 | [number, number, number];
    target: vec3 | [number, number, number];
    up: vec3 | [number, number, number];
    fov: number;
    near: number;
    far: number;
}

interface PointCloudViewerProps {
    // For controlled mode
    camera?: CameraParams;
    onCameraChange?: (camera: CameraParams) => void;
    // For uncontrolled mode
    defaultCamera?: CameraParams;
    // Other props
    points: PointCloudData;
    backgroundColor?: [number, number, number];
    className?: string;
    pointSize?: number;
}

class OrbitCamera {
    position: vec3;
    target: vec3;
    up: vec3;
    radius: number;
    phi: number;
    theta: number;

    constructor(position: vec3, target: vec3, up: vec3) {
        this.position = position;
        this.target = target;
        this.up = up;

        // Initialize orbit parameters
        this.radius = vec3.distance(position, target);
        this.phi = Math.acos(this.position[1] / this.radius);
        this.theta = Math.atan2(this.position[0], this.position[2]);
    }

    getViewMatrix(): mat4 {
        return mat4.lookAt(mat4.create(), this.position, this.target, this.up);
    }

    orbit(deltaX: number, deltaY: number): void {
        // Update angles
        this.theta += deltaX * 0.01;
        this.phi += deltaY * 0.01;

        // Calculate new position using spherical coordinates
        const sinPhi = Math.sin(this.phi);
        const cosPhi = Math.cos(this.phi);
        const sinTheta = Math.sin(this.theta);
        const cosTheta = Math.cos(this.theta);

        this.position[0] = this.target[0] + this.radius * sinPhi * sinTheta;
        this.position[1] = this.target[1] + this.radius * cosPhi;
        this.position[2] = this.target[2] + this.radius * sinPhi * cosTheta;
    }

    zoom(delta: number): void {
        // Update radius
        this.radius = Math.max(0.1, this.radius + delta * 0.1);

        // Get direction vector from position to target
        const direction = vec3.sub(vec3.create(), this.target, this.position);
        vec3.normalize(direction, direction);

        // Scale direction by new radius and update position
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

        // Scale the movement
        const scale = this.radius * 0.002; // Adjust pan speed based on distance

        // Move both position and target
        const movement = vec3.create();
        vec3.scaleAndAdd(movement, movement, right, -deltaX * scale);
        vec3.scaleAndAdd(movement, movement, actualUp, deltaY * scale);

        vec3.add(this.position, this.position, movement);
        vec3.add(this.target, this.target, movement);
    }
}

export function PointCloudViewer({
    points,
    camera,
    defaultCamera,
    onCameraChange,
    backgroundColor = [0.1, 0.1, 0.1],
    className,
    pointSize = 4.0
}: PointCloudViewerProps) {
    // Track whether we're in controlled mode
    const isControlled = camera !== undefined;

    // Use defaultCamera for initial setup only
    const initialCamera = isControlled ? camera : defaultCamera;

    // Memoize camera parameters to avoid unnecessary updates
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

    const canvasRef = useRef<HTMLCanvasElement>(null);
    const glRef = useRef<WebGL2RenderingContext>(null);
    const programRef = useRef<WebGLProgram>(null);
    const cameraRef = useRef<OrbitCamera>(null);
    const interactionState = useRef({
        isDragging: false,
        isPanning: false,
        lastX: 0,
        lastY: 0
    });
    const fpsRef = useRef<HTMLDivElement>(null);
    const lastFrameTimeRef = useRef<number>(0);
    const frameCountRef = useRef<number>(0);
    const lastFpsUpdateRef = useRef<number>(0);
    const animationFrameRef = useRef<number>();
    const needsRenderRef = useRef<boolean>(true);
    const lastFrameTimesRef = useRef<number[]>([]);
    const MAX_FRAME_SAMPLES = 10;  // Keep last 10 frames for averaging

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

    // Update handleMouseMove dependency
    const handleMouseMove = useCallback((e: MouseEvent) => {
        if (!cameraRef.current) return;

        const deltaX = e.clientX - interactionState.current.lastX;
        const deltaY = e.clientY - interactionState.current.lastY;

        if (interactionState.current.isDragging) {
            handleCameraUpdate(camera => camera.orbit(deltaX, deltaY));
        } else if (interactionState.current.isPanning) {
            handleCameraUpdate(camera => camera.pan(deltaX, deltaY));
        }

        interactionState.current.lastX = e.clientX;
        interactionState.current.lastY = e.clientY;
    }, [handleCameraUpdate]);

    const handleWheel = useCallback((e: WheelEvent) => {
        e.preventDefault();
        handleCameraUpdate(camera => camera.zoom(e.deltaY));
    }, [handleCameraUpdate]);

    useEffect(() => {
        if (!canvasRef.current) return;

        // Initialize WebGL2 context
        const gl = canvasRef.current.getContext('webgl2');
        if (!gl) {
            console.error('WebGL2 not supported');
            return;
        }
        glRef.current = gl;

        // Create and compile shaders
        const program = gl.createProgram();
        if (!program) return;
        programRef.current = program;

        const vShader = gl.createShader(gl.VERTEX_SHADER);
        const fShader = gl.createShader(gl.FRAGMENT_SHADER);
        if (!vShader || !fShader) return;

        gl.shaderSource(vShader, vertexShader);
        gl.shaderSource(fShader, fragmentShader);
        gl.compileShader(vShader);
        gl.compileShader(fShader);
        gl.attachShader(program, vShader);
        gl.attachShader(program, fShader);
        gl.linkProgram(program);

        // Initialize camera
        cameraRef.current = new OrbitCamera(
            Array.isArray(initialCamera.position) ? vec3.fromValues(...initialCamera.position) : initialCamera.position,
            Array.isArray(initialCamera.target) ? vec3.fromValues(...initialCamera.target) : initialCamera.target,
            Array.isArray(initialCamera.up) ? vec3.fromValues(...initialCamera.up) : initialCamera.up
        );

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

        gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

        gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
        gl.enableVertexAttribArray(1);
        gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);

        // Render function
        function render(timestamp: number) {
            if (!gl || !program || !cameraRef.current) return;

            // Only render if needed
            if (!needsRenderRef.current) {
                animationFrameRef.current = requestAnimationFrame(render);
                return;
            }

            // Reset the flag
            needsRenderRef.current = false;

            // Track actual frame time
            const frameTime = timestamp - lastFrameTimeRef.current;
            lastFrameTimeRef.current = timestamp;

            if (frameTime > 0) {  // Avoid division by zero
                lastFrameTimesRef.current.push(frameTime);
                if (lastFrameTimesRef.current.length > MAX_FRAME_SAMPLES) {
                    lastFrameTimesRef.current.shift();
                }

                // Calculate average frame time and FPS
                const avgFrameTime = lastFrameTimesRef.current.reduce((a, b) => a + b, 0) / lastFrameTimesRef.current.length;
                const fps = Math.round(1000 / avgFrameTime);

                if (fpsRef.current) {
                    fpsRef.current.textContent = `${fps} FPS`;
                }
            }

            gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
            gl.clearColor(backgroundColor[0], backgroundColor[1], backgroundColor[2], 1.0);
            gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
            gl.enable(gl.DEPTH_TEST);

            gl.useProgram(program);

            // Set up matrices
            const projectionMatrix = mat4.perspective(
                mat4.create(),
                cameraParams.fov * Math.PI / 180,
                gl.canvas.width / gl.canvas.height,
                cameraParams.near,
                cameraParams.far
            );

            const viewMatrix = cameraRef.current.getViewMatrix();

            const projectionLoc = gl.getUniformLocation(program, 'uProjectionMatrix');
            const viewLoc = gl.getUniformLocation(program, 'uViewMatrix');
            const pointSizeLoc = gl.getUniformLocation(program, 'uPointSize');

            gl.uniformMatrix4fv(projectionLoc, false, projectionMatrix);
            gl.uniformMatrix4fv(viewLoc, false, viewMatrix);
            gl.uniform1f(pointSizeLoc, pointSize);

            gl.drawArrays(gl.POINTS, 0, points.xyz.length / 3);

            // Store the animation frame ID
            animationFrameRef.current = requestAnimationFrame(render);
        }

        // Start the render loop
        animationFrameRef.current = requestAnimationFrame(render);

        const handleMouseDown = (e: MouseEvent) => {
            if (e.button === 0 && !e.shiftKey) {
                interactionState.current.isDragging = true;
            } else if (e.button === 1 || (e.button === 0 && e.shiftKey)) {
                interactionState.current.isPanning = true;
            }
            interactionState.current.lastX = e.clientX;
            interactionState.current.lastY = e.clientY;
        };

        const handleMouseUp = () => {
            interactionState.current.isDragging = false;
            interactionState.current.isPanning = false;
        };

        canvasRef.current.addEventListener('mousedown', handleMouseDown);
        canvasRef.current.addEventListener('mousemove', handleMouseMove);
        canvasRef.current.addEventListener('mouseup', handleMouseUp);
        canvasRef.current.addEventListener('wheel', handleWheel, { passive: false });

        requestRender(); // Request initial render

        return () => {
            if (animationFrameRef.current) {
                cancelAnimationFrame(animationFrameRef.current);
            }
            if (!canvasRef.current) return;
            canvasRef.current.removeEventListener('mousedown', handleMouseDown);
            canvasRef.current.removeEventListener('mousemove', handleMouseMove);
            canvasRef.current.removeEventListener('mouseup', handleMouseUp);
            canvasRef.current.removeEventListener('wheel', handleWheel);
        };
    }, [points, initialCamera, backgroundColor, handleCameraUpdate, handleMouseMove, handleWheel, requestRender, pointSize]);

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
