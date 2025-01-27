import * as glMatrix from 'gl-matrix';

export interface CameraParams {
    position: [number, number, number];
    target: [number, number, number];
    up: [number, number, number];
    fov: number;
    near: number;
    far: number;
}

export interface CameraState {
    position: glMatrix.vec3;
    target: glMatrix.vec3;
    up: glMatrix.vec3;
    radius: number;
    phi: number;
    theta: number;
    fov: number;
    near: number;
    far: number;
}

export const DEFAULT_CAMERA: CameraParams = {
    position: [1.5 * Math.sin(0.2) * Math.sin(1.0),
               1.5 * Math.cos(1.0),
               1.5 * Math.sin(0.2) * Math.cos(1.0)],
    target: [0, 0, 0],
    up: [0, 1, 0],
    fov: 60,
    near: 0.01,
    far: 100.0
};

export function createCameraState(params: CameraParams): CameraState {
    const position = glMatrix.vec3.fromValues(...params.position);
    const target = glMatrix.vec3.fromValues(...params.target);
    const up = glMatrix.vec3.fromValues(...params.up);

    const radius = glMatrix.vec3.distance(position, target);
    const dir = glMatrix.vec3.sub(glMatrix.vec3.create(), position, target);
    const phi = Math.acos(dir[1] / radius);
    const theta = Math.atan2(dir[0], dir[2]);

    return { position, target, up, radius, phi, theta, fov: params.fov, near: params.near, far: params.far };
}

export function createCameraParams(state: CameraState): CameraParams {
    return {
        position: Array.from(state.position) as [number, number, number],
        target: Array.from(state.target) as [number, number, number],
        up: Array.from(state.up) as [number, number, number],
        fov: state.fov,
        near: state.near,
        far: state.far
    };
}

export function orbit(camera: CameraState, deltaX: number, deltaY: number): CameraState {
    const newTheta = camera.theta - deltaX * 0.01;
    const newPhi = Math.max(0.01, Math.min(Math.PI - 0.01, camera.phi - deltaY * 0.01));

    const sinPhi = Math.sin(newPhi);
    const cosPhi = Math.cos(newPhi);
    const sinTheta = Math.sin(newTheta);
    const cosTheta = Math.cos(newTheta);

    const newPosition = glMatrix.vec3.fromValues(
        camera.target[0] + camera.radius * sinPhi * sinTheta,
        camera.target[1] + camera.radius * cosPhi,
        camera.target[2] + camera.radius * sinPhi * cosTheta
    );

    return {
        ...camera,
        position: newPosition,
        phi: newPhi,
        theta: newTheta
    };
}

export function pan(camera: CameraState, deltaX: number, deltaY: number): CameraState {
    const forward = glMatrix.vec3.sub(glMatrix.vec3.create(), camera.target, camera.position);
    const right = glMatrix.vec3.cross(glMatrix.vec3.create(), forward, camera.up);
    glMatrix.vec3.normalize(right, right);

    const actualUp = glMatrix.vec3.cross(glMatrix.vec3.create(), right, forward);
    glMatrix.vec3.normalize(actualUp, actualUp);

    // Scale movement by distance from target
    const scale = camera.radius * 0.002;
    const movement = glMatrix.vec3.create();
    glMatrix.vec3.scaleAndAdd(movement, movement, right, -deltaX * scale);
    glMatrix.vec3.scaleAndAdd(movement, movement, actualUp, deltaY * scale);

    // Update position and target
    const newPosition = glMatrix.vec3.add(glMatrix.vec3.create(), camera.position, movement);
    const newTarget = glMatrix.vec3.add(glMatrix.vec3.create(), camera.target, movement);

    return {
        ...camera,
        position: newPosition,
        target: newTarget
    };
}

export function zoom(camera: CameraState, deltaY: number): CameraState {
    const newRadius = Math.max(0.1, camera.radius * Math.exp(deltaY * 0.001));
    const direction = glMatrix.vec3.sub(glMatrix.vec3.create(), camera.position, camera.target);
    glMatrix.vec3.normalize(direction, direction);

    const newPosition = glMatrix.vec3.scaleAndAdd(
        glMatrix.vec3.create(),
        camera.target,
        direction,
        newRadius
    );

    return {
        ...camera,
        position: newPosition,
        radius: newRadius
    };
}

export function getViewMatrix(camera: CameraState): Float32Array {
    return glMatrix.mat4.lookAt(
        glMatrix.mat4.create(),
        camera.position,
        camera.target,
        camera.up
    ) as Float32Array;
}

// Add helper function to convert degrees to radians
function degreesToRadians(degrees: number): number {
    return degrees * (Math.PI / 180);
}

// Update getProjectionMatrix to handle degrees
export function getProjectionMatrix(camera: CameraState, aspect: number): Float32Array {
    return glMatrix.mat4.perspective(
        glMatrix.mat4.create(),
        degreesToRadians(camera.fov),  // Convert FOV to radians
        aspect,
        camera.near,
        camera.far
    ) as Float32Array;
}
