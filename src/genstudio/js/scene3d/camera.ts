import { useEffect, useRef, useCallback, useMemo } from 'react';
import { mat4, vec3 } from 'gl-matrix';
import { CameraParams, CameraState } from './types';

// Camera operations
export function createCamera(orientation: {
    position: vec3,
    target: vec3,
    up: vec3
}, perspective: {
    fov: number,
    near: number,
    far: number
}): CameraState {
    const radius = vec3.distance(orientation.position, orientation.target);
    const relativePos = vec3.sub(vec3.create(), orientation.position, orientation.target);
    const phi = Math.acos(relativePos[2] / radius);
    const theta = Math.atan2(relativePos[0], relativePos[1]);

    return {
        position: vec3.clone(orientation.position),
        target: vec3.clone(orientation.target),
        up: vec3.clone(orientation.up),
        radius,
        phi,
        theta,
        fov: perspective.fov,
        near: perspective.near,
        far: perspective.far
    };
}

export function setOrientation(camera: CameraState, orientation: {
    position: vec3,
    target: vec3,
    up: vec3
}): void {
    vec3.copy(camera.position, orientation.position);
    vec3.copy(camera.target, orientation.target);
    vec3.copy(camera.up, orientation.up);

    camera.radius = vec3.distance(orientation.position, orientation.target);
    const relativePos = vec3.sub(vec3.create(), orientation.position, orientation.target);
    camera.phi = Math.acos(relativePos[2] / camera.radius);
    camera.theta = Math.atan2(relativePos[0], relativePos[1]);
}

export function setPerspective(camera: CameraState, perspective: {
    fov: number,
    near: number,
    far: number
}): void {
    camera.fov = perspective.fov;
    camera.near = perspective.near;
    camera.far = perspective.far;
}

export function getOrientationMatrix(camera: CameraState): mat4 {
    return mat4.lookAt(mat4.create(), camera.position, camera.target, camera.up);
}

export function getPerspectiveMatrix(camera: CameraState, aspectRatio: number): mat4 {
    return mat4.perspective(
        mat4.create(),
        camera.fov * Math.PI / 180,
        aspectRatio,
        camera.near,
        camera.far
    );
}

export function orbit(camera: CameraState, deltaX: number, deltaY: number): void {
    camera.theta += deltaX * 0.01;
    camera.phi -= deltaY * 0.01;
    camera.phi = Math.max(0.01, Math.min(Math.PI - 0.01, camera.phi));

    const sinPhi = Math.sin(camera.phi);
    const cosPhi = Math.cos(camera.phi);
    const sinTheta = Math.sin(camera.theta);
    const cosTheta = Math.cos(camera.theta);

    camera.position[0] = camera.target[0] + camera.radius * sinPhi * sinTheta;
    camera.position[1] = camera.target[1] + camera.radius * sinPhi * cosTheta;
    camera.position[2] = camera.target[2] + camera.radius * cosPhi;
}

export function zoom(camera: CameraState, delta: number): void {
    camera.radius = Math.max(0.1, Math.min(1000, camera.radius + delta * 0.1));
    const direction = vec3.sub(vec3.create(), camera.target, camera.position);
    vec3.normalize(direction, direction);
    vec3.scaleAndAdd(camera.position, camera.target, direction, -camera.radius);
}

export function pan(camera: CameraState, deltaX: number, deltaY: number): void {
    const forward = vec3.sub(vec3.create(), camera.target, camera.position);
    const right = vec3.cross(vec3.create(), forward, camera.up);
    vec3.normalize(right, right);

    const actualUp = vec3.cross(vec3.create(), right, forward);
    vec3.normalize(actualUp, actualUp);

    const scale = camera.radius * 0.002;
    const movement = vec3.create();
    vec3.scaleAndAdd(movement, movement, right, -deltaX * scale);
    vec3.scaleAndAdd(movement, movement, actualUp, deltaY * scale);

    vec3.add(camera.position, camera.position, movement);
    vec3.add(camera.target, camera.target, movement);
}

export function useCamera(
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

    const cameraRef = useRef<CameraState | null>(null);

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

        cameraRef.current = createCamera(orientation, perspective);
    }, []);

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

        setOrientation(cameraRef.current, orientation);
        setPerspective(cameraRef.current, {
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

    const handleCameraMove = useCallback((action: (camera: CameraState) => void) => {
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

export default {
    useCamera,
    getPerspectiveMatrix,
    getOrientationMatrix,
    orbit,
    pan,
    zoom
};

export type { CameraState };
