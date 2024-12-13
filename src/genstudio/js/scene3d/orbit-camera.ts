import { vec3, mat4 } from 'gl-matrix';

export class OrbitCamera {
    position: vec3;
    target: vec3;
    up: vec3;
    radius: number;
    phi: number;
    theta: number;

    constructor(position: vec3, target: vec3, up: vec3) {
        this.position = vec3.clone(position);
        this.target = vec3.clone(target);
        this.up = vec3.clone(up);

        // Initialize orbit parameters
        this.radius = vec3.distance(position, target);

        // Calculate relative position from target
        const relativePos = vec3.sub(vec3.create(), position, target);

        // Calculate angles
        this.phi = Math.acos(relativePos[2] / this.radius);  // Changed from position[1]
        this.theta = Math.atan2(relativePos[0], relativePos[1]);  // Changed from position[0], position[2]

    }

    getViewMatrix(): mat4 {
        return mat4.lookAt(mat4.create(), this.position, this.target, this.up);
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
