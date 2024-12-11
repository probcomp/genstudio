import { vec3 } from 'gl-matrix';

export interface PointCloudData {
    xyz: Float32Array;
    rgb?: Uint8Array;
}

export interface CameraParams {
    position: vec3 | [number, number, number];
    target: vec3 | [number, number, number];
    up: vec3 | [number, number, number];
    fov: number;
    near: number;
    far: number;
}

export interface PointCloudViewerProps {
    // Data
    points: PointCloudData;

    // Camera control
    camera?: CameraParams;
    defaultCamera?: CameraParams;
    onCameraChange?: (camera: CameraParams) => void;

    // Appearance
    backgroundColor?: [number, number, number];
    className?: string;
    pointSize?: number;
    highlightColor?: [number, number, number];
    hoveredHighlightColor?: [number, number, number];

    // Interaction
    onPointClick?: (pointIndex: number, event: MouseEvent) => void;
    onPointHover?: (pointIndex: number | null) => void;
    pickingRadius?: number;
    highlights?: number[];
}
