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

export interface DecorationGroup {
  indexes: number[];

  // Visual modifications
  color?: [number, number, number];  // RGB color override
  alpha?: number;                    // 0-1 opacity
  scale?: number;                    // Size multiplier for points
  minSize?: number;           // Minimum size in pixels, regardless of distance

  // Color blend modes
  blendMode?: 'replace' | 'multiply' | 'add' | 'screen';
  blendStrength?: number;           // 0-1, how strongly to apply the blend
}

export interface DecorationGroups {
    [name: string]: DecorationGroup;
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

    // Dimensions
    width?: number;
    height?: number;
    aspectRatio?: number;

    // Interaction
    onPointClick?: (pointIndex: number, event: MouseEvent) => void;
    onPointHover?: (pointIndex: number | null) => void;
    pickingRadius?: number;
    decorations?: DecorationGroups;
}

export interface ShaderUniforms {
    projection: WebGLUniformLocation | null;
    view: WebGLUniformLocation | null;
    pointSize: WebGLUniformLocation | null;
    canvasSize: WebGLUniformLocation | null;

    // Decoration uniforms
    decorationIndices: WebGLUniformLocation | null;
    decorationScales: WebGLUniformLocation | null;
    decorationColors: WebGLUniformLocation | null;
    decorationAlphas: WebGLUniformLocation | null;
    decorationBlendModes: WebGLUniformLocation | null;
    decorationBlendStrengths: WebGLUniformLocation | null;
    decorationCount: WebGLUniformLocation | null;
}

export interface PickingUniforms {
    projection: WebGLUniformLocation | null;
    view: WebGLUniformLocation | null;
    pointSize: WebGLUniformLocation | null;
    canvasSize: WebGLUniformLocation | null;
}
