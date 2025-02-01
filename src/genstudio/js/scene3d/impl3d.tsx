/// <reference types="react" />

import * as glMatrix from 'gl-matrix';
import React, {
  // DO NOT require MouseEvent
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState
} from 'react';
import { throttle } from '../utils';
import { createCubeGeometry, createBeamGeometry, createSphereGeometry, createTorusGeometry } from './geometry';

import {
  CameraParams,
  CameraState,
  DEFAULT_CAMERA,
  createCameraParams,
  createCameraState,
  orbit,
  pan,
  zoom
} from './camera3d';

/******************************************************
 * 1) Types and Interfaces
 ******************************************************/

interface BaseComponentConfig {
  /**
   * Per-instance RGB color values as a Float32Array of RGB triplets.
   * Each instance requires 3 consecutive values in the range [0,1].
   */
  colors?: Float32Array;

  /**
   * Per-instance alpha (opacity) values.
   * Each value should be in the range [0,1].
   */
  alphas?: Float32Array;

  /**
   * Per-instance scale multipliers.
   * These multiply the base size/radius of each instance.
   */
  scales?: Float32Array;

  /**
   * Default RGB color applied to all instances without specific colors.
   * Values should be in range [0,1]. Defaults to [1,1,1] (white).
   */
  color?: [number, number, number];

  /**
   * Default alpha (opacity) for all instances without specific alpha.
   * Should be in range [0,1]. Defaults to 1.0.
   */
  alpha?: number;

  /**
   * Default scale multiplier for all instances without specific scale.
   * Defaults to 1.0.
   */
  scale?: number;

  /**
   * Callback fired when the mouse hovers over an instance.
   * The index parameter is the instance index, or null when hover ends.
   */
  onHover?: (index: number|null) => void;

  /**
   * Callback fired when an instance is clicked.
   * The index parameter is the clicked instance index.
   */
  onClick?: (index: number) => void;

  /**
   * Optional array of decorations to apply to specific instances.
   * Decorations can override colors, alpha, and scale for individual instances.
   */
  decorations?: Decoration[];
}

function getBaseDefaults(config: Partial<BaseComponentConfig>): Required<Omit<BaseComponentConfig, 'colors' | 'alphas' | 'scales' | 'decorations' | 'onHover' | 'onClick'>> {
  return {
    color: config.color ?? [1, 1, 1],
    alpha: config.alpha ?? 1.0,
    scale: config.scale ?? 1.0,
  };
}

function getColumnarParams(elem: BaseComponentConfig, count: number) {

  const hasValidColors = elem.colors instanceof Float32Array && elem.colors.length >= count * 3;
  const hasValidAlphas = elem.alphas instanceof Float32Array && elem.alphas.length >= count;
  const hasValidScales = elem.scales instanceof Float32Array && elem.scales.length >= count;

  return {
    colors: hasValidColors ? elem.colors : null,
    alphas: hasValidAlphas ? elem.alphas : null,
    scales: hasValidScales ? elem.scales : null
  };
}

export interface BufferInfo {
  buffer: GPUBuffer;
  offset: number;
  stride: number;
}

export interface RenderObject {
  pipeline?: GPURenderPipeline;
  vertexBuffers: Partial<[GPUBuffer, BufferInfo]>;  // Allow empty or partial arrays
  indexBuffer?: GPUBuffer;
  vertexCount?: number;
  indexCount?: number;
  instanceCount?: number;

  pickingPipeline?: GPURenderPipeline;
  pickingVertexBuffers: Partial<[GPUBuffer, BufferInfo]>;  // Allow empty or partial arrays
  pickingIndexBuffer?: GPUBuffer;
  pickingVertexCount?: number;
  pickingIndexCount?: number;
  pickingInstanceCount?: number;

  componentIndex: number;
  pickingDataStale: boolean;
}

export interface DynamicBuffers {
  renderBuffer: GPUBuffer;
  pickingBuffer: GPUBuffer;
  renderOffset: number;  // Current offset into render buffer
  pickingOffset: number; // Current offset into picking buffer
}

export interface SceneInnerProps {
  /** Array of 3D components to render in the scene */
  components: ComponentConfig[];

  /** Width of the container in pixels */
  containerWidth: number;

  /** Height of the container in pixels */
  containerHeight: number;

  /** Optional CSS styles to apply to the canvas */
  style?: React.CSSProperties;

  /** Optional controlled camera state. If provided, the component becomes controlled */
  camera?: CameraParams;

  /** Default camera configuration used when uncontrolled */
  defaultCamera?: CameraParams;

  /** Callback fired when camera parameters change */
  onCameraChange?: (camera: CameraParams) => void;

  /** Callback fired after each frame render with the render time in milliseconds */
  onFrameRendered?: (renderTime: number) => void;
}

/******************************************************
 * 2) Constants and Camera Functions
 ******************************************************/

/**
 * Global lighting configuration for the 3D scene.
 * Uses a simple Blinn-Phong lighting model with ambient, diffuse, and specular components.
 */
const LIGHTING = {
    /** Ambient light intensity, affects overall scene brightness */
    AMBIENT_INTENSITY: 0.4,

    /** Diffuse light intensity, affects surface shading based on light direction */
    DIFFUSE_INTENSITY: 0.6,

    /** Specular highlight intensity */
    SPECULAR_INTENSITY: 0.2,

    /** Specular power/shininess, higher values create sharper highlights */
    SPECULAR_POWER: 20.0,

    /** Light direction components relative to camera */
    DIRECTION: {
        /** Right component of light direction */
        RIGHT: 0.2,
        /** Up component of light direction */
        UP: 0.5,
        /** Forward component of light direction */
        FORWARD: 0,
    }
} as const;

/******************************************************
 * 3) Data Structures & Primitive Specs
 ******************************************************/

// Common shader code templates
const cameraStruct = /*wgsl*/`
struct Camera {
  mvp: mat4x4<f32>,
  cameraRight: vec3<f32>,
  _pad1: f32,
  cameraUp: vec3<f32>,
  _pad2: f32,
  lightDir: vec3<f32>,
  _pad3: f32,
  cameraPos: vec3<f32>,
  _pad4: f32,
};
@group(0) @binding(0) var<uniform> camera : Camera;`;

const lightingConstants = /*wgsl*/`
const AMBIENT_INTENSITY = ${LIGHTING.AMBIENT_INTENSITY}f;
const DIFFUSE_INTENSITY = ${LIGHTING.DIFFUSE_INTENSITY}f;
const SPECULAR_INTENSITY = ${LIGHTING.SPECULAR_INTENSITY}f;
const SPECULAR_POWER = ${LIGHTING.SPECULAR_POWER}f;`;

const lightingCalc = /*wgsl*/`
fn calculateLighting(baseColor: vec3<f32>, normal: vec3<f32>, worldPos: vec3<f32>) -> vec3<f32> {
  let N = normalize(normal);
  let L = normalize(camera.lightDir);
  let V = normalize(camera.cameraPos - worldPos);

  let lambert = max(dot(N, L), 0.0);
  let ambient = AMBIENT_INTENSITY;
  var color = baseColor * (ambient + lambert * DIFFUSE_INTENSITY);

  let H = normalize(L + V);
  let spec = pow(max(dot(N, H), 0.0), SPECULAR_POWER);
  color += vec3<f32>(1.0) * spec * SPECULAR_INTENSITY;

  return color;
}`;

interface PrimitiveSpec<E> {
  /**
   * Returns the number of instances in this component.
   * Used to allocate buffers and determine draw call parameters.
   */
  getCount(component: E): number;

  /**
   * Builds vertex buffer data for rendering.
   * Returns a Float32Array containing interleaved vertex attributes,
   * or null if the component has no renderable data.
   */
  buildRenderData(component: E): Float32Array | null;

  /**
   * Builds vertex buffer data for GPU-based picking.
   * Returns a Float32Array containing picking IDs and instance data,
   * or null if the component doesn't support picking.
   * @param baseID Starting ID for this component's instances
   */
  buildPickingData(component: E, baseID: number): Float32Array | null;

  /**
   * Default WebGPU rendering configuration for this primitive type.
   * Specifies face culling and primitive topology.
   */
  renderConfig: RenderConfig;

  /**
   * Creates or retrieves a cached WebGPU render pipeline for this primitive.
   * @param device The WebGPU device
   * @param bindGroupLayout Layout for uniform bindings
   * @param cache Pipeline cache to prevent duplicate creation
   */
  getRenderPipeline(
    device: GPUDevice,
    bindGroupLayout: GPUBindGroupLayout,
    cache: Map<string, PipelineCacheEntry>
  ): GPURenderPipeline;

  /**
   * Creates or retrieves a cached WebGPU pipeline for picking.
   * @param device The WebGPU device
   * @param bindGroupLayout Layout for uniform bindings
   * @param cache Pipeline cache to prevent duplicate creation
   */
  getPickingPipeline(
    device: GPUDevice,
    bindGroupLayout: GPUBindGroupLayout,
    cache: Map<string, PipelineCacheEntry>
  ): GPURenderPipeline;

  /**
   * Creates the base geometry buffers needed for this primitive type.
   * These buffers are shared across all instances of the primitive.
   */
  createGeometryResource(device: GPUDevice): { vb: GPUBuffer; ib: GPUBuffer; indexCount: number };
}

interface Decoration {
  indexes: number[];
  color?: [number, number, number];
  alpha?: number;
  scale?: number;
}

/** Helper function to apply decorations to an array of instances */
function applyDecorations(
  decorations: Decoration[] | undefined,
  instanceCount: number,
  setter: (i: number, dec: Decoration) => void
) {
  if (!decorations) return;
  for (const dec of decorations) {
    for (const idx of dec.indexes) {
      if (idx < 0 || idx >= instanceCount) continue;
      setter(idx, dec);
    }
  }
}

/** Configuration for how a primitive type should be rendered */
interface RenderConfig {
  /** How faces should be culled */
  cullMode: GPUCullMode;
  /** How vertices should be interpreted */
  topology: GPUPrimitiveTopology;
}

/** ===================== POINT CLOUD ===================== **/


const billboardVertCode = /*wgsl*/`
${cameraStruct}

struct VSOut {
  @builtin(position) Position: vec4<f32>,
  @location(0) color: vec3<f32>,
  @location(1) alpha: f32
};

@vertex
fn vs_main(
  @location(0) localPos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) instancePos: vec3<f32>,
  @location(3) col: vec3<f32>,
  @location(4) alpha: f32,
  @location(5) size: f32
)-> VSOut {
  // Create camera-facing orientation
  let right = camera.cameraRight;
  let up = camera.cameraUp;

  // Transform quad vertices to world space
  let scaledRight = right * (localPos.x * size);
  let scaledUp = up * (localPos.y * size);
  let worldPos = instancePos + scaledRight + scaledUp;

  var out: VSOut;
  out.Position = camera.mvp * vec4<f32>(worldPos, 1.0);
  out.color = col;
  out.alpha = alpha;
  return out;
}`;

const billboardPickingVertCode = /*wgsl*/`
@vertex
fn vs_pointcloud(
  @location(0) localPos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) instancePos: vec3<f32>,
  @location(3) pickID: f32,
  @location(4) size: f32
)-> VSOut {
  // Create camera-facing orientation
  let right = camera.cameraRight;
  let up = camera.cameraUp;

  // Transform quad vertices to world space
  let scaledRight = right * (localPos.x * size);
  let scaledUp = up * (localPos.y * size);
  let worldPos = instancePos + scaledRight + scaledUp;

  var out: VSOut;
  out.pos = camera.mvp * vec4<f32>(worldPos, 1.0);
  out.pickID = pickID;
  return out;
}`;

const billboardFragCode = /*wgsl*/`
@fragment
fn fs_main(@location(0) color: vec3<f32>, @location(1) alpha: f32)-> @location(0) vec4<f32> {
  return vec4<f32>(color, alpha);
}`;

export interface PointCloudComponentConfig extends BaseComponentConfig {
  type: 'PointCloud';
  positions: Float32Array;
  sizes?: Float32Array;     // Per-point sizes
  size?: number;           // Default size, defaults to 0.02
}

const pointCloudSpec: PrimitiveSpec<PointCloudComponentConfig> = {
  getCount(elem) {
    return elem.positions.length / 3;
  },

  buildRenderData(elem) {
    const count = elem.positions.length / 3;
    if(count === 0) return null;

    const defaults = getBaseDefaults(elem);
    const { colors, alphas, scales } = getColumnarParams(elem, count);

    const size = elem.size ?? 0.02;
    const sizes = elem.sizes instanceof Float32Array && elem.sizes.length >= count ? elem.sizes : null;

    const arr = new Float32Array(count * 8);
    for(let i=0; i<count; i++) {
      arr[i*8+0] = elem.positions[i*3+0];
      arr[i*8+1] = elem.positions[i*3+1];
      arr[i*8+2] = elem.positions[i*3+2];

      if(colors) {
        arr[i*8+3] = colors[i*3+0];
        arr[i*8+4] = colors[i*3+1];
        arr[i*8+5] = colors[i*3+2];
      } else {
        arr[i*8+3] = defaults.color[0];
        arr[i*8+4] = defaults.color[1];
        arr[i*8+5] = defaults.color[2];
      }

      arr[i*8+6] = alphas ? alphas[i] : defaults.alpha;
      const pointSize = sizes  ? sizes[i] : size;
      const scale = scales ? scales[i] : defaults.scale;
      arr[i*8+7] = pointSize * scale;
    }

    // Apply decorations last
    applyDecorations(elem.decorations, count, (idx, dec) => {
      if(dec.color) {
        arr[idx*8+3] = dec.color[0];
        arr[idx*8+4] = dec.color[1];
        arr[idx*8+5] = dec.color[2];
      }
      if(dec.alpha !== undefined) {
        arr[idx*8+6] = dec.alpha;
      }
      if(dec.scale !== undefined) {
        arr[idx*8+7] *= dec.scale;
      }
    });

    return arr;
  },

  buildPickingData(elem, baseID) {
    const count = elem.positions.length / 3;
    if(count === 0) return null;

    const size = elem.size ?? 0.02;
    const arr = new Float32Array(count * 5);

    // Check array validities once before the loop
    const hasValidSizes = elem.sizes && elem.sizes.length >= count;
    const sizes = hasValidSizes ? elem.sizes : null;

    for(let i=0; i<count; i++) {
      arr[i*5+0] = elem.positions[i*3+0];
      arr[i*5+1] = elem.positions[i*3+1];
      arr[i*5+2] = elem.positions[i*3+2];
      arr[i*5+3] = baseID + i;
      arr[i*5+4] = sizes?.[i] ?? size;
    }
    return arr;
  },

  // Rendering configuration
  renderConfig: {
    cullMode: 'none',
    topology: 'triangle-list'
  },

  // Pipeline creation methods
  getRenderPipeline(device, bindGroupLayout, cache) {
    const format = navigator.gpu.getPreferredCanvasFormat();
    return getOrCreatePipeline(
      device,
      "PointCloudShading",
      () => createRenderPipeline(device, bindGroupLayout, {
        vertexShader: billboardVertCode,
        fragmentShader: billboardFragCode,
        vertexEntryPoint: 'vs_main',
        fragmentEntryPoint: 'fs_main',
        bufferLayouts: [POINT_CLOUD_GEOMETRY_LAYOUT, POINT_CLOUD_INSTANCE_LAYOUT],
        primitive: this.renderConfig,
        blend: {
          color: {
            srcFactor: 'src-alpha',
            dstFactor: 'one-minus-src-alpha',
            operation: 'add'
          },
          alpha: {
            srcFactor: 'one',
            dstFactor: 'one-minus-src-alpha',
            operation: 'add'
          }
        },
        depthStencil: {
          format: 'depth24plus',
          depthWriteEnabled: true,
          depthCompare: 'less'
        }
      }, format),
      cache
    );
  },

  getPickingPipeline(device, bindGroupLayout, cache) {
    return getOrCreatePipeline(
      device,
      "PointCloudPicking",
      () => createRenderPipeline(device, bindGroupLayout, {
        vertexShader: pickingVertCode,
        fragmentShader: pickingVertCode,
        vertexEntryPoint: 'vs_pointcloud',
        fragmentEntryPoint: 'fs_pick',
        bufferLayouts: [POINT_CLOUD_GEOMETRY_LAYOUT, POINT_CLOUD_PICKING_INSTANCE_LAYOUT]
      }, 'rgba8unorm'),
      cache
    );
  },

  createGeometryResource(device) {
    return createBuffers(device, {
      vertexData: new Float32Array([
        -0.5, -0.5, 0.0,     0.0, 0.0, 1.0,
         0.5, -0.5, 0.0,     0.0, 0.0, 1.0,
        -0.5,  0.5, 0.0,     0.0, 0.0, 1.0,
         0.5,  0.5, 0.0,     0.0, 0.0, 1.0
      ]),
      indexData: new Uint16Array([0,1,2, 2,1,3])
    });
  }
};

/** ===================== ELLIPSOID ===================== **/


const ellipsoidVertCode = /*wgsl*/`
${cameraStruct}

struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) baseColor: vec3<f32>,
  @location(3) alpha: f32,
  @location(4) worldPos: vec3<f32>,
  @location(5) instancePos: vec3<f32>
};

@vertex
fn vs_main(
  @location(0) inPos: vec3<f32>,
  @location(1) inNorm: vec3<f32>,
  @location(2) iPos: vec3<f32>,
  @location(3) iScale: vec3<f32>,
  @location(4) iColor: vec3<f32>,
  @location(5) iAlpha: f32
)-> VSOut {
  let worldPos = iPos + (inPos * iScale);
  let scaledNorm = normalize(inNorm / iScale);

  var out: VSOut;
  out.pos = camera.mvp * vec4<f32>(worldPos,1.0);
  out.normal = scaledNorm;
  out.baseColor = iColor;
  out.alpha = iAlpha;
  out.worldPos = worldPos;
  out.instancePos = iPos;
  return out;
}`;

const ellipsoidPickingVertCode = /*wgsl*/`
@vertex
fn vs_ellipsoid(
  @location(0) inPos: vec3<f32>,
  @location(1) inNorm: vec3<f32>,
  @location(2) iPos: vec3<f32>,
  @location(3) iScale: vec3<f32>,
  @location(4) pickID: f32
)-> VSOut {
  let wp = iPos + (inPos * iScale);
  var out: VSOut;
  out.pos = camera.mvp*vec4<f32>(wp,1.0);
  out.pickID = pickID;
  return out;
}`;

const ellipsoidFragCode = /*wgsl*/`
${cameraStruct}
${lightingConstants}
${lightingCalc}

@fragment
fn fs_main(
  @location(1) normal: vec3<f32>,
  @location(2) baseColor: vec3<f32>,
  @location(3) alpha: f32,
  @location(4) worldPos: vec3<f32>,
  @location(5) instancePos: vec3<f32>
)-> @location(0) vec4<f32> {
  let color = calculateLighting(baseColor, normal, worldPos);
  return vec4<f32>(color, alpha);
}`;


export interface EllipsoidComponentConfig extends BaseComponentConfig {
  type: 'Ellipsoid';
  centers: Float32Array;
  radii?: Float32Array;     // Per-ellipsoid radii
  radius?: [number, number, number]; // Default radius, defaults to [1,1,1]
}

const ellipsoidSpec: PrimitiveSpec<EllipsoidComponentConfig> = {
  getCount(elem) {
    return elem.centers.length / 3;
  },

  buildRenderData(elem) {
    const count = elem.centers.length / 3;
    if(count === 0) return null;

    const defaults = getBaseDefaults(elem);
    const { colors, alphas, scales } = getColumnarParams(elem, count);

    const defaultRadius = elem.radius ?? [1, 1, 1];
    const radii = elem.radii && elem.radii.length >= count * 3 ? elem.radii : null;

    const arr = new Float32Array(count * 10);
    for(let i = 0; i < count; i++) {

      // Centers
      arr[i*10+0] = elem.centers[i*3+0];
      arr[i*10+1] = elem.centers[i*3+1];
      arr[i*10+2] = elem.centers[i*3+2];

      // Radii (with scale)
      const scale = scales ? scales[i] : defaults.scale;

      if(radii) {
        arr[i*10+3] = radii[i*3+0] * scale;
        arr[i*10+4] = radii[i*3+1] * scale;
        arr[i*10+5] = radii[i*3+2] * scale;
      } else {
        arr[i*10+3] = defaultRadius[0] * scale;
        arr[i*10+4] = defaultRadius[1] * scale;
        arr[i*10+5] = defaultRadius[2] * scale;
      }

      if(colors) {
        arr[i*10+6] = colors[i*3+0];
        arr[i*10+7] = colors[i*3+1];
        arr[i*10+8] = colors[i*3+2];
      } else {
        arr[i*10+6] = defaults.color[0];
        arr[i*10+7] = defaults.color[1];
        arr[i*10+8] = defaults.color[2];
      }

      arr[i*10+9] = alphas ? alphas[i] : defaults.alpha;
    }

    applyDecorations(elem.decorations, count, (idx, dec) => {
      if(dec.color) {
        arr[idx*10+6] = dec.color[0];
        arr[idx*10+7] = dec.color[1];
        arr[idx*10+8] = dec.color[2];
      }
      if(dec.alpha !== undefined) {
        arr[idx*10+9] = dec.alpha;
      }
      if(dec.scale !== undefined) {
        arr[idx*10+3] *= dec.scale;
        arr[idx*10+4] *= dec.scale;
        arr[idx*10+5] *= dec.scale;
      }
    });

    return arr;
  },

  buildPickingData(elem, baseID) {
    const count = elem.centers.length / 3;
    if(count === 0) return null;

    const defaultRadius = elem.radius ?? [1, 1, 1];
    const arr = new Float32Array(count * 7);

    // Check if we have valid radii array once before the loop
    const hasValidRadii = elem.radii && elem.radii.length >= count * 3;
    const radii = hasValidRadii ? elem.radii : null;

    for(let i = 0; i < count; i++) {
      arr[i*7+0] = elem.centers[i*3+0];
      arr[i*7+1] = elem.centers[i*3+1];
      arr[i*7+2] = elem.centers[i*3+2];

      if(radii) {
        arr[i*7+3] = radii[i*3+0];
        arr[i*7+4] = radii[i*3+1];
        arr[i*7+5] = radii[i*3+2];
      } else {
        arr[i*7+3] = defaultRadius[0];
        arr[i*7+4] = defaultRadius[1];
        arr[i*7+5] = defaultRadius[2];
      }
      arr[i*7+6] = baseID + i;
    }
    return arr;
  },

  renderConfig: {
    cullMode: 'back',
    topology: 'triangle-list'
  },

  getRenderPipeline(device, bindGroupLayout, cache) {
    const format = navigator.gpu.getPreferredCanvasFormat();
    return getOrCreatePipeline(
      device,
      "EllipsoidShading",
      () => createTranslucentGeometryPipeline(device, bindGroupLayout, {
        vertexShader: ellipsoidVertCode,
        fragmentShader: ellipsoidFragCode,
        vertexEntryPoint: 'vs_main',
        fragmentEntryPoint: 'fs_main',
        bufferLayouts: [MESH_GEOMETRY_LAYOUT, ELLIPSOID_INSTANCE_LAYOUT]
      }, format, ellipsoidSpec),
      cache
    );
  },

  getPickingPipeline(device, bindGroupLayout, cache) {
    return getOrCreatePipeline(
      device,
      "EllipsoidPicking",
      () => createRenderPipeline(device, bindGroupLayout, {
        vertexShader: pickingVertCode,
        fragmentShader: pickingVertCode,
        vertexEntryPoint: 'vs_ellipsoid',
        fragmentEntryPoint: 'fs_pick',
        bufferLayouts: [MESH_GEOMETRY_LAYOUT, MESH_PICKING_INSTANCE_LAYOUT]
      }, 'rgba8unorm'),
      cache
    );
  },

  createGeometryResource(device) {
    return createBuffers(device, createSphereGeometry(32, 48));
  }
};

/** ===================== ELLIPSOID AXES ===================== **/


const ringVertCode = /*wgsl*/`
${cameraStruct}

struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) color: vec3<f32>,
  @location(3) alpha: f32,
  @location(4) worldPos: vec3<f32>,
};

@vertex
fn vs_main(
  @builtin(instance_index) instID: u32,
  @location(0) inPos: vec3<f32>,
  @location(1) inNorm: vec3<f32>,
  @location(2) center: vec3<f32>,
  @location(3) scale: vec3<f32>,
  @location(4) color: vec3<f32>,
  @location(5) alpha: f32
)-> VSOut {
  let ringIndex = i32(instID % 3u);
  var lp = inPos;
  // rotate the ring geometry differently for x-y-z rings
  if(ringIndex==0){
    let tmp = lp.z;
    lp.z = -lp.y;
    lp.y = tmp;
  } else if(ringIndex==1){
    let px = lp.x;
    lp.x = -lp.y;
    lp.y = px;
    let pz = lp.z;
    lp.z = lp.x;
    lp.x = pz;
  }
  lp *= scale;
  let wp = center + lp;
  var out: VSOut;
  out.pos = camera.mvp * vec4<f32>(wp,1.0);
  out.normal = inNorm;
  out.color = color;
  out.alpha = alpha;
  out.worldPos = wp;
  return out;
}`;


const ringPickingVertCode = /*wgsl*/`
@vertex
fn vs_rings(
  @builtin(instance_index) instID:u32,
  @location(0) inPos: vec3<f32>,
  @location(1) inNorm: vec3<f32>,
  @location(2) center: vec3<f32>,
  @location(3) scale: vec3<f32>,
  @location(4) pickID: f32
)-> VSOut {
  let ringIndex=i32(instID%3u);
  var lp=inPos;
  if(ringIndex==0){
    let tmp=lp.z; lp.z=-lp.y; lp.y=tmp;
  } else if(ringIndex==1){
    let px=lp.x; lp.x=-lp.y; lp.y=px;
    let pz=lp.z; lp.z=lp.x; lp.x=pz;
  }
  lp*=scale;
  let wp=center+lp;
  var out:VSOut;
  out.pos=camera.mvp*vec4<f32>(wp,1.0);
  out.pickID=pickID;
  return out;
}`;

const ringFragCode = /*wgsl*/`
@fragment
fn fs_main(
  @location(1) n: vec3<f32>,
  @location(2) c: vec3<f32>,
  @location(3) a: f32,
  @location(4) wp: vec3<f32>
)-> @location(0) vec4<f32> {
  // simple color (no shading)
  return vec4<f32>(c, a);
}`;

export interface EllipsoidAxesComponentConfig extends BaseComponentConfig {
  type: 'EllipsoidAxes';
  centers: Float32Array;
  radii?: Float32Array;
  radius?: [number, number, number];  // Make optional since we have BaseComponentConfig defaults
  colors?: Float32Array;
}

const ellipsoidAxesSpec: PrimitiveSpec<EllipsoidAxesComponentConfig> = {
  getCount(elem) {
    // Each ellipsoid has 3 rings
    return (elem.centers.length / 3) * 3;
  },

  buildRenderData(elem) {
    const count = elem.centers.length / 3;
    if(count === 0) return null;


    const defaults = getBaseDefaults(elem);
    const { colors, alphas, scales } = getColumnarParams(elem, count);

    const defaultRadius = elem.radius ?? [1, 1, 1];
    const radii = elem.radii;
    const ringCount = count * 3;

    const arr = new Float32Array(ringCount * 10);
    for(let i = 0; i < count; i++) {
      const cx = elem.centers[i*3+0];
      const cy = elem.centers[i*3+1];
      const cz = elem.centers[i*3+2];
      // Get radii with scale
      const scale = scales ? scales[i] : defaults.scale;

      let rx: number, ry: number, rz: number;
      if (radii) {
        rx = radii[i*3+0];
        ry = radii[i*3+1];
        rz = radii[i*3+2];
      } else {
        rx = defaultRadius[0];
        ry = defaultRadius[1];
        rz = defaultRadius[2];
      }
      rx *= scale;
      ry *= scale;
      rz *= scale;

      // Get colors
      let cr: number, cg: number, cb: number;
      if (colors) {
        cr = colors[i*3+0];
        cg = colors[i*3+1];
        cb = colors[i*3+2];
      } else {
        cr = defaults.color[0];
        cg = defaults.color[1];
        cb = defaults.color[2];
      }
      let alpha = alphas ? alphas[i] : defaults.alpha;

      // Fill 3 rings
      for(let ring = 0; ring < 3; ring++) {
        const idx = i*3 + ring;
        arr[idx*10+0] = cx;
        arr[idx*10+1] = cy;
        arr[idx*10+2] = cz;
        arr[idx*10+3] = rx;
        arr[idx*10+4] = ry;
        arr[idx*10+5] = rz;
        arr[idx*10+6] = cr;
        arr[idx*10+7] = cg;
        arr[idx*10+8] = cb;
        arr[idx*10+9] = alpha;
      }
    }

    // Apply decorations after the main loop, accounting for ring structure
    applyDecorations(elem.decorations, count, (idx, dec) => {
      // For each decorated ellipsoid, update all 3 of its rings
      for(let ring = 0; ring < 3; ring++) {
        const arrIdx = idx*3 + ring;
        if(dec.color) {
          arr[arrIdx*10+6] = dec.color[0];
          arr[arrIdx*10+7] = dec.color[1];
          arr[arrIdx*10+8] = dec.color[2];
        }
        if(dec.alpha !== undefined) {
          arr[arrIdx*10+9] = dec.alpha;
        }
        if(dec.scale !== undefined) {
          arr[arrIdx*10+3] *= dec.scale;
          arr[arrIdx*10+4] *= dec.scale;
          arr[arrIdx*10+5] *= dec.scale;
        }
      }
    });
    return arr;
  },

  buildPickingData(elem, baseID) {
    const count = elem.centers.length / 3;
    if(count === 0) return null;

    const defaultRadius = elem.radius ?? [1, 1, 1];
    const ringCount = count * 3;
    const arr = new Float32Array(ringCount * 7);

    for(let i = 0; i < count; i++) {
      const cx = elem.centers[i*3+0];
      const cy = elem.centers[i*3+1];
      const cz = elem.centers[i*3+2];
      const rx = elem.radii?.[i*3+0] ?? defaultRadius[0];
      const ry = elem.radii?.[i*3+1] ?? defaultRadius[1];
      const rz = elem.radii?.[i*3+2] ?? defaultRadius[2];
      const thisID = baseID + i;

      for(let ring = 0; ring < 3; ring++) {
        const idx = i*3 + ring;
        arr[idx*7+0] = cx;
        arr[idx*7+1] = cy;
        arr[idx*7+2] = cz;
        arr[idx*7+3] = rx;
        arr[idx*7+4] = ry;
        arr[idx*7+5] = rz;
        arr[idx*7+6] = thisID;
      }
    }
    return arr;
  },

  renderConfig: {
    cullMode: 'back',
    topology: 'triangle-list'
  },

  getRenderPipeline(device, bindGroupLayout, cache) {
    const format = navigator.gpu.getPreferredCanvasFormat();
    return getOrCreatePipeline(
      device,
      "EllipsoidAxesShading",
      () => createTranslucentGeometryPipeline(device, bindGroupLayout, {
        vertexShader: ringVertCode,
        fragmentShader: ringFragCode,
        vertexEntryPoint: 'vs_main',
        fragmentEntryPoint: 'fs_main',
        bufferLayouts: [MESH_GEOMETRY_LAYOUT, ELLIPSOID_INSTANCE_LAYOUT],
        blend: {} // Use defaults
      }, format, ellipsoidAxesSpec),
      cache
    );
  },

  getPickingPipeline(device, bindGroupLayout, cache) {
    return getOrCreatePipeline(
      device,
      "EllipsoidAxesPicking",
      () => createRenderPipeline(device, bindGroupLayout, {
        vertexShader: pickingVertCode,
        fragmentShader: pickingVertCode,
        vertexEntryPoint: 'vs_rings',
        fragmentEntryPoint: 'fs_pick',
        bufferLayouts: [MESH_GEOMETRY_LAYOUT, MESH_PICKING_INSTANCE_LAYOUT]
      }, 'rgba8unorm'),
      cache
    );
  },

  createGeometryResource(device) {
    return createBuffers(device, createTorusGeometry(1.0, 0.03, 40, 12));
  }
};

/** ===================== CUBOID ===================== **/


const cuboidVertCode = /*wgsl*/`
${cameraStruct}

struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) baseColor: vec3<f32>,
  @location(3) alpha: f32,
  @location(4) worldPos: vec3<f32>
};

@vertex
fn vs_main(
  @location(0) inPos: vec3<f32>,
  @location(1) inNorm: vec3<f32>,
  @location(2) center: vec3<f32>,
  @location(3) size: vec3<f32>,
  @location(4) color: vec3<f32>,
  @location(5) alpha: f32
)-> VSOut {
  let worldPos = center + (inPos * size);
  let scaledNorm = normalize(inNorm / size);
  var out: VSOut;
  out.pos = camera.mvp * vec4<f32>(worldPos,1.0);
  out.normal = scaledNorm;
  out.baseColor = color;
  out.alpha = alpha;
  out.worldPos = worldPos;
  return out;
}`;

const cuboidPickingVertCode = /*wgsl*/`
@vertex
fn vs_cuboid(
  @location(0) inPos: vec3<f32>,
  @location(1) inNorm: vec3<f32>,
  @location(2) center: vec3<f32>,
  @location(3) size: vec3<f32>,
  @location(4) pickID: f32
)-> VSOut {
  let wp = center + (inPos * size);
  var out: VSOut;
  out.pos = camera.mvp*vec4<f32>(wp,1.0);
  out.pickID = pickID;
  return out;
}`;

const cuboidFragCode = /*wgsl*/`
${cameraStruct}
${lightingConstants}
${lightingCalc}

@fragment
fn fs_main(
  @location(1) normal: vec3<f32>,
  @location(2) baseColor: vec3<f32>,
  @location(3) alpha: f32,
  @location(4) worldPos: vec3<f32>
)-> @location(0) vec4<f32> {
  let color = calculateLighting(baseColor, normal, worldPos);
  return vec4<f32>(color, alpha);
}`;

export interface CuboidComponentConfig extends BaseComponentConfig {
  type: 'Cuboid';
  centers: Float32Array;
  sizes: Float32Array;
  size?: [number, number, number];
}

const cuboidSpec: PrimitiveSpec<CuboidComponentConfig> = {
  getCount(elem){
    return elem.centers.length / 3;
  },
  buildRenderData(elem) {
    const count = elem.centers.length / 3;
    if(count === 0) return null;

    const defaults = getBaseDefaults(elem);
    const { colors, alphas, scales } = getColumnarParams(elem, count);

    const defaultSize = elem.size || [0.1, 0.1, 0.1];
    const sizes = elem.sizes && elem.sizes.length >= count * 3 ? elem.sizes : null;

    const arr = new Float32Array(count * 10);
    for(let i = 0; i < count; i++) {
      const cx = elem.centers[i*3+0];
      const cy = elem.centers[i*3+1];
      const cz = elem.centers[i*3+2];
      const scale = scales ? scales[i] : defaults.scale;

      // Get sizes with scale
      const sx = (sizes ? sizes[i*3+0] : defaultSize[0]) * scale;
      const sy = (sizes ? sizes[i*3+1] : defaultSize[1]) * scale;
      const sz = (sizes ? sizes[i*3+2] : defaultSize[2]) * scale;

      // Get colors
      let cr: number, cg: number, cb: number;
      if (colors) {
        cr = colors[i*3+0];
        cg = colors[i*3+1];
        cb = colors[i*3+2];
      } else {
        cr = defaults.color[0];
        cg = defaults.color[1];
        cb = defaults.color[2];
      }
      const alpha = alphas ? alphas[i] : defaults.alpha;

      // Fill array
      const idx = i * 10;
      arr[idx+0] = cx;
      arr[idx+1] = cy;
      arr[idx+2] = cz;
      arr[idx+3] = sx;
      arr[idx+4] = sy;
      arr[idx+5] = sz;
      arr[idx+6] = cr;
      arr[idx+7] = cg;
      arr[idx+8] = cb;
      arr[idx+9] = alpha;
    }

    applyDecorations(elem.decorations, count, (idx, dec) => {
      if(dec.color) {
        arr[idx*10+6] = dec.color[0];
        arr[idx*10+7] = dec.color[1];
        arr[idx*10+8] = dec.color[2];
      }
      if(dec.alpha !== undefined) {
        arr[idx*10+9] = dec.alpha;
      }
      if(dec.scale !== undefined) {
        arr[idx*10+3] *= dec.scale;
        arr[idx*10+4] *= dec.scale;
        arr[idx*10+5] *= dec.scale;
      }
    });

    return arr;
  },
  buildPickingData(elem, baseID){
    const count = elem.centers.length / 3;
    if(count === 0) return null;

    const defaultSize = elem.size || [0.1, 0.1, 0.1];
    const sizes = elem.sizes && elem.sizes.length >= count * 3 ? elem.sizes : null;
    const { scales } = getColumnarParams(elem, count);

    const arr = new Float32Array(count * 7);
    for(let i = 0; i < count; i++) {
      const scale = scales ? scales[i] : 1;
      arr[i*7+0] = elem.centers[i*3+0];
      arr[i*7+1] = elem.centers[i*3+1];
      arr[i*7+2] = elem.centers[i*3+2];
      arr[i*7+3] = (sizes ? sizes[i*3+0] : defaultSize[0]) * scale;
      arr[i*7+4] = (sizes ? sizes[i*3+1] : defaultSize[1]) * scale;
      arr[i*7+5] = (sizes ? sizes[i*3+2] : defaultSize[2]) * scale;
      arr[i*7+6] = baseID + i;
    }

    // Apply scale decorations
    applyDecorations(elem.decorations, count, (idx, dec) => {
      if(dec.scale !== undefined) {
        arr[idx*7+3] *= dec.scale;
        arr[idx*7+4] *= dec.scale;
        arr[idx*7+5] *= dec.scale;
      }
    });

    return arr;
  },
  renderConfig: {
    cullMode: 'none',  // Cuboids need to be visible from both sides
    topology: 'triangle-list'
  },
  getRenderPipeline(device, bindGroupLayout, cache) {
    const format = navigator.gpu.getPreferredCanvasFormat();
    return getOrCreatePipeline(
      device,
      "CuboidShading",
      () => createTranslucentGeometryPipeline(device, bindGroupLayout, {
        vertexShader: cuboidVertCode,
        fragmentShader: cuboidFragCode,
        vertexEntryPoint: 'vs_main',
        fragmentEntryPoint: 'fs_main',
        bufferLayouts: [MESH_GEOMETRY_LAYOUT, ELLIPSOID_INSTANCE_LAYOUT]
      }, format, cuboidSpec),
      cache
    );
  },

  getPickingPipeline(device, bindGroupLayout, cache) {
    return getOrCreatePipeline(
      device,
      "CuboidPicking",
      () => createRenderPipeline(device, bindGroupLayout, {
        vertexShader: pickingVertCode,
        fragmentShader: pickingVertCode,
        vertexEntryPoint: 'vs_cuboid',
        fragmentEntryPoint: 'fs_pick',
        bufferLayouts: [MESH_GEOMETRY_LAYOUT, MESH_PICKING_INSTANCE_LAYOUT],
        primitive: this.renderConfig
      }, 'rgba8unorm'),
      cache
    );
  },

  createGeometryResource(device) {
    return createBuffers(device, createCubeGeometry());
  }
};

/******************************************************
 *  LineBeams Type
 ******************************************************/


const lineBeamVertCode = /*wgsl*/`// lineBeamVertCode.wgsl
${cameraStruct}
${lightingConstants}

struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) baseColor: vec3<f32>,
  @location(3) alpha: f32,
  @location(4) worldPos: vec3<f32>,
};

@vertex
fn vs_main(
  @location(0) inPos: vec3<f32>,
  @location(1) inNorm: vec3<f32>,

  @location(2) startPos: vec3<f32>,
  @location(3) endPos: vec3<f32>,
  @location(4) size: f32,
  @location(5) color: vec3<f32>,
  @location(6) alpha: f32
) -> VSOut
{
  // The unit beam is from z=0..1 along local Z, size=1 in XY
  // We'll transform so it goes from start->end with size=size.
  let segDir = endPos - startPos;
  let length = max(length(segDir), 0.000001);
  let zDir   = normalize(segDir);

  // build basis xDir,yDir from zDir
  var tempUp = vec3<f32>(0,0,1);
  if (abs(dot(zDir, tempUp)) > 0.99) {
    tempUp = vec3<f32>(0,1,0);
  }
  let xDir = normalize(cross(zDir, tempUp));
  let yDir = cross(zDir, xDir);

  // For cuboid, we want corners at ±size in both x and y
  let localX = inPos.x * size;
  let localY = inPos.y * size;
  let localZ = inPos.z * length;
  let worldPos = startPos
    + xDir * localX
    + yDir * localY
    + zDir * localZ;

  // transform normal similarly
  let rawNormal = vec3<f32>(inNorm.x, inNorm.y, inNorm.z);
  let nWorld = normalize(
    xDir*rawNormal.x +
    yDir*rawNormal.y +
    zDir*rawNormal.z
  );

  var out: VSOut;
  out.pos = camera.mvp * vec4<f32>(worldPos, 1.0);
  out.normal = nWorld;
  out.baseColor = color;
  out.alpha = alpha;
  out.worldPos = worldPos;
  return out;
}`;

const lineBeamFragCode = /*wgsl*/`// lineBeamFragCode.wgsl
${cameraStruct}
${lightingConstants}
${lightingCalc}

@fragment
fn fs_main(
  @location(1) normal: vec3<f32>,
  @location(2) baseColor: vec3<f32>,
  @location(3) alpha: f32,
  @location(4) worldPos: vec3<f32>
)-> @location(0) vec4<f32>
{
  let color = calculateLighting(baseColor, normal, worldPos);
  return vec4<f32>(color, alpha);
}`

const lineBeamPickingVertCode = /*wgsl*/`
@vertex
fn vs_lineBeam(  // Rename from vs_lineCyl to vs_lineBeam
  @location(0) inPos: vec3<f32>,
  @location(1) inNorm: vec3<f32>,

  @location(2) startPos: vec3<f32>,
  @location(3) endPos: vec3<f32>,
  @location(4) size: f32,
  @location(5) pickID: f32
) -> VSOut {
  let segDir = endPos - startPos;
  let length = max(length(segDir), 0.000001);
  let zDir = normalize(segDir);

  var tempUp = vec3<f32>(0,0,1);
  if (abs(dot(zDir, tempUp)) > 0.99) {
    tempUp = vec3<f32>(0,1,0);
  }
  let xDir = normalize(cross(zDir, tempUp));
  let yDir = cross(zDir, xDir);

  let localX = inPos.x * size;
  let localY = inPos.y * size;
  let localZ = inPos.z * length;
  let worldPos = startPos
    + xDir*localX
    + yDir*localY
    + zDir*localZ;

  var out: VSOut;
  out.pos = camera.mvp * vec4<f32>(worldPos, 1.0);
  out.pickID = pickID;
  return out;
}`;

export interface LineBeamsComponentConfig extends BaseComponentConfig {
  type: 'LineBeams';
  positions: Float32Array;  // [x,y,z,i, x,y,z,i, ...]
  sizes?: Float32Array;     // Per-line sizes
  size?: number;         // Default size, defaults to 0.02
}

function countSegments(positions: Float32Array): number {
  const pointCount = positions.length / 4;
  if (pointCount < 2) return 0;

  let segCount = 0;
  for (let p = 0; p < pointCount - 1; p++) {
    const iCurr = positions[p * 4 + 3];
    const iNext = positions[(p+1) * 4 + 3];
    if (iCurr === iNext) {
      segCount++;
    }
  }
  return segCount;
}

const lineBeamsSpec: PrimitiveSpec<LineBeamsComponentConfig> = {
  getCount(elem) {
    return countSegments(elem.positions);
  },

  buildRenderData(elem) {
    const segCount = this.getCount(elem);
    if(segCount === 0) return null;

    const defaults = getBaseDefaults(elem);
    const { colors, alphas, scales } = getColumnarParams(elem, segCount);

    const defaultSize = elem.size ?? 0.02;
    const sizes = elem.sizes instanceof Float32Array && elem.sizes.length >= segCount ? elem.sizes : null;

    const arr = new Float32Array(segCount * 11);
    let segIndex = 0;

    const pointCount = elem.positions.length / 4;
    for(let p = 0; p < pointCount - 1; p++) {
      const iCurr = elem.positions[p * 4 + 3];
      const iNext = elem.positions[(p+1) * 4 + 3];
      if(iCurr !== iNext) continue;

      const lineIndex = Math.floor(iCurr);

      // Start point
      arr[segIndex*11+0] = elem.positions[p * 4 + 0];
      arr[segIndex*11+1] = elem.positions[p * 4 + 1];
      arr[segIndex*11+2] = elem.positions[p * 4 + 2];

      // End point
      arr[segIndex*11+3] = elem.positions[(p+1) * 4 + 0];
      arr[segIndex*11+4] = elem.positions[(p+1) * 4 + 1];
      arr[segIndex*11+5] = elem.positions[(p+1) * 4 + 2];

      // Size with scale
      const scale = scales ? scales[lineIndex] : defaults.scale;
      arr[segIndex*11+6] = (sizes ? sizes[lineIndex] : defaultSize) * scale;

      // Colors
      if(colors) {
        arr[segIndex*11+7] = colors[lineIndex*3+0];
        arr[segIndex*11+8] = colors[lineIndex*3+1];
        arr[segIndex*11+9] = colors[lineIndex*3+2];
      } else {
        arr[segIndex*11+7] = defaults.color[0];
        arr[segIndex*11+8] = defaults.color[1];
        arr[segIndex*11+9] = defaults.color[2];
      }

      arr[segIndex*11+10] = alphas ? alphas[lineIndex] : defaults.alpha;

      segIndex++;
    }

    // Apply decorations last
    applyDecorations(elem.decorations, segCount, (idx, dec) => {
      if(dec.color) {
        arr[idx*11+7] = dec.color[0];
        arr[idx*11+8] = dec.color[1];
        arr[idx*11+9] = dec.color[2];
      }
      if(dec.alpha !== undefined) {
        arr[idx*11+10] = dec.alpha;
      }
      if(dec.scale !== undefined) {
        arr[idx*11+6] *= dec.scale;
      }
    });

    return arr;
  },

  buildPickingData(elem, baseID) {
    const segCount = this.getCount(elem);
    if(segCount === 0) return null;

    const defaultSize = elem.size ?? 0.02;
    const floatsPerSeg = 8;
    const arr = new Float32Array(segCount * floatsPerSeg);

    const pointCount = elem.positions.length / 4;
    let segIndex = 0;

    for(let p = 0; p < pointCount - 1; p++) {
      const iCurr = elem.positions[p * 4 + 3];
      const iNext = elem.positions[(p+1) * 4 + 3];
      if(iCurr !== iNext) continue;

      const lineIndex = Math.floor(iCurr);
      let size = elem.sizes?.[lineIndex] ?? defaultSize;
      const scale = elem.scales?.[lineIndex] ?? 1.0;

      size *= scale;

      // Apply decorations that affect size
      applyDecorations(elem.decorations, lineIndex + 1, (idx, dec) => {
        if(idx === lineIndex && dec.scale !== undefined) {
          size *= dec.scale;
        }
      });

      const base = segIndex * floatsPerSeg;
      arr[base + 0] = elem.positions[p * 4 + 0];     // start.x
      arr[base + 1] = elem.positions[p * 4 + 1];     // start.y
      arr[base + 2] = elem.positions[p * 4 + 2];     // start.z
      arr[base + 3] = elem.positions[(p+1) * 4 + 0]; // end.x
      arr[base + 4] = elem.positions[(p+1) * 4 + 1]; // end.y
      arr[base + 5] = elem.positions[(p+1) * 4 + 2]; // end.z
      arr[base + 6] = size;                        // size
      arr[base + 7] = baseID + segIndex;            // pickID

      segIndex++;
    }
    return arr;
  },

  // Standard triangle-list, cull as you like
  renderConfig: {
    cullMode: 'none',
    topology: 'triangle-list'
  },

  getRenderPipeline(device, bindGroupLayout, cache) {
    const format = navigator.gpu.getPreferredCanvasFormat();
    return getOrCreatePipeline(
      device,
      "LineBeamsShading",
      () => createTranslucentGeometryPipeline(device, bindGroupLayout, {
        vertexShader: lineBeamVertCode,   // defined below
        fragmentShader: lineBeamFragCode, // defined below
        vertexEntryPoint: 'vs_main',
        fragmentEntryPoint: 'fs_main',
        bufferLayouts: [ MESH_GEOMETRY_LAYOUT, LINE_BEAM_INSTANCE_LAYOUT ],
      }, format, this),
      cache
    );
  },

  getPickingPipeline(device, bindGroupLayout, cache) {
    return getOrCreatePipeline(
      device,
      "LineBeamsPicking",
      () => createRenderPipeline(device, bindGroupLayout, {
        vertexShader: pickingVertCode, // We'll add a vs_lineCyl entry
        fragmentShader: pickingVertCode,
        vertexEntryPoint: 'vs_lineBeam',
        fragmentEntryPoint: 'fs_pick',
        bufferLayouts: [ MESH_GEOMETRY_LAYOUT, LINE_BEAM_PICKING_INSTANCE_LAYOUT ],
        primitive: this.renderConfig
      }, 'rgba8unorm'),
      cache
    );
  },

  createGeometryResource(device) {
    return createBuffers(device, createBeamGeometry());
  }
};


const pickingVertCode = /*wgsl*/`
${cameraStruct}

struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) pickID: f32
};

@fragment
fn fs_pick(@location(0) pickID: f32)-> @location(0) vec4<f32> {
  let iID = u32(pickID);
  let r = f32(iID & 255u)/255.0;
  let g = f32((iID>>8)&255u)/255.0;
  let b = f32((iID>>16)&255u)/255.0;
  return vec4<f32>(r,g,b,1.0);
}

${billboardPickingVertCode}
${ellipsoidPickingVertCode}
${ringPickingVertCode}
${cuboidPickingVertCode}
${lineBeamPickingVertCode}
`;


/******************************************************
 * 4) Pipeline Cache Helper
 ******************************************************/
// Update the pipeline cache to include device reference
export interface PipelineCacheEntry {
  pipeline: GPURenderPipeline;
  device: GPUDevice;
}

function getOrCreatePipeline(
  device: GPUDevice,
  key: string,
  createFn: () => GPURenderPipeline,
  cache: Map<string, PipelineCacheEntry>  // This will be the instance cache
): GPURenderPipeline {
  const entry = cache.get(key);
  if (entry && entry.device === device) {
    return entry.pipeline;
  }

  // Create new pipeline and cache it with device reference
  const pipeline = createFn();
  cache.set(key, { pipeline, device });
  return pipeline;
}

/******************************************************
 * 5) Common Resources: Geometry, Layout, etc.
 ******************************************************/
export interface GeometryResource {
  vb: GPUBuffer;
  ib: GPUBuffer;
  indexCount?: number;
}

export type GeometryResources = {
  [K in keyof typeof primitiveRegistry]: GeometryResource | null;
}

function getGeometryResource(resources: GeometryResources, type: keyof GeometryResources): GeometryResource {
  const resource = resources[type];
  if (!resource) {
    throw new Error(`No geometry resource found for type ${type}`);
  }
  return resource;
}


const createBuffers = (device: GPUDevice, { vertexData, indexData }: { vertexData: Float32Array, indexData: Uint16Array | Uint32Array }) => {
  const vb = device.createBuffer({
    size: vertexData.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
  });
  device.queue.writeBuffer(vb, 0, vertexData);

  const ib = device.createBuffer({
    size: indexData.byteLength,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST
  });
  device.queue.writeBuffer(ib, 0, indexData);

  return { vb, ib, indexCount: indexData.length };
};

function initGeometryResources(device: GPUDevice, resources: GeometryResources) {
  // Create geometry for each primitive type
  for (const [primitiveName, spec] of Object.entries(primitiveRegistry)) {
    const typedName = primitiveName as keyof GeometryResources;
    if (!resources[typedName]) {
      resources[typedName] = spec.createGeometryResource(device);
    }
  }
}

/******************************************************
 * 6) Pipeline Configuration Helpers
 ******************************************************/
interface VertexBufferLayout {
  arrayStride: number;
  stepMode?: GPUVertexStepMode;
  attributes: {
    shaderLocation: number;
    offset: number;
    format: GPUVertexFormat;
  }[];
}

interface PipelineConfig {
  vertexShader: string;
  fragmentShader: string;
  vertexEntryPoint: string;
  fragmentEntryPoint: string;
  bufferLayouts: VertexBufferLayout[];
  primitive?: {
    topology?: GPUPrimitiveTopology;
    cullMode?: GPUCullMode;
  };
  blend?: {
    color?: GPUBlendComponent;
    alpha?: GPUBlendComponent;
  };
  depthStencil?: {
    format: GPUTextureFormat;
    depthWriteEnabled: boolean;
    depthCompare: GPUCompareFunction;
  };
  colorWriteMask?: GPUColorWriteFlags;  // Add this to control color writes
}

function createRenderPipeline(
  device: GPUDevice,
  bindGroupLayout: GPUBindGroupLayout,
  config: PipelineConfig,
  format: GPUTextureFormat
): GPURenderPipeline {
  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout]
  });

  // Get primitive configuration with defaults
  const primitiveConfig = {
    topology: config.primitive?.topology || 'triangle-list',
    cullMode: config.primitive?.cullMode || 'back'
  };

  return device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: {
      module: device.createShaderModule({ code: config.vertexShader }),
      entryPoint: config.vertexEntryPoint,
      buffers: config.bufferLayouts
    },
    fragment: {
      module: device.createShaderModule({ code: config.fragmentShader }),
      entryPoint: config.fragmentEntryPoint,
      targets: [{
        format,
        writeMask: config.colorWriteMask ?? GPUColorWrite.ALL,
        ...(config.blend && {
          blend: {
            color: config.blend.color || {
              srcFactor: 'src-alpha',
              dstFactor: 'one-minus-src-alpha'
            },
            alpha: config.blend.alpha || {
              srcFactor: 'one',
              dstFactor: 'one-minus-src-alpha'
            }
          }
        })
      }]
    },
    primitive: primitiveConfig,
    depthStencil: config.depthStencil || {
      format: 'depth24plus',
      depthWriteEnabled: true,
      depthCompare: 'less'
    }
  });
}

function createTranslucentGeometryPipeline(
  device: GPUDevice,
  bindGroupLayout: GPUBindGroupLayout,
  config: PipelineConfig,
  format: GPUTextureFormat,
  primitiveSpec: PrimitiveSpec<any>  // Take the primitive spec instead of just type
): GPURenderPipeline {
  return createRenderPipeline(device, bindGroupLayout, {
    ...config,
    primitive: primitiveSpec.renderConfig,
    blend: {
      color: {
        srcFactor: 'src-alpha',
        dstFactor: 'one-minus-src-alpha',
        operation: 'add'
      },
      alpha: {
        srcFactor: 'one',
        dstFactor: 'one-minus-src-alpha',
        operation: 'add'
      }
    },
    depthStencil: {
      format: 'depth24plus',
      depthWriteEnabled: true,
      depthCompare: 'less'
    }
  }, format);
}


// Common vertex buffer layouts
const POINT_CLOUD_GEOMETRY_LAYOUT: VertexBufferLayout = {
  arrayStride: 24,  // 6 floats * 4 bytes
  attributes: [
    {  // position xyz
      shaderLocation: 0,
      offset: 0,
      format: 'float32x3'
    },
    {  // normal xyz
      shaderLocation: 1,
      offset: 12,
      format: 'float32x3'
    }
  ]
};

const POINT_CLOUD_INSTANCE_LAYOUT: VertexBufferLayout = {
  arrayStride: 32,  // 8 floats * 4 bytes
  stepMode: 'instance',
  attributes: [
    {shaderLocation: 2, offset: 0,  format: 'float32x3'},  // instancePos
    {shaderLocation: 3, offset: 12, format: 'float32x3'},  // color
    {shaderLocation: 4, offset: 24, format: 'float32'},    // alpha
    {shaderLocation: 5, offset: 28, format: 'float32'}     // size
  ]
};

const POINT_CLOUD_PICKING_INSTANCE_LAYOUT: VertexBufferLayout = {
  arrayStride: 20,  // 5 floats * 4 bytes
  stepMode: 'instance',
  attributes: [
    {shaderLocation: 2, offset: 0,   format: 'float32x3'},  // instancePos
    {shaderLocation: 3, offset: 12,  format: 'float32'},    // pickID
    {shaderLocation: 4, offset: 16,  format: 'float32'}     // size
  ]
};

const MESH_GEOMETRY_LAYOUT: VertexBufferLayout = {
  arrayStride: 6*4,
  attributes: [
    {shaderLocation: 0, offset: 0,   format: 'float32x3'},
    {shaderLocation: 1, offset: 3*4, format: 'float32x3'}
  ]
};

const ELLIPSOID_INSTANCE_LAYOUT: VertexBufferLayout = {
  arrayStride: 10*4,
  stepMode: 'instance',
  attributes: [
    {shaderLocation: 2, offset: 0,     format: 'float32x3'},
    {shaderLocation: 3, offset: 3*4,   format: 'float32x3'},
    {shaderLocation: 4, offset: 6*4,   format: 'float32x3'},
    {shaderLocation: 5, offset: 9*4,   format: 'float32'}
  ]
};

const MESH_PICKING_INSTANCE_LAYOUT: VertexBufferLayout = {
  arrayStride: 7*4,
  stepMode: 'instance',
  attributes: [
    {shaderLocation: 2, offset: 0,   format: 'float32x3'},
    {shaderLocation: 3, offset: 3*4, format: 'float32x3'},
    {shaderLocation: 4, offset: 6*4, format: 'float32'}
  ]
};

const CYL_GEOMETRY_LAYOUT: VertexBufferLayout = {
  arrayStride: 6 * 4, // (pos.x, pos.y, pos.z, norm.x, norm.y, norm.z)
  attributes: [
    { shaderLocation: 0, offset: 0,  format: 'float32x3' }, // position
    { shaderLocation: 1, offset: 12, format: 'float32x3' } // normal
  ]
};

// For rendering: 11 floats
// (start.xyz, end.xyz, size, color.rgb, alpha)
const LINE_BEAM_INSTANCE_LAYOUT: VertexBufferLayout = {
  arrayStride: 11 * 4,
  stepMode: 'instance',
  attributes: [
    { shaderLocation: 2, offset:  0,  format: 'float32x3' }, // startPos
    { shaderLocation: 3, offset: 12,  format: 'float32x3' }, // endPos
    { shaderLocation: 4, offset: 24,  format: 'float32'   }, // size
    { shaderLocation: 5, offset: 28,  format: 'float32x3' }, // color
    { shaderLocation: 6, offset: 40,  format: 'float32'   }, // alpha
  ]
};

// For picking: 8 floats
// (start.xyz, end.xyz, size, pickID)
const LINE_BEAM_PICKING_INSTANCE_LAYOUT: VertexBufferLayout = {
  arrayStride: 8 * 4,
  stepMode: 'instance',
  attributes: [
    { shaderLocation: 2, offset:  0,  format: 'float32x3' },
    { shaderLocation: 3, offset: 12,  format: 'float32x3' },
    { shaderLocation: 4, offset: 24,  format: 'float32'   }, // size
    { shaderLocation: 5, offset: 28,  format: 'float32'   },
  ]
};

/******************************************************
 * 7) Primitive Registry
 ******************************************************/
export type ComponentConfig =
  | PointCloudComponentConfig
  | EllipsoidComponentConfig
  | EllipsoidAxesComponentConfig
  | CuboidComponentConfig
  | LineBeamsComponentConfig;

const primitiveRegistry: Record<ComponentConfig['type'], PrimitiveSpec<any>> = {
  PointCloud: pointCloudSpec,  // Use consolidated spec
  Ellipsoid: ellipsoidSpec,
  EllipsoidAxes: ellipsoidAxesSpec,
  Cuboid: cuboidSpec,
  LineBeams: lineBeamsSpec
};


/******************************************************
 * 8) Scene
 ******************************************************/

export function SceneInner({
  components,
  containerWidth,
  containerHeight,
  style,
  camera: controlledCamera,
  defaultCamera,
  onCameraChange,
  onFrameRendered
}: SceneInnerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // We'll store references to the GPU + other stuff in a ref object
  const gpuRef = useRef<{
    device: GPUDevice;
    context: GPUCanvasContext;
    uniformBuffer: GPUBuffer;
    uniformBindGroup: GPUBindGroup;
    bindGroupLayout: GPUBindGroupLayout;
    depthTexture: GPUTexture | null;
    pickTexture: GPUTexture | null;
    pickDepthTexture: GPUTexture | null;
    readbackBuffer: GPUBuffer;

    renderObjects: RenderObject[];
    componentBaseId: number[];
    idToComponent: ({componentIdx: number, instanceIdx: number} | null)[];
    pipelineCache: Map<string, PipelineCacheEntry>;  // Add this
    dynamicBuffers: DynamicBuffers | null;
    resources: GeometryResources;  // Add this
  } | null>(null);

  const [isReady, setIsReady] = useState(false);

  // Helper function to safely convert to array
  function toArray(value: [number, number, number] | Float32Array | undefined): [number, number, number] {
      if (!value) return [0, 0, 0];
      return Array.from(value) as [number, number, number];
  }

  // Update the camera initialization
  const [internalCamera, setInternalCamera] = useState<CameraState>(() => {
      const initial = defaultCamera || DEFAULT_CAMERA;
      return createCameraState(initial);
  });

  // Use the appropriate camera state based on whether we're controlled or not
  const activeCamera = useMemo(() => {
      if (controlledCamera) {
          return createCameraState(controlledCamera);
      }
      return internalCamera;
  }, [controlledCamera, internalCamera]);

  // Update handleCameraUpdate to use activeCamera
  const handleCameraUpdate = useCallback((updateFn: (camera: CameraState) => CameraState) => {
    const newCameraState = updateFn(activeCamera);

    if (controlledCamera) {
        onCameraChange?.(createCameraParams(newCameraState));
    } else {
        setInternalCamera(newCameraState);
        onCameraChange?.(createCameraParams(newCameraState));
    }
}, [activeCamera, controlledCamera, onCameraChange]);

  // We'll also track a picking lock
  const pickingLockRef = useRef(false);

  // Add hover state tracking
  const lastHoverState = useRef<{componentIdx: number, instanceIdx: number} | null>(null);

  /******************************************************
   * A) initWebGPU
   ******************************************************/
  const initWebGPU = useCallback(async()=>{
    if(!canvasRef.current) return;
    if(!navigator.gpu) {
      console.error("WebGPU not supported in this browser.");
      return;
    }
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if(!adapter) throw new Error("No GPU adapter found");
      const device = await adapter.requestDevice();

      const context = canvasRef.current.getContext('webgpu') as GPUCanvasContext;
      const format = navigator.gpu.getPreferredCanvasFormat();
      context.configure({ device, format, alphaMode:'premultiplied' });

      // Create bind group layout
      const bindGroupLayout = device.createBindGroupLayout({
        entries: [{
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: {type:'uniform'}
        }]
      });

      // Create uniform buffer
      const uniformBufferSize=128;
      const uniformBuffer=device.createBuffer({
        size: uniformBufferSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
      });

      // Create bind group using the new layout
      const uniformBindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [{ binding:0, resource:{ buffer:uniformBuffer } }]
      });

      // Readback buffer for picking
      const readbackBuffer = device.createBuffer({
        size: 256,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        label: 'Picking readback buffer'
      });

      // First create gpuRef.current with empty resources
      gpuRef.current = {
        device,
        context,
        uniformBuffer,
        uniformBindGroup,
        bindGroupLayout,
        depthTexture: null,
        pickTexture: null,
        pickDepthTexture: null,
        readbackBuffer,
        renderObjects: [],
        componentBaseId: [],
        idToComponent: [null],  // First ID (0) is reserved
        pipelineCache: new Map(),
        dynamicBuffers: null,
        resources: {
          PointCloud: null,
          Ellipsoid: null,
          EllipsoidAxes: null,
          Cuboid: null,
          LineBeams: null
        }
      };

      // Now initialize geometry resources
      initGeometryResources(device, gpuRef.current.resources);

      setIsReady(true);
    } catch(err){
      console.error("initWebGPU error:", err);
    }
  },[]);

  /******************************************************
   * B) Depth & Pick textures
   ******************************************************/
  const createOrUpdateDepthTexture = useCallback(() => {
    if(!gpuRef.current || !canvasRef.current) return;
    const { device, depthTexture } = gpuRef.current;

    // Get the actual canvas size
    const canvas = canvasRef.current;
    const displayWidth = canvas.width;
    const displayHeight = canvas.height;

    if(depthTexture) depthTexture.destroy();
    const dt = device.createTexture({
        size: [displayWidth, displayHeight],
        format: 'depth24plus',
        usage: GPUTextureUsage.RENDER_ATTACHMENT
    });
    gpuRef.current.depthTexture = dt;
}, []);

  const createOrUpdatePickTextures = useCallback(() => {
    if(!gpuRef.current || !canvasRef.current) return;
    const { device, pickTexture, pickDepthTexture } = gpuRef.current;

    // Get the actual canvas size
    const canvas = canvasRef.current;
    const displayWidth = canvas.width;
    const displayHeight = canvas.height;

    if(pickTexture) pickTexture.destroy();
    if(pickDepthTexture) pickDepthTexture.destroy();

    const colorTex = device.createTexture({
        size: [displayWidth, displayHeight],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC
    });
    const depthTex = device.createTexture({
        size: [displayWidth, displayHeight],
        format: 'depth24plus',
        usage: GPUTextureUsage.RENDER_ATTACHMENT
    });
    gpuRef.current.pickTexture = colorTex;
    gpuRef.current.pickDepthTexture = depthTex;
}, []);

  /******************************************************
   * C) Building the RenderObjects (no if/else)
   ******************************************************/
  // Move ID mapping logic to a separate function
  const buildComponentIdMapping = useCallback((components: ComponentConfig[]) => {
    if (!gpuRef.current) return;

    // Reset ID mapping
    gpuRef.current.idToComponent = [null];  // First ID (0) is reserved
    let currentID = 1;

    // Build new mapping
    components.forEach((elem, componentIdx) => {
      const spec = primitiveRegistry[elem.type];
      if (!spec) {
        gpuRef.current!.componentBaseId[componentIdx] = 0;
        return;
      }

      const count = spec.getCount(elem);
      gpuRef.current!.componentBaseId[componentIdx] = currentID;

      // Expand global ID table
      for (let j = 0; j < count; j++) {
        gpuRef.current!.idToComponent[currentID + j] = {
          componentIdx: componentIdx,
          instanceIdx: j
        };
      }
      currentID += count;
    });
  }, []);

  // Fix the calculateBufferSize function
  function calculateBufferSize(components: ComponentConfig[]): { renderSize: number, pickingSize: number } {
    let renderSize = 0;
    let pickingSize = 0;

    components.forEach(elem => {
      const spec = primitiveRegistry[elem.type];
      if (!spec) return;

      const count = spec.getCount(elem);
      const renderData = spec.buildRenderData(elem);
      const pickData = spec.buildPickingData(elem, 0);

      if (renderData) {
        // Calculate stride and ensure it's aligned to 4 bytes
        const floatsPerInstance = renderData.length / count;
        const renderStride = Math.ceil(floatsPerInstance) * 4;
        // Add to total size (not max)
        const alignedSize = renderStride * count;
        renderSize += alignedSize;
      }

      if (pickData) {
        // Calculate stride and ensure it's aligned to 4 bytes
        const floatsPerInstance = pickData.length / count;
        const pickStride = Math.ceil(floatsPerInstance) * 4;
        // Add to total size (not max)
        const alignedSize = pickStride * count;
        pickingSize += alignedSize;
      }
    });

    // Add generous padding (100%) and align to 4 bytes
    renderSize = Math.ceil((renderSize * 2) / 4) * 4;
    pickingSize = Math.ceil((pickingSize * 2) / 4) * 4;

    return { renderSize, pickingSize };
  }

  // Modify createDynamicBuffers to take specific sizes
  function createDynamicBuffers(device: GPUDevice, renderSize: number, pickingSize: number) {
    const renderBuffer = device.createBuffer({
      size: renderSize,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: false
    });

    const pickingBuffer = device.createBuffer({
      size: pickingSize,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: false
    });

    return {
      renderBuffer,
      pickingBuffer,
      renderOffset: 0,
      pickingOffset: 0
    };
  }

  // Update buildRenderObjects to use correct stride calculation
  function buildRenderObjects(components: ComponentConfig[]): RenderObject[] {
    if(!gpuRef.current) return [];
    const { device, bindGroupLayout, pipelineCache, resources } = gpuRef.current;

      // Calculate required buffer sizes
    const { renderSize, pickingSize } = calculateBufferSize(components);

    // Create or recreate dynamic buffers if needed
    if (!gpuRef.current.dynamicBuffers ||
        gpuRef.current.dynamicBuffers.renderBuffer.size < renderSize ||
        gpuRef.current.dynamicBuffers.pickingBuffer.size < pickingSize) {

      // Cleanup old buffers if they exist
      if (gpuRef.current.dynamicBuffers) {
        gpuRef.current.dynamicBuffers.renderBuffer.destroy();
        gpuRef.current.dynamicBuffers.pickingBuffer.destroy();
      }

      gpuRef.current.dynamicBuffers = createDynamicBuffers(device, renderSize, pickingSize);
    }
    const dynamicBuffers = gpuRef.current.dynamicBuffers!;

      // Reset buffer offsets
      dynamicBuffers.renderOffset = 0;
      dynamicBuffers.pickingOffset = 0;

      // Initialize componentBaseId array
      gpuRef.current.componentBaseId = [];

      // Build ID mapping
      buildComponentIdMapping(components);

    const validRenderObjects: RenderObject[] = [];

      components.forEach((elem, i) => {
      const spec = primitiveRegistry[elem.type];
      if(!spec) {
        console.warn(`Unknown primitive type: ${elem.type}`);
        return;
      }

      try {
        const count = spec.getCount(elem);
        if (count === 0) {
          console.warn(`Component ${i} (${elem.type}) has no instances`);
          return;
        }

        const renderData = spec.buildRenderData(elem);
        if (!renderData) {
          console.warn(`Failed to build render data for component ${i} (${elem.type})`);
          return;
        }

        let renderOffset = 0;
        let stride = 0;
        if(renderData.length > 0) {
          renderOffset = Math.ceil(dynamicBuffers.renderOffset / 4) * 4;
          // Calculate stride based on float count (4 bytes per float)
          const floatsPerInstance = renderData.length / count;
          stride = Math.ceil(floatsPerInstance) * 4;

          device.queue.writeBuffer(
            dynamicBuffers.renderBuffer,
            renderOffset,
            renderData.buffer,
            renderData.byteOffset,
            renderData.byteLength
          );
          dynamicBuffers.renderOffset = renderOffset + (stride * count);
        }

        const pipeline = spec.getRenderPipeline(device, bindGroupLayout, pipelineCache);
        if (!pipeline) {
          console.warn(`Failed to create pipeline for component ${i} (${elem.type})`);
          return;
        }

        // Create render object directly instead of using createRenderObject
        const geometryResource = getGeometryResource(resources, elem.type);
        const renderObject: RenderObject = {
          pipeline,
          pickingPipeline: undefined,
          vertexBuffers: [
            geometryResource.vb,
            {  // Pass buffer info for instances
              buffer: dynamicBuffers.renderBuffer,
              offset: renderOffset,
              stride: stride
            }
          ],
          indexBuffer: geometryResource.ib,
          indexCount: geometryResource.indexCount,
          instanceCount: count,
          pickingVertexBuffers: [undefined, undefined] as [GPUBuffer | undefined, BufferInfo | undefined],
          pickingDataStale: true,
          componentIndex: i
        };

        if (!renderObject.vertexBuffers || renderObject.vertexBuffers.length !== 2) {
          console.warn(`Invalid vertex buffers for component ${i} (${elem.type})`);
          return;
        }

        validRenderObjects.push(renderObject);
      } catch (error) {
        console.error(`Error creating render object for component ${i} (${elem.type}):`, error);
      }
      });

    return validRenderObjects;
  }

  /******************************************************
   * D) Render pass (single call, no loop)
   ******************************************************/

  // Add validation helper
function isValidRenderObject(ro: RenderObject): ro is Required<Pick<RenderObject, 'pipeline' | 'vertexBuffers' | 'instanceCount'>> & {
  vertexBuffers: [GPUBuffer, BufferInfo];
} & RenderObject {
  return (
    ro.pipeline !== undefined &&
    Array.isArray(ro.vertexBuffers) &&
    ro.vertexBuffers.length === 2 &&
    ro.vertexBuffers[0] !== undefined &&
    ro.vertexBuffers[1] !== undefined &&
    'buffer' in ro.vertexBuffers[1] &&
    'offset' in ro.vertexBuffers[1] &&
    (ro.indexBuffer !== undefined || ro.vertexCount !== undefined) &&
    typeof ro.instanceCount === 'number' &&
    ro.instanceCount > 0
  );
}

  const renderFrame = useCallback((camState: CameraState) => {
    if(!gpuRef.current) return;
    const {
      device, context, uniformBuffer, uniformBindGroup,
      renderObjects, depthTexture
    } = gpuRef.current;

    const startTime = performance.now();  // Add timing measurement

    // Update camera uniforms
    const aspect = containerWidth / containerHeight;
    const view = glMatrix.mat4.lookAt(
      glMatrix.mat4.create(),
      camState.position,
      camState.target,
      camState.up
    );

    const proj = glMatrix.mat4.perspective(
      glMatrix.mat4.create(),
      glMatrix.glMatrix.toRadian(camState.fov),
      aspect,
      camState.near,
      camState.far
    );

    // Compute MVP matrix
    const mvp = glMatrix.mat4.multiply(
      glMatrix.mat4.create(),
      proj,
      view
    );

    // Compute camera vectors for lighting
    const forward = glMatrix.vec3.sub(glMatrix.vec3.create(), camState.target, camState.position);
    const right = glMatrix.vec3.cross(glMatrix.vec3.create(), forward, camState.up);
    glMatrix.vec3.normalize(right, right);

    const camUp = glMatrix.vec3.cross(glMatrix.vec3.create(), right, forward);
    glMatrix.vec3.normalize(camUp, camUp);
    glMatrix.vec3.normalize(forward, forward);

    // Compute light direction in camera space
    const lightDir = glMatrix.vec3.create();
    glMatrix.vec3.scaleAndAdd(lightDir, lightDir, right, LIGHTING.DIRECTION.RIGHT);
    glMatrix.vec3.scaleAndAdd(lightDir, lightDir, camUp, LIGHTING.DIRECTION.UP);
    glMatrix.vec3.scaleAndAdd(lightDir, lightDir, forward, LIGHTING.DIRECTION.FORWARD);
    glMatrix.vec3.normalize(lightDir, lightDir);

    // Write uniforms
    const uniformData = new Float32Array([
      ...Array.from(mvp),
      right[0], right[1], right[2], 0,  // pad to vec4
      camUp[0], camUp[1], camUp[2], 0,  // pad to vec4
      lightDir[0], lightDir[1], lightDir[2], 0,  // pad to vec4
      camState.position[0], camState.position[1], camState.position[2], 0  // Add camera position
    ]);
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    // Begin render pass
    const cmd = device.createCommandEncoder();
    const pass = cmd.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: 'clear',
        storeOp: 'store'
      }],
      depthStencilAttachment: depthTexture ? {
        view: depthTexture.createView(),
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store'
      } : undefined
    });

    // Draw each object
    for(const ro of renderObjects) {
      if (!isValidRenderObject(ro)) {
        continue;
      }

      // Now TypeScript knows ro.pipeline is defined
      pass.setPipeline(ro.pipeline);
      pass.setBindGroup(0, uniformBindGroup);

      // And ro.vertexBuffers[0] is defined
      pass.setVertexBuffer(0, ro.vertexBuffers[0]);

      // And ro.vertexBuffers[1] is defined
      const instanceInfo = ro.vertexBuffers[1];
      pass.setVertexBuffer(1, instanceInfo.buffer, instanceInfo.offset);

      if(ro.indexBuffer) {
        pass.setIndexBuffer(ro.indexBuffer, 'uint16');
        pass.drawIndexed(ro.indexCount ?? 0, ro.instanceCount ?? 1);
      } else {
        pass.draw(ro.vertexCount ?? 0, ro.instanceCount ?? 1);
      }
    }

    pass.end();
    device.queue.submit([cmd.finish()]);

    // Measure frame time and report it
    const endTime = performance.now();
    const frameTime = endTime - startTime;
    onFrameRendered?.(frameTime);
  }, [containerWidth, containerHeight, onFrameRendered]);  // Add onFrameRendered to deps

  /******************************************************
   * E) Pick pass (on hover/click)
   ******************************************************/
  async function pickAtScreenXY(screenX: number, screenY: number, mode: 'hover'|'click') {
    if(!gpuRef.current || !canvasRef.current || pickingLockRef.current) return;
    const pickingId = Date.now();
    const currentPickingId = pickingId;
    pickingLockRef.current = true;

    try {
      const {
        device, pickTexture, pickDepthTexture, readbackBuffer,
        uniformBindGroup, renderObjects, idToComponent
      } = gpuRef.current;
      if(!pickTexture || !pickDepthTexture || !readbackBuffer) return;
      if (currentPickingId !== pickingId) return;

      // Ensure picking data is ready for all objects
      renderObjects.forEach((ro, i) => {
        if (ro.pickingDataStale) {
          ensurePickingData(ro, components[i]);
        }
      });

      // Convert screen coordinates to device pixels
      const dpr = window.devicePixelRatio || 1;
      const pickX = Math.floor(screenX * dpr);
      const pickY = Math.floor(screenY * dpr);
      const displayWidth = Math.floor(containerWidth * dpr);
      const displayHeight = Math.floor(containerHeight * dpr);

      if(pickX < 0 || pickY < 0 || pickX >= displayWidth || pickY >= displayHeight) {
        if(mode === 'hover') handleHoverID(0);
        return;
      }

      const cmd = device.createCommandEncoder({label: 'Picking encoder'});
      const passDesc: GPURenderPassDescriptor = {
        colorAttachments:[{
          view: pickTexture.createView(),
          clearValue:{r:0,g:0,b:0,a:1},
          loadOp:'clear',
          storeOp:'store'
        }],
        depthStencilAttachment:{
          view: pickDepthTexture.createView(),
          depthClearValue:1.0,
          depthLoadOp:'clear',
          depthStoreOp:'store'
        }
      };
      const pass = cmd.beginRenderPass(passDesc);
      pass.setBindGroup(0, uniformBindGroup);

      for(const ro of renderObjects) {
        if (!ro.pickingPipeline || !ro.pickingVertexBuffers || ro.pickingVertexBuffers.length !== 2) {
          continue;
        }

        pass.setPipeline(ro.pickingPipeline);

        // Set geometry buffer (always first)
        const geometryBuffer = ro.pickingVertexBuffers[0];
        pass.setVertexBuffer(0, geometryBuffer);

        // Set instance buffer (always second)
        const instanceInfo = ro.pickingVertexBuffers[1];
        if (instanceInfo) {
          pass.setVertexBuffer(1, instanceInfo.buffer, instanceInfo.offset);
        }

        if(ro.pickingIndexBuffer) {
          pass.setIndexBuffer(ro.pickingIndexBuffer, 'uint16');
          pass.drawIndexed(ro.pickingIndexCount ?? 0, ro.pickingInstanceCount ?? 1);
        } else {
          pass.draw(ro.pickingVertexCount ?? 0, ro.pickingInstanceCount ?? 1);
        }
      }

      pass.end();

      cmd.copyTextureToBuffer(
        {texture: pickTexture, origin:{x:pickX,y:pickY}},
        {buffer: readbackBuffer, bytesPerRow:256, rowsPerImage:1},
        [1,1,1]
      );
      device.queue.submit([cmd.finish()]);

      if (currentPickingId !== pickingId) return;
      await readbackBuffer.mapAsync(GPUMapMode.READ);
      if (currentPickingId !== pickingId) {
        readbackBuffer.unmap();
        return;
      }
      const arr = new Uint8Array(readbackBuffer.getMappedRange());
      const r=arr[0], g=arr[1], b=arr[2];
      readbackBuffer.unmap();
      const pickedID = (b<<16)|(g<<8)|r;

      if(mode==='hover'){
        handleHoverID(pickedID);
      } else {
        handleClickID(pickedID);
      }
    } finally {
      pickingLockRef.current = false;
    }
  }

  function handleHoverID(pickedID: number) {
    if (!gpuRef.current) return;
    const { idToComponent } = gpuRef.current;

    // Get new hover state
    const newHoverState = idToComponent[pickedID] || null;

    // If hover state hasn't changed, do nothing
    if ((!lastHoverState.current && !newHoverState) ||
        (lastHoverState.current && newHoverState &&
         lastHoverState.current.componentIdx === newHoverState.componentIdx &&
         lastHoverState.current.instanceIdx === newHoverState.instanceIdx)) {
      return;
    }

    // Clear previous hover if it exists
    if (lastHoverState.current) {
      const prevComponent = components[lastHoverState.current.componentIdx];
      prevComponent?.onHover?.(null);
    }

    // Set new hover if it exists
    if (newHoverState) {
      const { componentIdx, instanceIdx } = newHoverState;
      if (componentIdx >= 0 && componentIdx < components.length) {
        components[componentIdx].onHover?.(instanceIdx);
      }
    }

    // Update last hover state
    lastHoverState.current = newHoverState;
  }

  function handleClickID(pickedID:number){
    if(!gpuRef.current) return;
    const {idToComponent} = gpuRef.current;
    const rec = idToComponent[pickedID];
    if(!rec) return;
    const {componentIdx, instanceIdx} = rec;
    if(componentIdx<0||componentIdx>=components.length) return;
    components[componentIdx].onClick?.(instanceIdx);
  }

  /******************************************************
   * F) Mouse Handling
   ******************************************************/
  /**
   * Tracks the current state of mouse interaction with the scene.
   * Used for camera control and picking operations.
   */
  interface MouseState {
    /** Current interaction mode */
    type: 'idle'|'dragging';

    /** Which mouse button initiated the drag (0=left, 1=middle, 2=right) */
    button?: number;

    /** Initial X coordinate when drag started */
    startX?: number;

    /** Initial Y coordinate when drag started */
    startY?: number;

    /** Most recent X coordinate during drag */
    lastX?: number;

    /** Most recent Y coordinate during drag */
    lastY?: number;

    /** Whether shift key was held when drag started */
    isShiftDown?: boolean;

    /** Accumulated drag distance in pixels */
    dragDistance?: number;
  }
  const mouseState=useRef<MouseState>({type:'idle'});

  // Add throttling for hover picking
  const throttledPickAtScreenXY = useCallback(
    throttle((x: number, y: number, mode: 'hover'|'click') => {
      pickAtScreenXY(x, y, mode);
    }, 32), // ~30fps
    [pickAtScreenXY]
  );

  // Rename to be more specific to scene3d
  const handleScene3dMouseMove = useCallback((e: MouseEvent) => {
    if(!canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const st = mouseState.current;
    if(st.type === 'dragging' && st.lastX !== undefined && st.lastY !== undefined) {
        const dx = e.clientX - st.lastX;
        const dy = e.clientY - st.lastY;
        st.dragDistance = (st.dragDistance||0) + Math.sqrt(dx*dx + dy*dy);

        if(st.button === 2 || st.isShiftDown) {
            handleCameraUpdate(cam => pan(cam, dx, dy));
        } else if(st.button === 0) {
            handleCameraUpdate(cam => orbit(cam, dx, dy));
        }

        st.lastX = e.clientX;
        st.lastY = e.clientY;
    } else if(st.type === 'idle') {
        throttledPickAtScreenXY(x, y, 'hover');
    }
}, [handleCameraUpdate, throttledPickAtScreenXY]);

  const handleScene3dMouseDown = useCallback((e: MouseEvent) => {
    mouseState.current = {
      type: 'dragging',
      button: e.button,
      startX: e.clientX,
      startY: e.clientY,
      lastX: e.clientX,
      lastY: e.clientY,
      isShiftDown: e.shiftKey,
      dragDistance: 0
    };
    e.preventDefault();
  }, []);

  const handleScene3dMouseUp = useCallback((e: MouseEvent) => {
    const st = mouseState.current;
    if(st.type === 'dragging' && st.startX !== undefined && st.startY !== undefined) {
      if(!canvasRef.current) return;
      const rect = canvasRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      if((st.dragDistance || 0) < 4) {
        pickAtScreenXY(x, y, 'click');
      }
    }
    mouseState.current = {type: 'idle'};
  }, [pickAtScreenXY]);

  const handleScene3dMouseLeave = useCallback(() => {
    mouseState.current = {type: 'idle'};
  }, []);

  // Update event listener references
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    canvas.addEventListener('mousemove', handleScene3dMouseMove);
    canvas.addEventListener('mousedown', handleScene3dMouseDown);
    canvas.addEventListener('mouseup', handleScene3dMouseUp);
    canvas.addEventListener('mouseleave', handleScene3dMouseLeave);

    return () => {
      canvas.removeEventListener('mousemove', handleScene3dMouseMove);
      canvas.removeEventListener('mousedown', handleScene3dMouseDown);
      canvas.removeEventListener('mouseup', handleScene3dMouseUp);
      canvas.removeEventListener('mouseleave', handleScene3dMouseLeave);
    };
  }, [handleScene3dMouseMove, handleScene3dMouseDown, handleScene3dMouseUp, handleScene3dMouseLeave]);

  /******************************************************
   * G) Lifecycle & Render-on-demand
   ******************************************************/
  // Init once
  useEffect(()=>{
    initWebGPU();
    return () => {
      if (gpuRef.current) {
        const { device, resources, pipelineCache } = gpuRef.current;

        device.queue.onSubmittedWorkDone().then(() => {
          for (const resource of Object.values(resources)) {
            if (resource) {
              resource.vb.destroy();
              resource.ib.destroy();
            }
          }

          // Clear instance pipeline cache
          pipelineCache.clear();
        });
      }
    };
  },[initWebGPU]);

  // Create/recreate depth + pick textures
  useEffect(()=>{
    if(isReady){
      createOrUpdateDepthTexture();
      createOrUpdatePickTextures();
    }
  },[isReady, containerWidth, containerHeight, createOrUpdateDepthTexture, createOrUpdatePickTextures]);

  // Update the render-triggering effects
  useEffect(() => {
    if (isReady) {
      renderFrame(activeCamera);  // Always render with current camera (controlled or internal)
    }
  }, [isReady, activeCamera, renderFrame]); // Watch the camera value

  // Update canvas size effect
  useEffect(() => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const dpr = window.devicePixelRatio || 1;
    const displayWidth = Math.floor(containerWidth * dpr);
    const displayHeight = Math.floor(containerHeight * dpr);

    // Only update if size actually changed
    if (canvas.width !== displayWidth || canvas.height !== displayHeight) {
        canvas.width = displayWidth;
        canvas.height = displayHeight;

        // Update textures after canvas size change
        createOrUpdateDepthTexture();
        createOrUpdatePickTextures();
        renderFrame(activeCamera);
    }
}, [containerWidth, containerHeight, createOrUpdateDepthTexture, createOrUpdatePickTextures, renderFrame, activeCamera]);

  // Update components effect
  useEffect(() => {
    if (isReady && gpuRef.current) {
      const ros = buildRenderObjects(components);
      gpuRef.current.renderObjects = ros;
      renderFrame(activeCamera);
    }
  }, [isReady, components]); // Remove activeCamera dependency

  // Add separate effect just for camera updates
  useEffect(() => {
    if (isReady && gpuRef.current) {
      renderFrame(activeCamera);
    }
  }, [isReady, activeCamera, renderFrame]);

  // Wheel handling
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const handleWheel = (e: WheelEvent) => {
        if (mouseState.current.type === 'idle') {
            e.preventDefault();
            handleCameraUpdate(cam => zoom(cam, e.deltaY));
        }
    };

    canvas.addEventListener('wheel', handleWheel, { passive: false });
    return () => canvas.removeEventListener('wheel', handleWheel);
  }, [handleCameraUpdate]);

  // Move ensurePickingData inside component
  const ensurePickingData = useCallback((renderObject: RenderObject, component: ComponentConfig) => {
    if (!renderObject.pickingDataStale) return;
    if (!gpuRef.current) return;

    const { device, bindGroupLayout, pipelineCache, resources } = gpuRef.current;

    // Calculate sizes before creating buffers
    const { renderSize, pickingSize } = calculateBufferSize(components);

    // Ensure dynamic buffers exist
    if (!gpuRef.current.dynamicBuffers) {
      gpuRef.current.dynamicBuffers = createDynamicBuffers(device, renderSize, pickingSize);
    }
    const dynamicBuffers = gpuRef.current.dynamicBuffers!;

    const spec = primitiveRegistry[component.type];
    if (!spec) return;

    // Build picking data
    const pickData = spec.buildPickingData(component, gpuRef.current.componentBaseId[renderObject.componentIndex]);
    if (pickData && pickData.length > 0) {
    const pickingOffset = Math.ceil(dynamicBuffers.pickingOffset / 4) * 4;
      // Calculate stride based on float count (4 bytes per float)
      const floatsPerInstance = pickData.length / renderObject.instanceCount!;
      const stride = Math.ceil(floatsPerInstance) * 4; // Align to 4 bytes

    device.queue.writeBuffer(
      dynamicBuffers.pickingBuffer,
      pickingOffset,
        pickData.buffer,
        pickData.byteOffset,
        pickData.byteLength
    );

      // Set picking buffers with offset info
      const geometryVB = renderObject.vertexBuffers[0];
    renderObject.pickingVertexBuffers = [
        geometryVB,
      {
        buffer: dynamicBuffers.pickingBuffer,
        offset: pickingOffset,
          stride: stride
      }
    ];

      dynamicBuffers.pickingOffset = pickingOffset + (stride * renderObject.instanceCount!);
    }

    // Get picking pipeline
    renderObject.pickingPipeline = spec.getPickingPipeline(device, bindGroupLayout, pipelineCache);

    // Copy over index buffer and counts from render object
    renderObject.pickingIndexBuffer = renderObject.indexBuffer;
    renderObject.pickingIndexCount = renderObject.indexCount;
    renderObject.pickingInstanceCount = renderObject.instanceCount;

    renderObject.pickingDataStale = false;
  }, [components, buildComponentIdMapping]);

  return (
    <div style={{ width: '100%', border: '1px solid #ccc' }}>
        <canvas
            ref={canvasRef}
            style={style}
        />
    </div>
  );
}
