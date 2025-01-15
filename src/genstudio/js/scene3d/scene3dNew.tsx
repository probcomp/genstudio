/// <reference path="./webgpu.d.ts" />
/// <reference types="react" />

import React, { useRef, useEffect, useState, useCallback } from 'react';
import { useContainerWidth } from '../utils';

/******************************************************
 * 1) Define Types
 ******************************************************/

interface PointCloudData {
  positions: Float32Array;     // [x, y, z, ...]
  colors?: Float32Array;       // [r, g, b, ...], 0..1
  scales?: Float32Array;       // optional
}

interface EllipsoidData {
  centers: Float32Array;       // [cx, cy, cz, ...]
  radii: Float32Array;         // [rx, ry, rz, ...]
  colors?: Float32Array;       // [r, g, b, ...], 0..1
}

interface Decoration {
  indexes: number[];
  color?: [number, number, number];
  alpha?: number;
  scale?: number;
  minSize?: number;
}

interface PointCloudElementConfig {
  type: 'PointCloud';
  data: PointCloudData;
  decorations?: Decoration[];
}

interface EllipsoidElementConfig {
  type: 'Ellipsoid';
  data: EllipsoidData;
  decorations?: Decoration[];
}

type SceneElementConfig = PointCloudElementConfig | EllipsoidElementConfig;

/******************************************************
 * 2) Minimal Camera State
 ******************************************************/
interface CameraState {
  orbitRadius: number;
  orbitTheta: number;
  orbitPhi: number;
  panX: number;
  panY: number;
  fov: number;
  near: number;
  far: number;
}

/******************************************************
 * 3) React Component Props
 ******************************************************/
interface SceneProps {
  elements: SceneElementConfig[];
  containerWidth: number;
}

/******************************************************
 * 4) SceneWrapper
 ******************************************************/
export function SceneWrapper({ elements }: { elements: SceneElementConfig[] }) {
  const [containerRef, measuredWidth] = useContainerWidth(1);
  return (
    <div ref={containerRef} style={{ width: '100%', height: '600px' }}>
      {measuredWidth > 0 && (
        <Scene containerWidth={measuredWidth} elements={elements} />
      )}
    </div>
  );
}

/******************************************************
 * 5) The Scene Component
 ******************************************************/
function Scene({ elements, containerWidth }: SceneProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const safeWidth = containerWidth > 0 ? containerWidth : 300;
  const canvasWidth = safeWidth;
  const canvasHeight = safeWidth;

  // GPU + pipeline references
  const gpuRef = useRef<{
    device: GPUDevice;
    context: GPUCanvasContext;
    // For point clouds
    billboardPipeline: GPURenderPipeline;
    billboardQuadVB: GPUBuffer;
    billboardQuadIB: GPUBuffer;
    // For ellipsoids
    ellipsoidPipeline: GPURenderPipeline;
    sphereVB: GPUBuffer;
    sphereIB: GPUBuffer;
    sphereIndexCount: number;

    uniformBuffer: GPUBuffer;
    uniformBindGroup: GPUBindGroup; // We'll store the uniform bind group here

    // Instances
    pcInstanceBuffer: GPUBuffer | null;
    pcInstanceCount: number;
    ellipsoidInstanceBuffer: GPUBuffer | null;
    ellipsoidInstanceCount: number;
  } | null>(null);

  // Render loop handle
  const rafIdRef = useRef<number>(0);

  const [isReady, setIsReady] = useState(false);

  // Camera
  const [camera, setCamera] = useState<CameraState>({
    orbitRadius: 2.0,
    orbitTheta: 0.2,
    orbitPhi: 1.0,
    panX: 0.0,
    panY: 0.0,
    fov: Math.PI / 3,
    near: 0.01,
    far: 100.0,
  });

  /******************************************************
   * A) Minimal Math
   ******************************************************/
  function mat4Multiply(a: Float32Array, b: Float32Array): Float32Array {
    const out = new Float32Array(16);
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        out[j * 4 + i] =
          a[i + 0] * b[j * 4 + 0] +
          a[i + 4] * b[j * 4 + 1] +
          a[i + 8] * b[j * 4 + 2] +
          a[i + 12] * b[j * 4 + 3];
      }
    }
    return out;
  }

  function mat4Perspective(fov: number, aspect: number, near: number, far: number): Float32Array {
    const out = new Float32Array(16);
    const f = 1.0 / Math.tan(fov / 2);
    out[0] = f / aspect;
    out[5] = f;
    out[10] = (far + near) / (near - far);
    out[11] = -1;
    out[14] = (2 * far * near) / (near - far);
    return out;
  }

  function mat4LookAt(eye: [number, number, number], target: [number, number, number], up: [number, number, number]): Float32Array {
    const zAxis = normalize([
      eye[0] - target[0],
      eye[1] - target[1],
      eye[2] - target[2],
    ]);
    const xAxis = normalize(cross(up, zAxis));
    const yAxis = cross(zAxis, xAxis);

    const out = new Float32Array(16);
    out[0] = xAxis[0];
    out[1] = yAxis[0];
    out[2] = zAxis[0];
    out[3] = 0;

    out[4] = xAxis[1];
    out[5] = yAxis[1];
    out[6] = zAxis[1];
    out[7] = 0;

    out[8] = xAxis[2];
    out[9] = yAxis[2];
    out[10] = zAxis[2];
    out[11] = 0;

    out[12] = -dot(xAxis, eye);
    out[13] = -dot(yAxis, eye);
    out[14] = -dot(zAxis, eye);
    out[15] = 1;
    return out;
  }

  function cross(a: [number, number, number], b: [number, number, number]): [number, number, number] {
    return [
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0],
    ];
  }
  function dot(a: [number, number, number], b: [number, number, number]): number {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  }
  function normalize(v: [number, number, number]): [number, number, number] {
    const len = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (len > 1e-6) {
      return [v[0] / len, v[1] / len, v[2] / len];
    }
    return [0, 0, 0];
  }

  /******************************************************
   * B) Generate Geometry
   ******************************************************/

  // 1) The old billboard quad
  const QUAD_VERTICES = new Float32Array([
    -0.5, -0.5,
     0.5, -0.5,
    -0.5,  0.5,
     0.5,  0.5,
  ]);
  const QUAD_INDICES = new Uint16Array([0, 1, 2, 2, 1, 3]);

  // 2) A sphere for the ellipsoids
  function createSphereGeometry(stacks = 16, slices = 24) {
    const positions: number[] = [];
    const indices: number[] = [];

    for (let i = 0; i <= stacks; i++) {
      const phi = (i / stacks) * Math.PI; // 0..π
      const y = Math.cos(phi);
      const r = Math.sin(phi);

      for (let j = 0; j <= slices; j++) {
        const theta = (j / slices) * 2 * Math.PI; // 0..2π
        const x = r * Math.sin(theta);
        const z = r * Math.cos(theta);
        positions.push(x, y, z);
      }
    }

    for (let i = 0; i < stacks; i++) {
      for (let j = 0; j < slices; j++) {
        const row1 = i * (slices + 1) + j;
        const row2 = (i + 1) * (slices + 1) + j;

        indices.push(row1, row2, row1 + 1);
        indices.push(row1 + 1, row2, row2 + 1);
      }
    }

    return {
      positions: new Float32Array(positions),
      indices: new Uint16Array(indices),
    };
  }

  /******************************************************
   * C) Initialize WebGPU
   ******************************************************/
  const initWebGPU = useCallback(async () => {
    if (!canvasRef.current) return;
    if (!navigator.gpu) {
      console.error('WebGPU not supported.');
      return;
    }

    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) throw new Error('Failed to get GPU adapter.');
      const device = await adapter.requestDevice();

      const context = canvasRef.current.getContext('webgpu') as GPUCanvasContext;
      const format = navigator.gpu.getPreferredCanvasFormat();
      context.configure({ device, format, alphaMode: 'opaque' });

      /*********************************
       * 1) Create geometry buffers
       *********************************/
      // For point-cloud billboards
      const billboardQuadVB = device.createBuffer({
        size: QUAD_VERTICES.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(billboardQuadVB, 0, QUAD_VERTICES);

      const billboardQuadIB = device.createBuffer({
        size: QUAD_INDICES.byteLength,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(billboardQuadIB, 0, QUAD_INDICES);

      // For ellipsoids (3D sphere)
      const sphereGeo = createSphereGeometry(16, 24);
      const sphereVB = device.createBuffer({
        size: sphereGeo.positions.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(sphereVB, 0, sphereGeo.positions);

      const sphereIB = device.createBuffer({
        size: sphereGeo.indices.byteLength,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(sphereIB, 0, sphereGeo.indices);

      /*********************************
       * 2) Create uniform buffer
       *********************************/
      // We'll store a 4x4 MVP (16 floats * 4 bytes = 64)
      const uniformBufferSize = 64;
      const uniformBuffer = device.createBuffer({
        size: uniformBufferSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });

      /*********************************
       * 3) Create a shared pipeline layout
       *********************************/
      // A) Create a uniform bind group layout
      const uniformBindGroupLayout = device.createBindGroupLayout({
        entries: [
          {
            binding: 0,
            visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
            buffer: { type: 'uniform' },
          },
        ],
      });

      // B) Create a pipeline layout that uses it for slot 0
      const pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [uniformBindGroupLayout],
      });

      // C) Create both pipelines using that pipelineLayout
      //    1) Billboard pipeline
      const billboardPipeline = device.createRenderPipeline({
        layout: pipelineLayout, // <--- explicit
        vertex: {
          module: device.createShaderModule({
            code: `
struct Camera {
  mvp : mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> camera : Camera;

struct VertexOut {
  @builtin(position) Position : vec4<f32>,
  @location(2) color : vec3<f32>,
  @location(3) alpha : f32,
};

@vertex
fn vs_main(
  @location(0) corner : vec2<f32>,
  @location(1) pos    : vec3<f32>,
  @location(2) col    : vec3<f32>,
  @location(3) alpha  : f32,
  @location(4) scaleX : f32,
  @location(5) scaleY : f32
) -> VertexOut {
  var out: VertexOut;
  let worldPos = vec4<f32>(
    pos.x + corner.x * scaleX,
    pos.y + corner.y * scaleY,
    pos.z,
    1.0
  );

  out.Position = camera.mvp * worldPos;
  out.color = col;
  out.alpha = alpha;
  return out;
}

@fragment
fn fs_main(
  @location(2) inColor : vec3<f32>,
  @location(3) inAlpha : f32
) -> @location(0) vec4<f32> {
  return vec4<f32>(inColor, inAlpha);
}
`
          }),
          entryPoint: 'vs_main',
          buffers: [
            // billboard corners
            {
              arrayStride: 2 * 4,
              attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x2' }],
            },
            // instance data (pos(3), color(3), alpha(1), scaleX(1), scaleY(1))
            {
              arrayStride: 9 * 4,
              stepMode: 'instance',
              attributes: [
                { shaderLocation: 1, offset: 0,       format: 'float32x3' },
                { shaderLocation: 2, offset: 3 * 4,   format: 'float32x3' },
                { shaderLocation: 3, offset: 6 * 4,   format: 'float32'   },
                { shaderLocation: 4, offset: 7 * 4,   format: 'float32'   },
                { shaderLocation: 5, offset: 8 * 4,   format: 'float32'   },
              ],
            }
          ],
        },
        fragment: {
          module: device.createShaderModule({
            code: `
@fragment
fn fs_main(
  @location(2) inColor : vec3<f32>,
  @location(3) inAlpha : f32
) -> @location(0) vec4<f32> {
  return vec4<f32>(inColor, inAlpha);
}
`
          }),
          entryPoint: 'fs_main',
          targets: [{ format }],
        },
        primitive: { topology: 'triangle-list' },
      });

      //    2) Ellipsoid pipeline
      const ellipsoidPipeline = device.createRenderPipeline({
        layout: pipelineLayout, // <--- explicit
        vertex: {
          module: device.createShaderModule({
            code: `
struct Camera {
  mvp : mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> camera : Camera;

struct VertexOut {
  @builtin(position) Position : vec4<f32>,
  @location(1) color : vec3<f32>,
  @location(2) alpha : f32,
};

@vertex
fn vs_main(
  @location(0) localPos : vec3<f32>,
  @location(1) pos      : vec3<f32>,
  @location(2) scale    : vec3<f32>,
  @location(3) col      : vec3<f32>,
  @location(4) alpha    : f32
) -> VertexOut {
  var out: VertexOut;
  let worldPos = vec4<f32>(
    pos.x + localPos.x * scale.x,
    pos.y + localPos.y * scale.y,
    pos.z + localPos.z * scale.z,
    1.0
  );
  out.Position = camera.mvp * worldPos;
  out.color = col;
  out.alpha = alpha;
  return out;
}

@fragment
fn fs_main(
  @location(1) inColor : vec3<f32>,
  @location(2) inAlpha : f32
) -> @location(0) vec4<f32> {
  return vec4<f32>(inColor, inAlpha);
}
`
          }),
          entryPoint: 'vs_main',
          buffers: [
            // sphere geometry
            {
              arrayStride: 3 * 4,
              attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }],
            },
            // instance data: pos(3), scale(3), color(3), alpha(1) => total 10 floats
            {
              arrayStride: 10 * 4,
              stepMode: 'instance',
              attributes: [
                { shaderLocation: 1, offset: 0,       format: 'float32x3' },
                { shaderLocation: 2, offset: 3 * 4,   format: 'float32x3' },
                { shaderLocation: 3, offset: 6 * 4,   format: 'float32x3' },
                { shaderLocation: 4, offset: 9 * 4,   format: 'float32'   },
              ],
            }
          ],
        },
        fragment: {
          module: device.createShaderModule({
            code: `
@fragment
fn fs_main(
  @location(1) inColor : vec3<f32>,
  @location(2) inAlpha : f32
) -> @location(0) vec4<f32> {
  return vec4<f32>(inColor, inAlpha);
}
`
          }),
          entryPoint: 'fs_main',
          targets: [{ format }],
        },
        primitive: { topology: 'triangle-list' },
      });

      /*********************************
       * 4) Create a uniform bind group
       *********************************/
      const uniformBindGroup = device.createBindGroup({
        layout: uniformBindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: uniformBuffer } },
        ],
      });

      // Final: store everything in our ref
      gpuRef.current = {
        device,
        context,
        billboardPipeline,
        billboardQuadVB,
        billboardQuadIB,
        ellipsoidPipeline,
        sphereVB,
        sphereIB,
        sphereIndexCount: sphereGeo.indices.length,
        uniformBuffer,
        uniformBindGroup,
        pcInstanceBuffer: null,
        pcInstanceCount: 0,
        ellipsoidInstanceBuffer: null,
        ellipsoidInstanceCount: 0,
      };

      setIsReady(true);
    } catch (err) {
      console.error('Error initializing WebGPU:', err);
    }
  }, []);

  /******************************************************
   * D) Render Loop
   ******************************************************/
  const renderFrame = useCallback(() => {
    if (rafIdRef.current) cancelAnimationFrame(rafIdRef.current);

    if (!gpuRef.current) {
      rafIdRef.current = requestAnimationFrame(renderFrame);
      return;
    }

    const {
      device, context,
      billboardPipeline, billboardQuadVB, billboardQuadIB,
      ellipsoidPipeline, sphereVB, sphereIB, sphereIndexCount,
      uniformBuffer, uniformBindGroup,
      pcInstanceBuffer, pcInstanceCount,
      ellipsoidInstanceBuffer, ellipsoidInstanceCount,
    } = gpuRef.current;

    // 1) Build MVP
    const aspect = canvasWidth / canvasHeight;
    const proj = mat4Perspective(camera.fov, aspect, camera.near, camera.far);

    const cx = camera.orbitRadius * Math.sin(camera.orbitPhi) * Math.sin(camera.orbitTheta);
    const cy = camera.orbitRadius * Math.cos(camera.orbitPhi);
    const cz = camera.orbitRadius * Math.sin(camera.orbitPhi) * Math.cos(camera.orbitTheta);
    const eye: [number, number, number] = [cx, cy, cz];
    const target: [number, number, number] = [camera.panX, camera.panY, 0];
    const view = mat4LookAt(eye, target, [0, 1, 0]);
    const mvp = mat4Multiply(proj, view);

    // 2) Write MVP
    device.queue.writeBuffer(uniformBuffer, 0, mvp);

    // 3) Acquire swapchain texture
    let texture: GPUTexture;
    try {
      texture = context.getCurrentTexture();
    } catch {
      rafIdRef.current = requestAnimationFrame(renderFrame);
      return;
    }

    // 4) Encode commands
    const passDesc: GPURenderPassDescriptor = {
      colorAttachments: [
        {
          view: texture.createView(),
          clearValue: { r: 0.15, g: 0.15, b: 0.15, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    };

    const cmdEncoder = device.createCommandEncoder();
    const passEncoder = cmdEncoder.beginRenderPass(passDesc);

    // A) Draw point cloud (billboards)
    if (pcInstanceBuffer && pcInstanceCount > 0) {
      passEncoder.setPipeline(billboardPipeline);
      passEncoder.setBindGroup(0, uniformBindGroup);
      passEncoder.setVertexBuffer(0, billboardQuadVB);
      passEncoder.setIndexBuffer(billboardQuadIB, 'uint16');
      passEncoder.setVertexBuffer(1, pcInstanceBuffer);
      passEncoder.drawIndexed(QUAD_INDICES.length, pcInstanceCount);
    }

    // B) Draw ellipsoids (3D sphere geometry)
    if (ellipsoidInstanceBuffer && ellipsoidInstanceCount > 0) {
      passEncoder.setPipeline(ellipsoidPipeline);
      passEncoder.setBindGroup(0, uniformBindGroup);
      passEncoder.setVertexBuffer(0, sphereVB);
      passEncoder.setIndexBuffer(sphereIB, 'uint16');
      passEncoder.setVertexBuffer(1, ellipsoidInstanceBuffer);
      passEncoder.drawIndexed(sphereIndexCount, ellipsoidInstanceCount);
    }

    passEncoder.end();
    device.queue.submit([cmdEncoder.finish()]);
    rafIdRef.current = requestAnimationFrame(renderFrame);
  }, [camera, canvasWidth, canvasHeight]);

  /******************************************************
   * E) Building Instance Data
   ******************************************************/
  function buildPCInstanceData(
    positions: Float32Array,
    colors?: Float32Array,
    scales?: Float32Array,
    decorations?: Decoration[]
  ) {
    const count = positions.length / 3;
    // pos(3), color(3), alpha(1), scaleX(1), scaleY(1) => 9
    const data = new Float32Array(count * 9);

    for (let i = 0; i < count; i++) {
      const x = positions[i * 3 + 0];
      const y = positions[i * 3 + 1];
      const z = positions[i * 3 + 2];
      data[i * 9 + 0] = x;
      data[i * 9 + 1] = y;
      data[i * 9 + 2] = z;

      if (colors && colors.length === count * 3) {
        data[i * 9 + 3] = colors[i * 3 + 0];
        data[i * 9 + 4] = colors[i * 3 + 1];
        data[i * 9 + 5] = colors[i * 3 + 2];
      } else {
        data[i * 9 + 3] = 1;
        data[i * 9 + 4] = 1;
        data[i * 9 + 5] = 1;
      }

      data[i * 9 + 6] = 1.0; // alpha
      let s = scales ? scales[i] : 0.02;
      data[i * 9 + 7] = s; // scaleX
      data[i * 9 + 8] = s; // scaleY
    }

    if (decorations) {
      for (const dec of decorations) {
        const { indexes, color, alpha, scale, minSize } = dec;
        for (const idx of indexes) {
          if (idx < 0 || idx >= count) continue;
          if (color) {
            data[idx * 9 + 3] = color[0];
            data[idx * 9 + 4] = color[1];
            data[idx * 9 + 5] = color[2];
          }
          if (alpha !== undefined) {
            data[idx * 9 + 6] = alpha;
          }
          if (scale !== undefined) {
            data[idx * 9 + 7] *= scale;
            data[idx * 9 + 8] *= scale;
          }
          if (minSize !== undefined) {
            if (data[idx * 9 + 7] < minSize) data[idx * 9 + 7] = minSize;
            if (data[idx * 9 + 8] < minSize) data[idx * 9 + 8] = minSize;
          }
        }
      }
    }
    return data;
  }

  function buildEllipsoidInstanceData(
    centers: Float32Array,
    radii: Float32Array,
    colors?: Float32Array,
    decorations?: Decoration[],
  ) {
    const count = centers.length / 3;
    // pos(3), scale(3), color(3), alpha(1) => 10 floats
    const data = new Float32Array(count * 10);

    for (let i = 0; i < count; i++) {
      const cx = centers[i * 3 + 0];
      const cy = centers[i * 3 + 1];
      const cz = centers[i * 3 + 2];
      const rx = radii[i * 3 + 0] || 0.1;
      const ry = radii[i * 3 + 1] || 0.1;
      const rz = radii[i * 3 + 2] || 0.1;

      data[i * 10 + 0] = cx;
      data[i * 10 + 1] = cy;
      data[i * 10 + 2] = cz;

      data[i * 10 + 3] = rx;
      data[i * 10 + 4] = ry;
      data[i * 10 + 5] = rz;

      if (colors && colors.length === count * 3) {
        data[i * 10 + 6] = colors[i * 3 + 0];
        data[i * 10 + 7] = colors[i * 3 + 1];
        data[i * 10 + 8] = colors[i * 3 + 2];
      } else {
        data[i * 10 + 6] = 1;
        data[i * 10 + 7] = 1;
        data[i * 10 + 8] = 1;
      }
      data[i * 10 + 9] = 1.0; // alpha
    }

    if (decorations) {
      for (const dec of decorations) {
        const { indexes, color, alpha, scale, minSize } = dec;
        for (const idx of indexes) {
          if (idx < 0 || idx >= count) continue;
          if (color) {
            data[idx * 10 + 6] = color[0];
            data[idx * 10 + 7] = color[1];
            data[idx * 10 + 8] = color[2];
          }
          if (alpha !== undefined) {
            data[idx * 10 + 9] = alpha;
          }
          if (scale !== undefined) {
            data[idx * 10 + 3] *= scale;
            data[idx * 10 + 4] *= scale;
            data[idx * 10 + 5] *= scale;
          }
          if (minSize !== undefined) {
            if (data[idx * 10 + 3] < minSize) data[idx * 10 + 3] = minSize;
            if (data[idx * 10 + 4] < minSize) data[idx * 10 + 4] = minSize;
            if (data[idx * 10 + 5] < minSize) data[idx * 10 + 5] = minSize;
          }
        }
      }
    }
    return data;
  }

  /******************************************************
   * F) Updating Buffers
   ******************************************************/
  const updateBuffers = useCallback((sceneElements: SceneElementConfig[]) => {
    if (!gpuRef.current) return;
    const { device } = gpuRef.current;

    let pcInstData: Float32Array | null = null;
    let pcCount = 0;

    let ellipsoidInstData: Float32Array | null = null;
    let ellipsoidCount = 0;

    // For brevity, we handle only one point cloud and one ellipsoid
    for (const elem of sceneElements) {
      if (elem.type === 'PointCloud') {
        const { positions, colors, scales } = elem.data;
        if (!positions || positions.length === 0) continue;
        pcInstData = buildPCInstanceData(positions, colors, scales, elem.decorations);
        pcCount = positions.length / 3;
      }
      else if (elem.type === 'Ellipsoid') {
        const { centers, radii, colors } = elem.data;
        if (!centers || centers.length === 0) continue;
        ellipsoidInstData = buildEllipsoidInstanceData(centers, radii, colors, elem.decorations);
        ellipsoidCount = centers.length / 3;
      }
    }

    // Create buffers
    if (pcInstData && pcCount > 0) {
      const buf = device.createBuffer({
        size: pcInstData.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(buf, 0, pcInstData);
      gpuRef.current.pcInstanceBuffer?.destroy();
      gpuRef.current.pcInstanceBuffer = buf;
      gpuRef.current.pcInstanceCount = pcCount;
    } else {
      gpuRef.current.pcInstanceBuffer?.destroy();
      gpuRef.current.pcInstanceBuffer = null;
      gpuRef.current.pcInstanceCount = 0;
    }

    if (ellipsoidInstData && ellipsoidCount > 0) {
      const buf = device.createBuffer({
        size: ellipsoidInstData.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(buf, 0, ellipsoidInstData);
      gpuRef.current.ellipsoidInstanceBuffer?.destroy();
      gpuRef.current.ellipsoidInstanceBuffer = buf;
      gpuRef.current.ellipsoidInstanceCount = ellipsoidCount;
    } else {
      gpuRef.current.ellipsoidInstanceBuffer?.destroy();
      gpuRef.current.ellipsoidInstanceBuffer = null;
      gpuRef.current.ellipsoidInstanceCount = 0;
    }
  }, []);

  /******************************************************
   * G) Mouse + Zoom
   ******************************************************/
  const mouseState = useRef<{ x: number; y: number; button: number } | null>(null);

  const onMouseDown = useCallback((e: MouseEvent) => {
    mouseState.current = { x: e.clientX, y: e.clientY, button: e.button };
  }, []);

  const onMouseMove = useCallback((e: MouseEvent) => {
    if (!mouseState.current) return;
    const dx = e.clientX - mouseState.current.x;
    const dy = e.clientY - mouseState.current.y;
    mouseState.current.x = e.clientX;
    mouseState.current.y = e.clientY;

    if (mouseState.current.button === 2 || e.shiftKey) {
      // pan
      setCamera(cam => ({
        ...cam,
        panX: cam.panX - dx * 0.002,
        panY: cam.panY + dy * 0.002,
      }));
    } else if (mouseState.current.button === 0) {
      // orbit
      setCamera(cam => {
        const newPhi = Math.max(0.1, Math.min(Math.PI - 0.1, cam.orbitPhi - dy * 0.01));
        return {
          ...cam,
          orbitTheta: cam.orbitTheta - dx * 0.01,
          orbitPhi: newPhi,
        };
      });
    }
  }, []);

  const onMouseUp = useCallback(() => {
    mouseState.current = null;
  }, []);

  const onWheel = useCallback((e: WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY * 0.01;
    setCamera(cam => ({
      ...cam,
      orbitRadius: Math.max(0.01, cam.orbitRadius + delta),
    }));
  }, []);

  useEffect(() => {
    const c = canvasRef.current;
    if (!c) return;
    c.addEventListener('mousedown', onMouseDown);
    c.addEventListener('mousemove', onMouseMove);
    c.addEventListener('mouseup', onMouseUp);
    c.addEventListener('wheel', onWheel, { passive: false });
    return () => {
      c.removeEventListener('mousedown', onMouseDown);
      c.removeEventListener('mousemove', onMouseMove);
      c.removeEventListener('mouseup', onMouseUp);
      c.removeEventListener('wheel', onWheel);
    };
  }, [onMouseDown, onMouseMove, onMouseUp, onWheel]);

  /******************************************************
   * H) Effects
   ******************************************************/
  // Init WebGPU
  useEffect(() => {
    initWebGPU();
    return () => {
      // Cleanup
      if (gpuRef.current) {
        const {
          billboardQuadVB,
          billboardQuadIB,
          sphereVB,
          sphereIB,
          uniformBuffer,
          pcInstanceBuffer,
          ellipsoidInstanceBuffer,
        } = gpuRef.current;
        billboardQuadVB.destroy();
        billboardQuadIB.destroy();
        sphereVB.destroy();
        sphereIB.destroy();
        uniformBuffer.destroy();
        pcInstanceBuffer?.destroy();
        ellipsoidInstanceBuffer?.destroy();
      }
      if (rafIdRef.current) cancelAnimationFrame(rafIdRef.current);
    };
  }, [initWebGPU]);

  // Canvas resize
  useEffect(() => {
    if (!canvasRef.current) return;
    canvasRef.current.width = canvasWidth;
    canvasRef.current.height = canvasHeight;
  }, [canvasWidth, canvasHeight]);

  // Start render loop
  useEffect(() => {
    if (isReady) {
      renderFrame();
      rafIdRef.current = requestAnimationFrame(renderFrame);
    }
    return () => {
      if (rafIdRef.current) cancelAnimationFrame(rafIdRef.current);
    };
  }, [isReady, renderFrame]);

  // Update buffers
  useEffect(() => {
    if (isReady) {
      updateBuffers(elements);
    }
  }, [isReady, elements, updateBuffers]);

  return (
    <div style={{ width: '100%', border: '1px solid #ccc' }}>
      <canvas ref={canvasRef} style={{ width: '100%', height: canvasHeight }} />
    </div>
  );
}

/******************************************************
 * 6) Example: App
 ******************************************************/
export function App() {
  // a small point cloud
  const pcPositions = new Float32Array([
    -0.5, -0.5, 0.0,
     0.5, -0.5, 0.0,
    -0.5,  0.5, 0.0,
     0.5,  0.5, 0.0,
  ]);
  const pcColors = new Float32Array([
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
  ]);
  const pcDecorations: Decoration[] = [
    {
      indexes: [0],
      color: [1, 0, 0],
      scale: 1.0,
      alpha: 1.0,
      minSize: 0.03,
    },
    {
      indexes: [1],
      color: [0, 1, 0],
      scale: 2.0,
      alpha: 0.7,
    }
  ];
  const pcElement: PointCloudElementConfig = {
    type: 'PointCloud',
    data: { positions: pcPositions, colors: pcColors },
    decorations: pcDecorations,
  };

  // a couple of ellipsoids with real 3D geometry
  const centers = new Float32Array([
    0, 0, 0,
    0.6, 0.3, -0.2,
  ]);
  const radii = new Float32Array([
    0.2, 0.3, 0.15,
    0.05, 0.15, 0.3,
  ]);
  const ellipsoidColors = new Float32Array([
    1.0, 0.5, 0.2,
    0.2, 0.9, 1.0,
  ]);
  const ellipsoidDecorations: Decoration[] = [
    {
      indexes: [0],
      color: [1, 0.5, 0.2],
      alpha: 1.0,
      scale: 1.2,
      minSize: 0.1,
    },
    {
      indexes: [1],
      color: [0.1, 0.8, 1.0],
      alpha: 0.7,
    }
  ];
  const ellipsoidElement: EllipsoidElementConfig = {
    type: 'Ellipsoid',
    data: { centers, radii, colors: ellipsoidColors },
    decorations: ellipsoidDecorations,
  };

  const testElements: SceneElementConfig[] = [
    pcElement,
    ellipsoidElement,
  ];

  return (
    <>
      <h2>Point Clouds (billboards) + Real 3D Ellipsoids with Explicit Layout</h2>
      <SceneWrapper elements={testElements} />
      <p>
        Left-drag = orbit, Right-drag or Shift+Left = pan,
        mousewheel = zoom.<br />
        We now share an explicit pipeline layout and a single uniform
        bind group across two pipelines, avoiding layout mismatch errors.
      </p>
    </>
  );
}

export function Torus(props) {
  return <SceneWrapper {...props} />
}
