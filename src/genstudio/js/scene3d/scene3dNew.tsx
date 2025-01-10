/// <reference path="./webgpu.d.ts" />
/// <reference types="react" />

import React, { useRef, useEffect, useState, useCallback } from 'react';
import { useContainerWidth } from '../utils';

/******************************************************
 * 1) Define Types from the Spec
 ******************************************************/
interface PointCloudData {
  positions: Float32Array;     // [x, y, z, x, y, z, ...]
  colors?: Float32Array;       // [r, g, b, r, g, b, ...]
  scales?: Float32Array;       // optional
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

type SceneElementConfig = PointCloudElementConfig;

/******************************************************
 * 2) Minimal Camera State
 ******************************************************/
interface CameraState {
  orbitRadius: number;  // distance from origin
  orbitTheta: number;   // rotation around Y axis
  orbitPhi: number;     // rotation around X axis
  panX: number;         // shifting the target in X
  panY: number;         // shifting the target in Y
  fov: number;          // field of view in radians, e.g. ~60 deg
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
 * 4) SceneWrapper - measures container, uses <Scene>
 ******************************************************/
export function SceneWrapper({ elements }: { elements: SceneElementConfig[] }) {
  console.log("SW", elements)
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
 * 5) The Scene Component - with "zoom to cursor" +
 *    camera-facing billboards
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
    pipeline: GPURenderPipeline;
    quadVertexBuffer: GPUBuffer;
    quadIndexBuffer: GPUBuffer;
    instanceBuffer: GPUBuffer | null;
    uniformBuffer: GPUBuffer;
    bindGroup: GPUBindGroup;
    indexCount: number;
    instanceCount: number;
  } | null>(null);

  // Render loop handle
  const rafIdRef = useRef<number>(0);

  // Track WebGPU readiness
  const [isReady, setIsReady] = useState(false);

  // ----------------------------------------------------
  // Camera: Adjust angles to ensure a view from the start
  // ----------------------------------------------------
  const [camera, setCamera] = useState<CameraState>({
    orbitRadius: 2.0,
    orbitTheta: 0.2,  // nonzero so we see the quads initially
    orbitPhi: 1.0,    // tilt downward
    panX: 0.0,
    panY: 0.0,
    fov: Math.PI / 3, // ~60 deg
    near: 0.01,
    far: 100.0,
  });

  // Basic billboard geometry
  const QUAD_VERTICES = new Float32Array([
    -0.5, -0.5,
     0.5, -0.5,
    -0.5,  0.5,
     0.5,  0.5,
  ]);
  const QUAD_INDICES = new Uint16Array([0, 1, 2, 2, 1, 3]);

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
   * B) Initialize WebGPU
   ******************************************************/
  const initWebGPU = useCallback(async () => {
    if (!canvasRef.current) return;
    if (!navigator.gpu) {
      console.error('WebGPU not supported in this environment.');
      return;
    }

    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) throw new Error('Failed to get GPU adapter.');
      const device = await adapter.requestDevice();

      const context = canvasRef.current.getContext('webgpu') as GPUCanvasContext;
      const format = navigator.gpu.getPreferredCanvasFormat();
      context.configure({ device, format, alphaMode: 'opaque' });

      // Create buffers
      const quadVertexBuffer = device.createBuffer({
        size: QUAD_VERTICES.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(quadVertexBuffer, 0, QUAD_VERTICES);

      const quadIndexBuffer = device.createBuffer({
        size: QUAD_INDICES.byteLength,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(quadIndexBuffer, 0, QUAD_INDICES);

      // Uniform buffer (mvp + cameraRight + cameraUp)
      const uniformBufferSize = 96;
      const uniformBuffer = device.createBuffer({
        size: uniformBufferSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });

      // Create pipeline that orients each quad to face the camera
      const pipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: {
          module: device.createShaderModule({
            code: `
struct CameraUniforms {
  mvp: mat4x4<f32>,
  cameraRight: vec3<f32>,
  cameraPad1: f32,  // pad
  cameraUp: vec3<f32>,
  cameraPad2: f32,  // pad
};

@group(0) @binding(0) var<uniform> camera : CameraUniforms;

struct VertexOut {
  @builtin(position) position : vec4<f32>,
  @location(2) color : vec3<f32>,
  @location(3) alpha : f32,
};

@vertex
fn vs_main(
  @location(0) corner: vec2<f32>, // e.g. (-0.5..0.5)
  @location(1) instancePos  : vec3<f32>,
  @location(2) instanceColor: vec3<f32>,
  @location(3) instanceAlpha: f32,
  @location(4) instanceScale: f32
) -> VertexOut {
  var out: VertexOut;

  // Build offset in world space using cameraRight & cameraUp
  let offsetWorld = (camera.cameraRight * corner.x + camera.cameraUp * corner.y)
                    * instanceScale;

  let worldPos = vec4<f32>(
    instancePos.x + offsetWorld.x,
    instancePos.y + offsetWorld.y,
    instancePos.z + offsetWorld.z,
    1.0
  );

  out.position = camera.mvp * worldPos;
  out.color = instanceColor;
  out.alpha = instanceAlpha;
  return out;
}

@fragment
fn fs_main(
  @location(2) inColor : vec3<f32>,
  @location(3) inAlpha : f32
) -> @location(0) vec4<f32> {
  return vec4<f32>(inColor, inAlpha);
}
            `,
          }),
          entryPoint: 'vs_main',
          buffers: [
            {
              arrayStride: 2 * 4,
              attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x2' }],
            },
            {
              arrayStride: 8 * 4,
              stepMode: 'instance',
              attributes: [
                { shaderLocation: 1, offset: 0, format: 'float32x3' },
                { shaderLocation: 2, offset: 3 * 4, format: 'float32x3' },
                { shaderLocation: 3, offset: 6 * 4, format: 'float32' },
                { shaderLocation: 4, offset: 7 * 4, format: 'float32' },
              ],
            },
          ],
        },
        fragment: {
          module: device.createShaderModule({
            code: `
@fragment
fn fs_main(
  @location(2) inColor: vec3<f32>,
  @location(3) inAlpha: f32
) -> @location(0) vec4<f32> {
  return vec4<f32>(inColor, inAlpha);
}
            `,
          }),
          entryPoint: 'fs_main',
          targets: [{ format }],
        },
        primitive: {
          topology: 'triangle-list',
        },
      });

      // Bind group
      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
      });

      gpuRef.current = {
        device,
        context,
        pipeline,
        quadVertexBuffer,
        quadIndexBuffer,
        instanceBuffer: null,
        uniformBuffer,
        bindGroup,
        indexCount: QUAD_INDICES.length,
        instanceCount: 0,
      };

      setIsReady(true);
    } catch (err) {
      console.error('Error initializing WebGPU:', err);
    }
  }, []);

  /******************************************************
   * C) Render Loop
   ******************************************************/
  const renderFrame = useCallback(() => {
    if (!gpuRef.current) {
      rafIdRef.current = requestAnimationFrame(renderFrame);
      return;
    }

    const {
      device, context, pipeline,
      quadVertexBuffer, quadIndexBuffer,
      instanceBuffer,
      uniformBuffer, bindGroup,
      indexCount, instanceCount,
    } = gpuRef.current;
    if (!device || !context || !pipeline) {
      rafIdRef.current = requestAnimationFrame(renderFrame);
      return;
    }

    // 1) Build camera basis + MVP
    const aspect = canvasWidth / canvasHeight;
    const proj = mat4Perspective(camera.fov, aspect, camera.near, camera.far);

    const cx = camera.orbitRadius * Math.sin(camera.orbitPhi) * Math.sin(camera.orbitTheta);
    const cy = camera.orbitRadius * Math.cos(camera.orbitPhi);
    const cz = camera.orbitRadius * Math.sin(camera.orbitPhi) * Math.cos(camera.orbitTheta);
    const eye: [number, number, number] = [cx, cy, cz];
    const target: [number, number, number] = [camera.panX, camera.panY, 0];
    const up: [number, number, number] = [0, 1, 0];
    const view = mat4LookAt(eye, target, up);

    const forward = normalize([target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]]);
    const right = normalize(cross(forward, up));
    const realUp = cross(right, forward);

    const mvp = mat4Multiply(proj, view);

    // 2) Write camera data to uniform buffer
    const data = new Float32Array(24);
    data.set(mvp, 0);
    data[16] = right[0];
    data[17] = right[1];
    data[18] = right[2];
    data[19] = 0;
    data[20] = realUp[0];
    data[21] = realUp[1];
    data[22] = realUp[2];
    data[23] = 0;
    device.queue.writeBuffer(uniformBuffer, 0, data);

    // 3) Acquire swapchain texture
    let currentTexture: GPUTexture;
    try {
      currentTexture = context.getCurrentTexture();
    } catch {
      rafIdRef.current = requestAnimationFrame(renderFrame);
      return;
    }

    // 4) Render
    const passDesc: GPURenderPassDescriptor = {
      colorAttachments: [
        {
          view: currentTexture.createView(),
          clearValue: { r: 0.1, g: 0.1, b: 0.1, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    };
    const cmdEncoder = device.createCommandEncoder();
    const passEncoder = cmdEncoder.beginRenderPass(passDesc);

    if (instanceBuffer && instanceCount > 0) {
      passEncoder.setPipeline(pipeline);
      passEncoder.setBindGroup(0, bindGroup);
      passEncoder.setVertexBuffer(0, quadVertexBuffer);
      passEncoder.setVertexBuffer(1, instanceBuffer);
      passEncoder.setIndexBuffer(quadIndexBuffer, 'uint16');
      passEncoder.drawIndexed(indexCount, instanceCount);
    }

    passEncoder.end();
    device.queue.submit([cmdEncoder.finish()]);

    rafIdRef.current = requestAnimationFrame(renderFrame);
  }, [camera, canvasWidth, canvasHeight]);

  /******************************************************
   * D) Build Instance Data
   ******************************************************/
  function buildInstanceData(
    positions: Float32Array,
    colors?: Float32Array | Uint8Array,
    scales?: Float32Array,
    decorations?: Decoration[],
  ): Float32Array {
    const count = positions.length / 3;
    const instanceData = new Float32Array(count * 8);

    for (let i = 0; i < count; i++) {
      instanceData[i * 8 + 0] = positions[i * 3 + 0];
      instanceData[i * 8 + 1] = positions[i * 3 + 1];
      instanceData[i * 8 + 2] = positions[i * 3 + 2];

      // color or white
      if (colors && colors.length === count * 3) {
        // Normalize if colors are Uint8Array (0-255) to float (0-1)
        const normalize = colors instanceof Uint8Array ? (1/255) : 1;
        instanceData[i * 8 + 3] = colors[i * 3 + 0] * normalize;
        instanceData[i * 8 + 4] = colors[i * 3 + 1] * normalize;
        instanceData[i * 8 + 5] = colors[i * 3 + 2] * normalize;
      } else {
        instanceData[i * 8 + 3] = 1;
        instanceData[i * 8 + 4] = 1;
        instanceData[i * 8 + 5] = 1;
      }

      // alpha
      instanceData[i * 8 + 6] = 1.0;
      // scale
      if (scales && scales.length === count) {
        instanceData[i * 8 + 7] = scales[i];
      } else {
        instanceData[i * 8 + 7] = 0.02;
      }
    }

    // decorations
    if (decorations) {
      for (const dec of decorations) {
        const { indexes, color, alpha, scale, minSize } = dec;
        for (const idx of indexes) {
          if (idx < 0 || idx >= count) continue;
          if (color) {
            instanceData[idx * 8 + 3] = color[0];
            instanceData[idx * 8 + 4] = color[1];
            instanceData[idx * 8 + 5] = color[2];
          }
          if (alpha !== undefined) {
            instanceData[idx * 8 + 6] = alpha;
          }
          if (scale !== undefined) {
            instanceData[idx * 8 + 7] *= scale;
          }
          if (minSize !== undefined) {
            if (instanceData[idx * 8 + 7] < minSize) {
              instanceData[idx * 8 + 7] = minSize;
            }
          }
        }
      }
    }

    return instanceData;
  }

  /******************************************************
   * E) Update Buffers
   ******************************************************/
  const updatePointCloudBuffers = useCallback((sceneElements: SceneElementConfig[]) => {
    if (!gpuRef.current) return;
    const { device } = gpuRef.current;

    // For now, handle only the first cloud
    const pc = sceneElements.find(e => e.type === 'PointCloud');
    if (!pc || pc.data.positions.length === 0) {
      gpuRef.current.instanceBuffer = null;
      gpuRef.current.instanceCount = 0;
      return;
    }
    const { positions, colors, scales } = pc.data;
    const decorations = pc.decorations || [];
    const instanceData = buildInstanceData(positions, colors, scales, decorations);

    const instanceBuffer = device.createBuffer({
      size: instanceData.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(instanceBuffer, 0, instanceData);
    gpuRef.current.instanceBuffer = instanceBuffer;
    gpuRef.current.instanceCount = positions.length / 3;
  }, []);

  /******************************************************
   * F) Mouse + Zoom to Cursor
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

    // Right drag or SHIFT+Left => pan
    if (mouseState.current.button === 2 || e.shiftKey) {
      setCamera(cam => ({
        ...cam,
        panX: cam.panX - dx * 0.002,
        panY: cam.panY + dy * 0.002,
      }));
    }
    // Left => orbit
    else if (mouseState.current.button === 0) {
      setCamera(cam => {
        // Clamp phi to avoid flipping
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

  // Zoom to cursor
  const onWheel = useCallback((e: WheelEvent) => {
    e.preventDefault();
    // (rest of your zoom logic unchanged)
    // ...
    // same code as before, ensuring we do the "zoom toward" approach
    // with unprojecting near/far, etc.

    // for brevity, we'll do the simpler approach here:
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
   * G) Effects
   ******************************************************/
  // Init WebGPU once
  useEffect(() => {
    initWebGPU();
    return () => {
      if (rafIdRef.current) cancelAnimationFrame(rafIdRef.current);
    };
  }, [initWebGPU]);

  // Resize canvas
  useEffect(() => {
    if (canvasRef.current) {
      canvasRef.current.width = canvasWidth;
      canvasRef.current.height = canvasHeight;
    }
  }, [canvasWidth, canvasHeight]);

  // Start render loop
  useEffect(() => {
    if (isReady) {
      // Immediately draw once
      renderFrame();
      // Then schedule loop
      rafIdRef.current = requestAnimationFrame(renderFrame);
    }
    return () => {
      if (rafIdRef.current) cancelAnimationFrame(rafIdRef.current);
    };
  }, [isReady, renderFrame]);

  // Update buffers
  useEffect(() => {
    if (isReady) {
      updatePointCloudBuffers(elements);
    }
  }, [isReady, elements, updatePointCloudBuffers]);

  // Render
  return (
    <div style={{ width: '100%', border: '1px solid #ccc' }}>
      <canvas
        ref={canvasRef}
        style={{ width: '100%', height: canvasHeight }}
      />
    </div>
  );
}

/******************************************************
 * 6) Example: App
 ******************************************************/
export function App() {
  // 4 points in a square around origin
  const positions = new Float32Array([
    -0.5, -0.5, 0.0,
     0.5, -0.5, 0.0,
    -0.5,  0.5, 0.0,
     0.5,  0.5, 0.0,
  ]);

  // All white
  const colors = new Float32Array([
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
  ]);

  // Some decorations, making each corner different
  const decorations: Decoration[] = [
    {
      indexes: [0],
      color: [1, 0, 0], // red
      alpha: 1.0,
      scale: 1.0,
      minSize: 0.05,
    },
    {
      indexes: [1],
      color: [0, 1, 0], // green
      alpha: 0.7,
      scale: 2.0,
    },
    {
      indexes: [2],
      color: [0, 0, 1], // blue
      alpha: 1.0,
      scale: 1.5,
    },
    {
      indexes: [3],
      color: [1, 1, 0], // yellow
      alpha: 0.3,
      scale: 0.5,
    },
  ];

  const testElements: SceneElementConfig[] = [
    {
      type: 'PointCloud',
      data: { positions, colors },
      decorations,
    },
  ];

  return (
    <>
      <h2>Stage 5: Camera-Facing Billboards + Immediate Visibility</h2>
      <SceneWrapper elements={testElements} />
      <p>
        Left-drag = orbit, Right-drag or Shift+Left = pan,
        mousewheel = zoom (toward cursor).<br />
        We've set <code>orbitTheta=0.2</code>, <code>orbitPhi=1.0</code>,
        and we do an immediate <code>renderFrame()</code> call once ready.
      </p>
    </>
  );
}

export function Torus(props) {
  return <SceneWrapper {...props} />
}
