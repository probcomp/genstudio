/// <reference path="./webgpu.d.ts" />

import React, { useRef, useEffect, useState, useCallback } from 'react';
import { useContainerWidth } from '../utils';

/******************************************************
 * 1) Define Types from the Spec
 ******************************************************/
interface PointCloudData {
  positions: Float32Array;     // [x1, y1, z1, x2, y2, z2, ...]
  colors?: Float32Array;       // [r1, g1, b1, r2, g2, b2, ...] (each in [0..1])
  scales?: Float32Array;       // per-point scale factors (optional)
}

interface Decoration {
  indexes: number[];                // indices of points to style
  color?: [number, number, number]; // override color if defined
  alpha?: number;                   // override alpha in [0..1]
  scale?: number;                   // additional scale multiplier
  minSize?: number;                 // minimum size in (clip space) "units"
}

interface PointCloudElementConfig {
  type: 'PointCloud';
  data: PointCloudData;
  pointSize?: number;       // Not directly used in this example
  decorations?: Decoration[];
}

// We only implement PointCloud for now
type SceneElementConfig = PointCloudElementConfig;

/******************************************************
 * 2) React Component Props
 ******************************************************/
interface SceneProps {
  elements: SceneElementConfig[];
  containerWidth: number;
}

/******************************************************
 * SceneWrapper
 *  - Measures container width using useContainerWidth
 *  - Renders <Scene> once we have a non-zero width
 ******************************************************/
export function SceneWrapper({ elements }: { elements: SceneElementConfig[] }) {
  const [containerRef, measuredWidth] = useContainerWidth(1);

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      {measuredWidth > 0 && (
        <Scene
          elements={elements}
          containerWidth={measuredWidth}
        />
      )}
    </div>
  );
}

/******************************************************
 * The Scene Component
 *  - Renders a set of points as billboard quads using WebGPU.
 *  - Demonstrates scale, color, alpha, and decorations.
 ******************************************************/
function Scene({ elements, containerWidth }: SceneProps) {
  // ----------------------------------------------------
  // A) Canvas Setup
  // ----------------------------------------------------
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Ensure canvas dimensions are non-zero
  const safeWidth = containerWidth > 0 ? containerWidth : 300;
  const canvasWidth = safeWidth;
  const canvasHeight = safeWidth;

  // GPU references in a single object
  const gpuRef = useRef<{
    device: GPUDevice;
    context: GPUCanvasContext;
    pipeline: GPURenderPipeline;
    quadVertexBuffer: GPUBuffer;
    quadIndexBuffer: GPUBuffer;
    instanceBuffer: GPUBuffer | null;
    indexCount: number;
    instanceCount: number;
  } | null>(null);

  // Animation handle
  const rafIdRef = useRef<number>(0);

  // Track WebGPU readiness
  const [isWebGPUReady, setIsWebGPUReady] = useState(false);

  // ----------------------------------------------------
  // B) Static Quad Data
  // ----------------------------------------------------
  // A 2D quad, centered at (0,0).
  const QUAD_VERTICES = new Float32Array([
    //   x,    y
    -0.5, -0.5,  // bottom-left
     0.5, -0.5,  // bottom-right
    -0.5,  0.5,  // top-left
     0.5,  0.5,  // top-right
  ]);
  const QUAD_INDICES = new Uint16Array([
    0, 1, 2,
    2, 1, 3
  ]);

  // Each instance has 8 floats:
  //   position (x,y,z) => 3
  //   color    (r,g,b) => 3
  //   alpha              => 1
  //   scale              => 1
  // => total 8 floats (32 bytes)

  // ----------------------------------------------------
  // C) Initialize WebGPU
  // ----------------------------------------------------
  const initWebGPU = useCallback(async () => {
    if (!canvasRef.current) return;
    if (!navigator.gpu) {
      console.error('WebGPU not supported.');
      return;
    }

    try {
      // 1) Request adapter & device
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) {
        throw new Error('Failed to get GPU adapter. Check browser/system support.');
      }
      const device = await adapter.requestDevice();

      // 2) Acquire WebGPU context
      const context = canvasRef.current.getContext('webgpu') as GPUCanvasContext;
      const format = navigator.gpu.getPreferredCanvasFormat();

      // 3) Configure the swap chain
      context.configure({
        device,
        format,
        alphaMode: 'opaque',
      });

      // 4) Create buffers for the static quad
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

      // 5) Create the render pipeline
      //    NOTE: The fragment expects a color at @location(2),
      //          so the vertex must OUTPUT something at location(2).
      const pipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: {
          module: device.createShaderModule({
            code: `
struct VertexOut {
  @builtin(position) position : vec4<f32>,
  @location(2) color : vec3<f32>,
  @location(3) alpha : f32,
};

@vertex
fn vs_main(
  // Quad corner in 2D
  @location(0) corner: vec2<f32>,

  // Per-instance data
  @location(1) instancePos  : vec3<f32>,
  @location(2) instanceColor: vec3<f32>,
  @location(3) instanceAlpha: f32,
  @location(4) instanceScale: f32
) -> VertexOut {
  var out: VertexOut;

  let cornerOffset = corner * instanceScale;
  let finalPos = vec3<f32>(
    instancePos.x + cornerOffset.x,
    instancePos.y + cornerOffset.y,
    instancePos.z
  );

  out.position = vec4<f32>(finalPos, 1.0);
  out.color = instanceColor;
  out.alpha = instanceAlpha;

  return out;
}

@fragment
fn fs_main(
  @location(2) inColor: vec3<f32>,
  @location(3) inAlpha: f32
) -> @location(0) vec4<f32> {
  return vec4<f32>(inColor, inAlpha);
}
            `,
          }),
          entryPoint: 'vs_main',
          buffers: [
            // Buffer(0) => the quad corners
            {
              arrayStride: 2 * 4, // 2 floats * 4 bytes
              attributes: [
                {
                  shaderLocation: 0, // "corner"
                  offset: 0,
                  format: 'float32x2',
                },
              ],
            },
            // Buffer(1) => per-instance data
            {
              arrayStride: 8 * 4, // 8 floats * 4 bytes each
              stepMode: 'instance',
              attributes: [
                // instancePos (3 floats)
                {
                  shaderLocation: 1,
                  offset: 0,
                  format: 'float32x3',
                },
                // instanceColor (3 floats)
                {
                  shaderLocation: 2,
                  offset: 3 * 4,
                  format: 'float32x3',
                },
                // instanceAlpha (1 float)
                {
                  shaderLocation: 3,
                  offset: 6 * 4,
                  format: 'float32',
                },
                // instanceScale (1 float)
                {
                  shaderLocation: 4,
                  offset: 7 * 4,
                  format: 'float32',
                },
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

      gpuRef.current = {
        device,
        context,
        pipeline,
        quadVertexBuffer,
        quadIndexBuffer,
        instanceBuffer: null,
        indexCount: QUAD_INDICES.length,
        instanceCount: 0,
      };

      setIsWebGPUReady(true);
    } catch (err) {
      console.error('Error initializing WebGPU:', err);
    }
  }, []);

  // ----------------------------------------------------
  // D) Render Loop
  // ----------------------------------------------------
  const renderFrame = useCallback(() => {
    if (!gpuRef.current) {
      rafIdRef.current = requestAnimationFrame(renderFrame);
      return;
    }
    const {
      device,
      context,
      pipeline,
      quadVertexBuffer,
      quadIndexBuffer,
      instanceBuffer,
      indexCount,
      instanceCount,
    } = gpuRef.current;

    if (!device || !context || !pipeline) {
      rafIdRef.current = requestAnimationFrame(renderFrame);
      return;
    }

    // Attempt to grab the current swap-chain texture
    let currentTexture: GPUTexture;
    try {
      currentTexture = context.getCurrentTexture();
    } catch {
      // If canvas is size 0 or something else is amiss, try again next frame
      rafIdRef.current = requestAnimationFrame(renderFrame);
      return;
    }

    const renderPassDesc: GPURenderPassDescriptor = {
      colorAttachments: [
        {
          view: currentTexture.createView(),
          clearValue: { r: 0.1, g: 0.1, b: 0.1, a: 1.0 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    };

    // Encode commands
    const cmdEncoder = device.createCommandEncoder();
    const passEncoder = cmdEncoder.beginRenderPass(renderPassDesc);

    if (instanceBuffer && instanceCount > 0) {
      passEncoder.setPipeline(pipeline);
      passEncoder.setVertexBuffer(0, quadVertexBuffer);
      passEncoder.setVertexBuffer(1, instanceBuffer);
      passEncoder.setIndexBuffer(quadIndexBuffer, 'uint16');
      passEncoder.drawIndexed(indexCount, instanceCount);
    }

    passEncoder.end();
    device.queue.submit([cmdEncoder.finish()]);

    // Loop
    rafIdRef.current = requestAnimationFrame(renderFrame);
  }, []);

  // ----------------------------------------------------
  // E) buildInstanceData
  // ----------------------------------------------------
  function buildInstanceData(
    positions: Float32Array,
    colors?: Float32Array,
    scales?: Float32Array,
    decorations?: Decoration[]
  ): Float32Array {
    const count = positions.length / 3; // each point is (x,y,z)
    const instanceData = new Float32Array(count * 8);

    // Base fill: position, color, alpha=1, scale=0.02 (if missing)
    for (let i = 0; i < count; i++) {
      instanceData[i * 8 + 0] = positions[i * 3 + 0];
      instanceData[i * 8 + 1] = positions[i * 3 + 1];
      instanceData[i * 8 + 2] = positions[i * 3 + 2];

      if (colors && colors.length === count * 3) {
        instanceData[i * 8 + 3] = colors[i * 3 + 0];
        instanceData[i * 8 + 4] = colors[i * 3 + 1];
        instanceData[i * 8 + 5] = colors[i * 3 + 2];
      } else {
        instanceData[i * 8 + 3] = 1.0;
        instanceData[i * 8 + 4] = 1.0;
        instanceData[i * 8 + 5] = 1.0;
      }

      instanceData[i * 8 + 6] = 1.0; // alpha
      if (scales && scales.length === count) {
        instanceData[i * 8 + 7] = scales[i];
      } else {
        instanceData[i * 8 + 7] = 0.02;
      }
    }

    // Apply decorations
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
            const currentScale = instanceData[idx * 8 + 7];
            if (currentScale < minSize) {
              instanceData[idx * 8 + 7] = minSize;
            }
          }
        }
      }
    }

    return instanceData;
  }

  // ----------------------------------------------------
  // F) Update instance buffer from the FIRST PointCloud
  // ----------------------------------------------------
  const updatePointCloudBuffers = useCallback((sceneElements: SceneElementConfig[]) => {
    if (!gpuRef.current) return;
    const { device } = gpuRef.current;

    // For simplicity, handle only the FIRST PointCloud
    const pc = sceneElements.find(e => e.type === 'PointCloud');
    if (!pc || pc.data.positions.length === 0) {
      gpuRef.current.instanceBuffer = null;
      gpuRef.current.instanceCount = 0;
      return;
    }

    const { positions, colors, scales } = pc.data;
    const decorations = pc.decorations || [];

    // Build final data
    const instanceData = buildInstanceData(positions, colors, scales, decorations);

    const instanceBuffer = device.createBuffer({
      size: instanceData.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });

    device.queue.writeBuffer(
      instanceBuffer,
      0,
      instanceData.buffer,
      instanceData.byteOffset,
      instanceData.byteLength
    );

    gpuRef.current.instanceBuffer = instanceBuffer;
    gpuRef.current.instanceCount = positions.length / 3;
  }, []);

  // ----------------------------------------------------
  // G) Effects
  // ----------------------------------------------------
  // 1) Initialize WebGPU once
  useEffect(() => {
    initWebGPU();
    return () => {
      if (rafIdRef.current) cancelAnimationFrame(rafIdRef.current);
    };
  }, [initWebGPU]);

  // 2) Resize the canvas
  useEffect(() => {
    if (!canvasRef.current) return;
    canvasRef.current.width = canvasWidth;
    canvasRef.current.height = canvasHeight;
  }, [canvasWidth, canvasHeight]);

  // 3) Start render loop
  useEffect(() => {
    if (isWebGPUReady) {
      rafIdRef.current = requestAnimationFrame(renderFrame);
    }
    return () => {
      if (rafIdRef.current) cancelAnimationFrame(rafIdRef.current);
    };
  }, [isWebGPUReady, renderFrame]);

  // 4) Rebuild instance buffer when elements change
  useEffect(() => {
    if (isWebGPUReady) {
      updatePointCloudBuffers(elements);
    }
  }, [isWebGPUReady, elements, updatePointCloudBuffers]);

  // ----------------------------------------------------
  // H) Render
  // ----------------------------------------------------
  return (
    <div style={{ width: '100%', border: '1px solid #ccc' }}>
      <canvas
        ref={canvasRef}
        style={{
          width: '100%',
          height: canvasHeight,
          display: 'block',
        }}
      />
    </div>
  );
}

/******************************************************
 * Example: App Component
 ******************************************************/
export function App() {
  // 4 points in clip space
  const positions = new Float32Array([
    -0.5, -0.5, 0.0,
     0.5, -0.5, 0.0,
    -0.5,  0.5, 0.0,
     0.5,  0.5, 0.0,
  ]);

  // All white by default
  const colors = new Float32Array([
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
  ]);

  // Base scale
  const scales = new Float32Array([0.05, 0.05, 0.05, 0.05]);

  // Simple decorations
  const decorations: Decoration[] = [
    {
      indexes: [0],
      color: [1, 0, 0],   // red
      alpha: 1.0,
      scale: 1.0,
      minSize: 0.05,
    },
    {
      indexes: [1],
      color: [0, 1, 0],   // green
      alpha: 0.7,
      scale: 2.0,
    },
    {
      indexes: [2],
      color: [0, 0, 1],   // blue
      alpha: 1.0,
      scale: 1.5,
      minSize: 0.08,
    },
    {
      indexes: [3],
      color: [1, 1, 0],   // yellow
      alpha: 0.3,
      scale: 0.5,
    },
  ];

  const testElements: SceneElementConfig[] = [
    {
      type: 'PointCloud',
      data: { positions, colors, scales },
      decorations,
    },
  ];

  return (
    <div style={{ width: '600px' }}>
      <h2>Stage 4: Scale & Decorations (Billboard Quads)</h2>
      <SceneWrapper elements={testElements} />
    </div>
  );
}
