/// <reference path="./webgpu.d.ts" />

import React, { useRef, useEffect, useState, useCallback } from 'react';
import { useContainerWidth } from '../utils';

/******************************************************
 * 1) Define Types from the Spec
 ******************************************************/
interface PointCloudData {
  positions: Float32Array;     // [x1, y1, z1, x2, y2, z2, ...]
  colors?: Float32Array;       // [r1, g1, b1, r2, g2, b2, ...] (each in [0..1])
  scales?: Float32Array;       // per-point scale factors
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
  pointSize?: number;       // (We won't use this anymore; we rely on scales[] for size)
  decorations?: Decoration[];
}

// For now, we only implement PointCloud
type SceneElementConfig = PointCloudElementConfig;

/******************************************************
 * 2) React Component Props
 ******************************************************/
interface SceneProps {
  elements: SceneElementConfig[];
}

/******************************************************
 * 3) The Scene Component (Stage 4: Scale & Decorations)
 ******************************************************/
export function Scene({ elements }: SceneProps) {
  // --------------------------------------
  // a) Canvas & Container
  // --------------------------------------
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [containerRef, containerWidth] = useContainerWidth(1);
  const canvasHeight = 400;

  // Combine GPU state into a single ref
  const gpuRef = useRef<{
    device: GPUDevice;
    context: GPUCanvasContext;
    pipeline: GPURenderPipeline;
    quadVertexBuffer: GPUBuffer;
    quadIndexBuffer: GPUBuffer;
    instanceBuffer: GPUBuffer | null;
    indexCount: number;       // for the quad
    instanceCount: number;    // number of points
  } | null>(null);

  // Animation handle
  const rafIdRef = useRef<number>(0);

  // Track readiness
  const [isWebGPUReady, setIsWebGPUReady] = useState(false);

  // --------------------------------------
  // b) Create a static "quad" for the billboard
  //    We'll center it at (0,0) so we can
  //    scale/translate it per-point in the shader.
  // --------------------------------------
  const QUAD_VERTICES = new Float32Array([
    //   x,    y
    -0.5, -0.5,   // bottom-left
     0.5, -0.5,   // bottom-right
    -0.5,  0.5,   // top-left
     0.5,  0.5,   // top-right
  ]);
  const QUAD_INDICES = new Uint16Array([
    0, 1, 2,  // first triangle (bottom-left, bottom-right, top-left)
    2, 1, 3   // second triangle (top-left, bottom-right, top-right)
  ]);

  // We'll define the instance layout as:
  //   struct InstanceData {
  //       position : vec3<f32>,
  //       color    : vec3<f32>,
  //       alpha    : f32,
  //       scale    : f32,
  //   };
  // => total 8 floats (32 bytes)

  // --------------------------------------
  // c) Init WebGPU (once)
  // --------------------------------------
  const initWebGPU = useCallback(async () => {
    if (!canvasRef.current) return;
    if (!navigator.gpu) {
      console.error('WebGPU not supported.');
      return;
    }

    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) {
        throw new Error('Failed to get GPU adapter.');
      }
      const device = await adapter.requestDevice();
      const context = canvasRef.current.getContext('webgpu') as GPUCanvasContext;
      const format = navigator.gpu.getPreferredCanvasFormat();

      // Configure
      context.configure({
        device,
        format,
        alphaMode: 'opaque',
      });

      // Create buffers for the quad
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

      // Create pipeline
      const pipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: {
          module: device.createShaderModule({
            code: `
            struct InstanceData {
  position : vec3<f32>,
  color    : vec3<f32>,
  alpha    : f32,
  scale    : f32,
};

@vertex
fn vs_main(
  // Quad corner offset in 2D
  @location(0) corner: vec2<f32>,
  // Instance data
  @location(1) instancePos  : vec3<f32>,
  @location(2) instanceColor: vec3<f32>,
  @location(3) instanceAlpha: f32,
  @location(4) instanceScale: f32
) -> @builtin(position) vec4<f32> {
  // We'll expand the corner by scale, then translate by instancePos
  let scaledCorner = corner * instanceScale;
  let finalPos = vec3<f32>(
    instancePos.x + scaledCorner.x,
    instancePos.y + scaledCorner.y,
    instancePos.z
  );
  return vec4<f32>(finalPos, 1.0);
};

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
            // Buffer(0): The quad's corner offsets (non-instanced)
            {
              arrayStride: 2 * 4, // 2 floats (x,y) * 4 bytes
              attributes: [
                {
                  shaderLocation: 0,
                  offset: 0,
                  format: 'float32x2',
                },
              ],
            },
            // Buffer(1): The per-instance data
            {
              arrayStride: 8 * 4, // 8 floats * 4 bytes
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
              @location(2) inColor : vec3<f32>,
              @location(3) inAlpha : f32
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

  // --------------------------------------
  // d) Render Loop
  // --------------------------------------
  const renderFrame = useCallback(() => {
    if (!gpuRef.current) {
      rafIdRef.current = requestAnimationFrame(renderFrame);
      return;
    }

    const { device, context, pipeline, quadVertexBuffer, quadIndexBuffer, instanceBuffer, indexCount, instanceCount } = gpuRef.current;
    if (!device || !context || !pipeline) {
      rafIdRef.current = requestAnimationFrame(renderFrame);
      return;
    }

    // 1) Get current texture
    const currentTexture = context.getCurrentTexture();
    const renderPassDesc: GPURenderPassDescriptor = {
      colorAttachments: [
        {
          view: currentTexture.createView(),
          clearValue: { r: 0.2, g: 0.2, b: 0.4, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    };

    // 2) Encode
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginRenderPass(renderPassDesc);

    if (instanceBuffer && instanceCount > 0) {
      passEncoder.setPipeline(pipeline);
      // Binding 0 => the quad
      passEncoder.setVertexBuffer(0, quadVertexBuffer);
      // Binding 1 => the instance data
      passEncoder.setVertexBuffer(1, instanceBuffer);
      passEncoder.setIndexBuffer(quadIndexBuffer, 'uint16');
      // Draw 6 indices (2 triangles), with `instanceCount` instances
      passEncoder.drawIndexed(indexCount, instanceCount, 0, 0, 0);
    }

    passEncoder.end();

    // 3) Submit
    const gpuCommands = commandEncoder.finish();
    device.queue.submit([gpuCommands]);

    // 4) Loop
    rafIdRef.current = requestAnimationFrame(renderFrame);
  }, []);

  // --------------------------------------
  // e) Build final arrays (position, color, alpha, scale) + apply decorations
  // --------------------------------------
  function buildInstanceData(
    positions: Float32Array,
    colors: Float32Array | undefined,
    scales: Float32Array | undefined,
    decorations: Decoration[] | undefined
  ): Float32Array {
    const count = positions.length / 3;
    const instanceData = new Float32Array(count * 8);

    // We’ll fill in base data from positions, colors, scales
    // Layout per point i:
    //   0..2: (x,y,z)
    //   3..5: (r,g,b)
    //   6   : alpha
    //   7   : scale
    for (let i = 0; i < count; i++) {
      // position
      instanceData[i * 8 + 0] = positions[i * 3 + 0];
      instanceData[i * 8 + 1] = positions[i * 3 + 1];
      instanceData[i * 8 + 2] = positions[i * 3 + 2];
      // color
      if (colors && colors.length === count * 3) {
        instanceData[i * 8 + 3] = colors[i * 3 + 0];
        instanceData[i * 8 + 4] = colors[i * 3 + 1];
        instanceData[i * 8 + 5] = colors[i * 3 + 2];
      } else {
        // default to white
        instanceData[i * 8 + 3] = 1.0;
        instanceData[i * 8 + 4] = 1.0;
        instanceData[i * 8 + 5] = 1.0;
      }
      // alpha defaults to 1
      instanceData[i * 8 + 6] = 1.0;
      // scale
      if (scales && scales.length === count) {
        instanceData[i * 8 + 7] = scales[i];
      } else {
        // default scale
        instanceData[i * 8 + 7] = 0.02; // arbitrary small
      }
    }

    // Apply any decorations
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
            // multiply the existing scale
            instanceData[idx * 8 + 7] *= scale;
          }
          if (minSize !== undefined) {
            // clamp to minSize if needed
            if (instanceData[idx * 8 + 7] < minSize) {
              instanceData[idx * 8 + 7] = minSize;
            }
          }
        }
      }
    }

    return instanceData;
  }

  // --------------------------------------
  // f) Update instance buffer from the FIRST point cloud
  // --------------------------------------
  const updatePointCloudBuffers = useCallback((sceneElements: SceneElementConfig[]) => {
    if (!gpuRef.current) return;
    const { device } = gpuRef.current;

    // For simplicity, handle only FIRST pointcloud
    const pc = sceneElements.find(e => e.type === 'PointCloud');
    if (!pc || pc.data.positions.length === 0) {
      // no data
      gpuRef.current.instanceBuffer = null;
      gpuRef.current.instanceCount = 0;
      return;
    }

    const { positions, colors, scales } = pc.data;
    const decorations = pc.decorations || [];
    // Build final array
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

  // --------------------------------------
  // g) Effects
  // --------------------------------------
  // 1) Init once
  useEffect(() => {
    initWebGPU();
    return () => {
      if (rafIdRef.current) cancelAnimationFrame(rafIdRef.current);
    };
  }, [initWebGPU]);

  // 2) Resize
  useEffect(() => {
    if (!canvasRef.current) return;
    canvasRef.current.width = containerWidth;
    canvasRef.current.height = canvasHeight;
  }, [containerWidth]);

  // 3) Start render loop
  useEffect(() => {
    if (isWebGPUReady) {
      rafIdRef.current = requestAnimationFrame(renderFrame);
    }
    return () => {
      if (rafIdRef.current) cancelAnimationFrame(rafIdRef.current);
    };
  }, [isWebGPUReady, renderFrame]);

  // 4) Whenever the elements change, re-build instance buffer
  useEffect(() => {
    if (isWebGPUReady) {
      updatePointCloudBuffers(elements);
    }
  }, [isWebGPUReady, elements, updatePointCloudBuffers]);

  // --------------------------------------
  // h) Render
  // --------------------------------------
  return (
    <div ref={containerRef} style={{ width: '100%', border: '1px solid #ccc' }}>
      <canvas ref={canvasRef} style={{ width: '100%', height: canvasHeight }} />
    </div>
  );
}

/******************************************************
 * 4) Example: App Component for Stage 4
 ******************************************************/
export function App() {
  // We'll test:
  //  - 4 points with some "base" scale in scales[]
  //  - Then apply decorations to override color/alpha/scale
  const positions = new Float32Array([
    -0.5, -0.5, 0.0,   // bottom-left
     0.5, -0.5, 0.0,   // bottom-right
    -0.5,  0.5, 0.0,   // top-left
     0.5,  0.5, 0.0,   // top-right
  ]);
  // Base color: all white
  const colors = new Float32Array([
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    1, 1, 1
  ]);
  // Base scale: small
  const scales = new Float32Array([
    0.05, 0.05, 0.05, 0.05
  ]);

  const decorations: Decoration[] = [
    {
      indexes: [0],
      color: [1, 0, 0],  // red
      alpha: 1.0,
      scale: 1.0,
      minSize: 0.05,     // ensures at least 0.05
    },
    {
      indexes: [1],
      color: [0, 1, 0],  // green
      alpha: 0.7,
      scale: 2.0,        // doubles the base scale
    },
    {
      indexes: [2],
      color: [0, 0, 1],  // blue
      alpha: 1.0,
      scale: 1.5,
      minSize: 0.08,     // overrides the final scale if < 0.08
    },
    {
      indexes: [3],
      color: [1, 1, 0],  // yellow
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
      <Scene elements={testElements} />
    </div>
  );
}
