/// <reference path="./webgpu.d.ts" />
/// <reference types="react" />

import React, {
  useRef, useEffect, useState, useCallback, MouseEvent as ReactMouseEvent, useMemo
} from 'react';
import { useContainerWidth } from '../utils';

/******************************************************
 * 0) Rendering + Lighting Constants
 ******************************************************/
const LIGHTING = {
  AMBIENT_INTENSITY: 0.4,
  DIFFUSE_INTENSITY: 0.6,
  SPECULAR_INTENSITY: 0.2,
  SPECULAR_POWER: 20.0,
  DIRECTION: {
    RIGHT: 0.2,
    UP: 0.5,
    FORWARD: 0,
  }
} as const;

// Add the default camera configuration
const DEFAULT_CAMERA = {
  orbitRadius: 1.5,
  orbitTheta: 0.2,
  orbitPhi: 1.0,
  panX: 0,
  panY: 0,
  fov: Math.PI/3,
  near: 0.01,
  far: 100.0
} as const;

/******************************************************
 * 1) Data Structures & "Spec" System
 ******************************************************/
interface PrimitiveSpec<E> {
  getCount(element: E): number;
  buildRenderData(element: E): Float32Array | null;
  buildPickingData(element: E, baseID: number): Float32Array | null;
}

interface Decoration {
  indexes: number[];
  color?: [number, number, number];
  alpha?: number;
  scale?: number;
  minSize?: number;
}

/** ===================== POINT CLOUD ===================== **/
interface PointCloudData {
  positions: Float32Array;
  colors?: Float32Array;
  scales?: Float32Array;
}
export interface PointCloudElementConfig {
  type: 'PointCloud';
  data: PointCloudData;
  decorations?: Decoration[];
  onHover?: (index: number|null) => void;
  onClick?: (index: number) => void;
}

const pointCloudSpec: PrimitiveSpec<PointCloudElementConfig> = {
  getCount(elem){
    return elem.data.positions.length / 3;
  },
  buildRenderData(elem) {
    const { positions, colors, scales } = elem.data;
    const count = positions.length / 3;
    if(count === 0) return null;

    // (pos.x, pos.y, pos.z, color.r, color.g, color.b, alpha, scaleX, scaleY)
    const arr = new Float32Array(count * 9);
    for(let i=0; i<count; i++){
      arr[i*9+0] = positions[i*3+0];
      arr[i*9+1] = positions[i*3+1];
      arr[i*9+2] = positions[i*3+2];
      if(colors && colors.length === count*3){
        arr[i*9+3] = colors[i*3+0];
        arr[i*9+4] = colors[i*3+1];
        arr[i*9+5] = colors[i*3+2];
      } else {
        arr[i*9+3] = 1;
        arr[i*9+4] = 1;
        arr[i*9+5] = 1;
      }
      arr[i*9+6] = 1.0; // alpha
      const s = scales ? scales[i] : 0.02;
      arr[i*9+7] = s;
      arr[i*9+8] = s;
    }
    // decorations
    if(elem.decorations){
      for(const dec of elem.decorations){
        for(const idx of dec.indexes){
          if(idx<0||idx>=count) continue;
          if(dec.color){
            arr[idx*9+3] = dec.color[0];
            arr[idx*9+4] = dec.color[1];
            arr[idx*9+5] = dec.color[2];
          }
          if(dec.alpha !== undefined){
            arr[idx*9+6] = dec.alpha;
          }
          if(dec.scale !== undefined){
            arr[idx*9+7] *= dec.scale;
            arr[idx*9+8] *= dec.scale;
          }
          if(dec.minSize !== undefined){
            if(arr[idx*9+7] < dec.minSize) arr[idx*9+7] = dec.minSize;
            if(arr[idx*9+8] < dec.minSize) arr[idx*9+8] = dec.minSize;
          }
        }
      }
    }
    return arr;
  },
  buildPickingData(elem, baseID){
    const { positions, scales } = elem.data;
    const count = positions.length / 3;
    if(count===0) return null;

    // (pos.x, pos.y, pos.z, pickID, scaleX, scaleY)
    const arr = new Float32Array(count*6);
    for(let i=0; i<count; i++){
      arr[i*6+0] = positions[i*3+0];
      arr[i*6+1] = positions[i*3+1];
      arr[i*6+2] = positions[i*3+2];
      arr[i*6+3] = baseID + i;
      const s = scales ? scales[i] : 0.02;
      arr[i*6+4] = s;
      arr[i*6+5] = s;
    }
    return arr;
  }
};

/** ===================== ELLIPSOID ===================== **/
interface EllipsoidData {
  centers: Float32Array;
  radii: Float32Array;
  colors?: Float32Array;
}
export interface EllipsoidElementConfig {
  type: 'Ellipsoid';
  data: EllipsoidData;
  decorations?: Decoration[];
  onHover?: (index: number|null) => void;
  onClick?: (index: number) => void;
}

const ellipsoidSpec: PrimitiveSpec<EllipsoidElementConfig> = {
  getCount(elem){
    return elem.data.centers.length / 3;
  },
  buildRenderData(elem){
    const { centers, radii, colors } = elem.data;
    const count = centers.length / 3;
    if(count===0)return null;

    // (pos.x, pos.y, pos.z, scale.x, scale.y, scale.z, col.r, col.g, col.b, alpha)
    const arr = new Float32Array(count*10);
    for(let i=0; i<count; i++){
      arr[i*10+0] = centers[i*3+0];
      arr[i*10+1] = centers[i*3+1];
      arr[i*10+2] = centers[i*3+2];
      arr[i*10+3] = radii[i*3+0] || 0.1;
      arr[i*10+4] = radii[i*3+1] || 0.1;
      arr[i*10+5] = radii[i*3+2] || 0.1;
      if(colors && colors.length===count*3){
        arr[i*10+6] = colors[i*3+0];
        arr[i*10+7] = colors[i*3+1];
        arr[i*10+8] = colors[i*3+2];
      } else {
        arr[i*10+6] = 1; arr[i*10+7] = 1; arr[i*10+8] = 1;
      }
      arr[i*10+9] = 1.0;
    }
    // decorations
    if(elem.decorations){
      for(const dec of elem.decorations){
        for(const idx of dec.indexes){
          if(idx<0||idx>=count) continue;
          if(dec.color){
            arr[idx*10+6] = dec.color[0];
            arr[idx*10+7] = dec.color[1];
            arr[idx*10+8] = dec.color[2];
          }
          if(dec.alpha!==undefined){
            arr[idx*10+9] = dec.alpha;
          }
          if(dec.scale!==undefined){
            arr[idx*10+3]*=dec.scale;
            arr[idx*10+4]*=dec.scale;
            arr[idx*10+5]*=dec.scale;
          }
          if(dec.minSize!==undefined){
            if(arr[idx*10+3]<dec.minSize) arr[idx*10+3] = dec.minSize;
            if(arr[idx*10+4]<dec.minSize) arr[idx*10+4] = dec.minSize;
            if(arr[idx*10+5]<dec.minSize) arr[idx*10+5] = dec.minSize;
          }
        }
      }
    }
    return arr;
  },
  buildPickingData(elem, baseID){
    const { centers, radii } = elem.data;
    const count=centers.length/3;
    if(count===0)return null;

    // (pos.x, pos.y, pos.z, scale.x, scale.y, scale.z, pickID)
    const arr = new Float32Array(count*7);
    for(let i=0; i<count; i++){
      arr[i*7+0] = centers[i*3+0];
      arr[i*7+1] = centers[i*3+1];
      arr[i*7+2] = centers[i*3+2];
      arr[i*7+3] = radii[i*3+0]||0.1;
      arr[i*7+4] = radii[i*3+1]||0.1;
      arr[i*7+5] = radii[i*3+2]||0.1;
      arr[i*7+6] = baseID + i;
    }
    return arr;
  }
};

/** ===================== ELLIPSOID BOUNDS ===================== **/
export interface EllipsoidBoundsElementConfig {
  type: 'EllipsoidBounds';
  data: EllipsoidData;
  decorations?: Decoration[];
  onHover?: (index: number|null) => void;
  onClick?: (index: number) => void;
}

const ellipsoidBoundsSpec: PrimitiveSpec<EllipsoidBoundsElementConfig> = {
  getCount(elem) {
    // each ellipsoid => 3 rings
    const c = elem.data.centers.length/3;
    return c*3;
  },
  buildRenderData(elem) {
    const { centers, radii, colors } = elem.data;
    const count = centers.length/3;
    if(count===0)return null;

    const ringCount = count*3;
    // (pos.x,pos.y,pos.z, scale.x,scale.y,scale.z, color.r,g,b, alpha)
    const arr = new Float32Array(ringCount*10);
    for(let i=0;i<count;i++){
      const cx=centers[i*3+0], cy=centers[i*3+1], cz=centers[i*3+2];
      const rx=radii[i*3+0]||0.1, ry=radii[i*3+1]||0.1, rz=radii[i*3+2]||0.1;
      let cr=1, cg=1, cb=1, alpha=1;
      if(colors && colors.length===count*3){
        cr=colors[i*3+0]; cg=colors[i*3+1]; cb=colors[i*3+2];
      }
      // decorations
      if(elem.decorations){
        for(const dec of elem.decorations){
          if(dec.indexes.includes(i)){
            if(dec.color){
              cr=dec.color[0]; cg=dec.color[1]; cb=dec.color[2];
            }
            if(dec.alpha!==undefined){
              alpha=dec.alpha;
            }
          }
        }
      }
      // fill 3 rings
      for(let ring=0; ring<3; ring++){
        const idx = i*3 + ring;
        arr[idx*10+0] = cx;
        arr[idx*10+1] = cy;
        arr[idx*10+2] = cz;
        arr[idx*10+3] = rx; arr[idx*10+4] = ry; arr[idx*10+5] = rz;
        arr[idx*10+6] = cr; arr[idx*10+7] = cg; arr[idx*10+8] = cb;
        arr[idx*10+9] = alpha;
      }
    }
    return arr;
  },
  buildPickingData(elem, baseID){
    const { centers, radii } = elem.data;
    const count=centers.length/3;
    if(count===0)return null;
    const ringCount = count*3;
    const arr = new Float32Array(ringCount*7);
    for(let i=0;i<count;i++){
      const cx=centers[i*3+0], cy=centers[i*3+1], cz=centers[i*3+2];
      const rx=radii[i*3+0]||0.1, ry=radii[i*3+1]||0.1, rz=radii[i*3+2]||0.1;
      const thisID = baseID + i;
      for(let ring=0; ring<3; ring++){
        const idx = i*3 + ring;
        arr[idx*7+0] = cx;
        arr[idx*7+1] = cy;
        arr[idx*7+2] = cz;
        arr[idx*7+3] = rx; arr[idx*7+4] = ry; arr[idx*7+5] = rz;
        arr[idx*7+6] = thisID;
      }
    }
    return arr;
  }
};

/** ===================== CUBOID ===================== **/
interface CuboidData {
  centers: Float32Array;
  sizes: Float32Array;
  colors?: Float32Array;
}
export interface CuboidElementConfig {
  type: 'Cuboid';
  data: CuboidData;
  decorations?: Decoration[];
  onHover?: (index: number|null) => void;
  onClick?: (index: number) => void;
}

const cuboidSpec: PrimitiveSpec<CuboidElementConfig> = {
  getCount(elem){
    return elem.data.centers.length / 3;
  },
  buildRenderData(elem){
    const { centers, sizes, colors } = elem.data;
    const count = centers.length / 3;
    if(count===0)return null;

    // (center.x, center.y, center.z, size.x, size.y, size.z, color.r, color.g, color.b, alpha)
    const arr = new Float32Array(count*10);
    for(let i=0; i<count; i++){
      arr[i*10+0] = centers[i*3+0];
      arr[i*10+1] = centers[i*3+1];
      arr[i*10+2] = centers[i*3+2];
      arr[i*10+3] = sizes[i*3+0] || 0.1;
      arr[i*10+4] = sizes[i*3+1] || 0.1;
      arr[i*10+5] = sizes[i*3+2] || 0.1;
      if(colors && colors.length===count*3){
        arr[i*10+6] = colors[i*3+0];
        arr[i*10+7] = colors[i*3+1];
        arr[i*10+8] = colors[i*3+2];
      } else {
        arr[i*10+6] = 1; arr[i*10+7] = 1; arr[i*10+8] = 1;
      }
      arr[i*10+9] = 1.0;
    }
    // decorations
    if(elem.decorations){
      for(const dec of elem.decorations){
        for(const idx of dec.indexes){
          if(idx<0||idx>=count) continue;
          if(dec.color){
            arr[idx*10+6] = dec.color[0];
            arr[idx*10+7] = dec.color[1];
            arr[idx*10+8] = dec.color[2];
          }
          if(dec.alpha!==undefined){
            arr[idx*10+9] = dec.alpha;
          }
          if(dec.scale!==undefined){
            arr[idx*10+3]*=dec.scale;
            arr[idx*10+4]*=dec.scale;
            arr[idx*10+5]*=dec.scale;
          }
          if(dec.minSize!==undefined){
            if(arr[idx*10+3]<dec.minSize) arr[idx*10+3] = dec.minSize;
            if(arr[idx*10+4]<dec.minSize) arr[idx*10+4] = dec.minSize;
            if(arr[idx*10+5]<dec.minSize) arr[idx*10+5] = dec.minSize;
          }
        }
      }
    }
    return arr;
  },
  buildPickingData(elem, baseID){
    const { centers, sizes } = elem.data;
    const count=centers.length/3;
    if(count===0)return null;

    // (center.x, center.y, center.z, size.x, size.y, size.z, pickID)
    const arr = new Float32Array(count*7);
    for(let i=0; i<count; i++){
      arr[i*7+0] = centers[i*3+0];
      arr[i*7+1] = centers[i*3+1];
      arr[i*7+2] = centers[i*3+2];
      arr[i*7+3] = sizes[i*3+0]||0.1;
      arr[i*7+4] = sizes[i*3+1]||0.1;
      arr[i*7+5] = sizes[i*3+2]||0.1;
      arr[i*7+6] = baseID + i;
    }
    return arr;
  }
};

/******************************************************
 * 1.5) Extended Spec: also handle pipeline & render objects
 ******************************************************/
interface RenderObject {
  pipeline: GPURenderPipeline;
  vertexBuffers: GPUBuffer[];
  indexBuffer?: GPUBuffer;
  vertexCount?: number;
  indexCount?: number;
  instanceCount?: number;

  pickingPipeline: GPURenderPipeline;
  pickingVertexBuffers: GPUBuffer[];
  pickingIndexBuffer?: GPUBuffer;
  pickingVertexCount?: number;
  pickingIndexCount?: number;
  pickingInstanceCount?: number;

  elementIndex: number;
}

interface ExtendedSpec<E> extends PrimitiveSpec<E> {
  /** Return (or lazily create) the GPU render pipeline for this type. */
  getRenderPipeline(
    device: GPUDevice,
    bindGroupLayout: GPUBindGroupLayout,
    cache: Map<string, PipelineCacheEntry>
  ): GPURenderPipeline;

  /** Return (or lazily create) the GPU picking pipeline for this type. */
  getPickingPipeline(
    device: GPUDevice,
    bindGroupLayout: GPUBindGroupLayout,
    cache: Map<string, PipelineCacheEntry>
  ): GPURenderPipeline;

  /** Create a RenderObject that references geometry + the two pipelines. */
  createRenderObject(
    device: GPUDevice,
    pipeline: GPURenderPipeline,
    pickingPipeline: GPURenderPipeline,
    instanceVB: GPUBuffer|null,
    pickingVB: GPUBuffer|null,
    instanceCount: number,
    resources: {  // Add this parameter
      sphereGeo: { vb: GPUBuffer; ib: GPUBuffer; indexCount: number } | null;
      ringGeo: { vb: GPUBuffer; ib: GPUBuffer; indexCount: number } | null;
      billboardQuad: { vb: GPUBuffer; ib: GPUBuffer; } | null;
      cubeGeo: { vb: GPUBuffer; ib: GPUBuffer; indexCount: number } | null;
    }
  ): RenderObject;
}

/******************************************************
 * 1.6) Pipeline Cache Helper
 ******************************************************/
// Update the pipeline cache to include device reference
interface PipelineCacheEntry {
  pipeline: GPURenderPipeline;
  device: GPUDevice;
}

// Update the cache type
const pipelineCache = new Map<string, PipelineCacheEntry>();

// Update the getOrCreatePipeline helper
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
 * 2) Common Resources: geometry, layout, etc.
 ******************************************************/
// Replace CommonResources.init with a function that initializes for a specific instance
function initGeometryResources(device: GPUDevice, resources: typeof gpuRef.current.resources) {
  // Create sphere geometry
  if(!resources.sphereGeo) {
    const { vertexData, indexData } = createSphereGeometry(16,24);
    const vb = device.createBuffer({
      size: vertexData.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(vb,0,vertexData);
    const ib = device.createBuffer({
      size: indexData.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(ib,0,indexData);
    resources.sphereGeo = { vb, ib, indexCount: indexData.length };
  }

  // Create ring geometry
  if(!resources.ringGeo) {
    const { vertexData, indexData } = createTorusGeometry(1.0,0.03,40,12);
    const vb = device.createBuffer({
      size: vertexData.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(vb,0,vertexData);
    const ib = device.createBuffer({
      size: indexData.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(ib,0,indexData);
    resources.ringGeo = { vb, ib, indexCount: indexData.length };
  }

  // Create billboard quad geometry
  if(!resources.billboardQuad) {
    const quadVerts = new Float32Array([-0.5,-0.5, 0.5,-0.5, -0.5,0.5, 0.5,0.5]);
    const quadIdx = new Uint16Array([0,1,2, 2,1,3]);
    const vb = device.createBuffer({
      size: quadVerts.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(vb,0,quadVerts);
    const ib = device.createBuffer({
      size: quadIdx.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(ib,0,quadIdx);
    resources.billboardQuad = { vb, ib };
  }

  // Create cube geometry
  if(!resources.cubeGeo) {
    const cube = createCubeGeometry();
    const vb = device.createBuffer({
      size: cube.vertexData.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(vb, 0, cube.vertexData);
    const ib = device.createBuffer({
      size: cube.indexData.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(ib, 0, cube.indexData);
    resources.cubeGeo = { vb, ib, indexCount: cube.indexData.length };
  }
}

/******************************************************
 * 2.1) PointCloud ExtendedSpec
 ******************************************************/
const pointCloudExtendedSpec: ExtendedSpec<PointCloudElementConfig> = {
  ...pointCloudSpec,

  getRenderPipeline(device, bindGroupLayout, cache) {
    const format = navigator.gpu.getPreferredCanvasFormat();
    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout]
    });

    return getOrCreatePipeline(
      device,
      "PointCloudShading",
      () => {
        return device.createRenderPipeline({
          layout: pipelineLayout,
          vertex:{
            module: device.createShaderModule({ code: billboardVertCode }),
            entryPoint:'vs_main',
            buffers:[
              // billboard corners
              {
                arrayStride: 8,
                attributes:[{shaderLocation:0, offset:0, format:'float32x2'}]
              },
              // instance data
              {
                arrayStride: 9*4,
                stepMode:'instance',
                attributes:[
                  {shaderLocation:1, offset:0,    format:'float32x3'},
                  {shaderLocation:2, offset:3*4,  format:'float32x3'},
                  {shaderLocation:3, offset:6*4,  format:'float32'},
                  {shaderLocation:4, offset:7*4,  format:'float32'},
                  {shaderLocation:5, offset:8*4,  format:'float32'}
                ]
              }
            ]
          },
          fragment:{
            module: device.createShaderModule({ code: billboardFragCode }),
            entryPoint:'fs_main',
            targets:[{
              format,
              blend:{
                color:{srcFactor:'src-alpha', dstFactor:'one-minus-src-alpha'},
                alpha:{srcFactor:'one', dstFactor:'one-minus-src-alpha'}
              }
            }]
          },
          primitive:{ topology:'triangle-list', cullMode:'back' },
          depthStencil:{ format:'depth24plus', depthWriteEnabled:true, depthCompare:'less-equal'}
        });
      },
      cache
    );
  },

  getPickingPipeline(device, bindGroupLayout, cache) {
    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout]
    });
    return getOrCreatePipeline(
      device,
      "PointCloudPicking",
      () => {
        return device.createRenderPipeline({
          layout: pipelineLayout,
          vertex:{
            module: device.createShaderModule({ code: pickingVertCode }),
            entryPoint: 'vs_pointcloud',
            buffers:[
              // corners
              {
                arrayStride: 8,
                attributes:[{shaderLocation:0, offset:0, format:'float32x2'}]
              },
              // instance data
              {
                arrayStride: 6*4,
                stepMode:'instance',
                attributes:[
                  {shaderLocation:1, offset:0,   format:'float32x3'},
                  {shaderLocation:2, offset:3*4, format:'float32'},  // pickID
                  {shaderLocation:3, offset:4*4, format:'float32'},  // scaleX
                  {shaderLocation:4, offset:5*4, format:'float32'}   // scaleY
                ]
              }
            ]
          },
          fragment:{
            module: device.createShaderModule({ code: pickingVertCode }),
            entryPoint:'fs_pick',
            targets:[{ format:'rgba8unorm' }]
          },
          primitive:{ topology:'triangle-list', cullMode:'back'},
          depthStencil:{ format:'depth24plus', depthWriteEnabled:true, depthCompare:'less-equal'}
        });
      },
      cache
    );
  },

  createRenderObject(device, pipeline, pickingPipeline, instanceVB, pickingInstanceVB, count, resources){
    if(!resources.billboardQuad){
      throw new Error("No billboard geometry available (not yet initialized).");
    }
    const { vb, ib } = resources.billboardQuad;
    return {
      pipeline,
      vertexBuffers: [vb, instanceVB!],
      indexBuffer: ib,
      indexCount: 6,
      instanceCount: count,

      pickingPipeline,
      pickingVertexBuffers: [vb, pickingInstanceVB!],
      pickingIndexBuffer: ib,
      pickingIndexCount: 6,
      pickingInstanceCount: count,

      elementIndex: -1
    };
  }
};

/******************************************************
 * 2.2) Ellipsoid ExtendedSpec
 ******************************************************/
const ellipsoidExtendedSpec: ExtendedSpec<EllipsoidElementConfig> = {
  ...ellipsoidSpec,

  getRenderPipeline(device, bindGroupLayout, cache) {
    const format = navigator.gpu.getPreferredCanvasFormat();
    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout]
    });
    return getOrCreatePipeline(
      device,
      "EllipsoidShading",
      () => {
        return device.createRenderPipeline({
          layout: pipelineLayout,
          vertex:{
            module: device.createShaderModule({ code: ellipsoidVertCode }),
            entryPoint:'vs_main',
            buffers:[
              // sphere geometry
              {
                arrayStride: 6*4,
                attributes:[
                  {shaderLocation:0, offset:0,   format:'float32x3'},
                  {shaderLocation:1, offset:3*4, format:'float32x3'}
                ]
              },
              // instance data
              {
                arrayStride: 10*4,
                stepMode:'instance',
                attributes:[
                  {shaderLocation:2, offset:0,     format:'float32x3'},
                  {shaderLocation:3, offset:3*4,   format:'float32x3'},
                  {shaderLocation:4, offset:6*4,   format:'float32x3'},
                  {shaderLocation:5, offset:9*4,   format:'float32'}
                ]
              }
            ]
          },
          fragment:{
            module: device.createShaderModule({ code: ellipsoidFragCode }),
            entryPoint:'fs_main',
            targets:[{
              format,
              blend:{
                color:{srcFactor:'src-alpha', dstFactor:'one-minus-src-alpha'},
                alpha:{srcFactor:'one', dstFactor:'one-minus-src-alpha'}
              }
            }]
          },
          primitive:{ topology:'triangle-list', cullMode:'back'},
          depthStencil:{format:'depth24plus', depthWriteEnabled:true, depthCompare:'less-equal'}
        });
      },
      cache
    );
  },

  getPickingPipeline(device, bindGroupLayout, cache) {
    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout]
    });
    return getOrCreatePipeline(
      device,
      "EllipsoidPicking",
      () => {
        return device.createRenderPipeline({
          layout: pipelineLayout,
          vertex:{
            module: device.createShaderModule({ code: pickingVertCode }),
            entryPoint:'vs_ellipsoid',
            buffers:[
              // sphere geometry
              {
                arrayStride:6*4,
                attributes:[
                  {shaderLocation:0, offset:0,   format:'float32x3'},
                  {shaderLocation:1, offset:3*4, format:'float32x3'}
                ]
              },
              // instance data
              {
                arrayStride:7*4,
                stepMode:'instance',
                attributes:[
                  {shaderLocation:2, offset:0,   format:'float32x3'},
                  {shaderLocation:3, offset:3*4, format:'float32x3'},
                  {shaderLocation:4, offset:6*4, format:'float32'}
                ]
              }
            ]
          },
          fragment:{
            module: device.createShaderModule({ code: pickingVertCode }),
            entryPoint:'fs_pick',
            targets:[{ format:'rgba8unorm' }]
          },
          primitive:{ topology:'triangle-list', cullMode:'back'},
          depthStencil:{ format:'depth24plus', depthWriteEnabled:true, depthCompare:'less-equal'}
        });
      },
      cache
    );
  },

  createRenderObject(device, pipeline, pickingPipeline, instanceVB, pickingInstanceVB, count, resources){
    if(!resources.sphereGeo) {
      throw new Error("No sphere geometry available (not yet initialized).");
    }
    const { vb, ib, indexCount } = resources.sphereGeo;
    return {
      pipeline,
      vertexBuffers: [vb, instanceVB!],
      indexBuffer: ib,
      indexCount,
      instanceCount: count,

      pickingPipeline,
      pickingVertexBuffers: [vb, pickingInstanceVB!],
      pickingIndexBuffer: ib,
      pickingIndexCount: indexCount,
      pickingInstanceCount: count,

      elementIndex: -1
    };
  }
};

/******************************************************
 * 2.3) EllipsoidBounds ExtendedSpec
 ******************************************************/
const ellipsoidBoundsExtendedSpec: ExtendedSpec<EllipsoidBoundsElementConfig> = {
  ...ellipsoidBoundsSpec,

  getRenderPipeline(device, bindGroupLayout, cache) {
    const format = navigator.gpu.getPreferredCanvasFormat();
    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout]
    });
    return getOrCreatePipeline(
      device,
      "EllipsoidBoundsShading",
      () => {
        return device.createRenderPipeline({
          layout: pipelineLayout,
          vertex:{
            module: device.createShaderModule({ code: ringVertCode }),
            entryPoint:'vs_main',
            buffers:[
              // ring geometry
              {
                arrayStride:6*4,
                attributes:[
                  {shaderLocation:0, offset:0,   format:'float32x3'},
                  {shaderLocation:1, offset:3*4, format:'float32x3'}
                ]
              },
              // instance data
              {
                arrayStride: 10*4,
                stepMode:'instance',
                attributes:[
                  {shaderLocation:2, offset:0,     format:'float32x3'},
                  {shaderLocation:3, offset:3*4,   format:'float32x3'},
                  {shaderLocation:4, offset:6*4,   format:'float32x3'},
                  {shaderLocation:5, offset:9*4,   format:'float32'}
                ]
              }
            ]
          },
          fragment:{
            module: device.createShaderModule({ code: ringFragCode }),
            entryPoint:'fs_main',
            targets:[{
              format,
              blend:{
                color:{srcFactor:'src-alpha', dstFactor:'one-minus-src-alpha'},
                alpha:{srcFactor:'one', dstFactor:'one-minus-src-alpha'}
              }
            }]
          },
          primitive:{ topology:'triangle-list', cullMode:'back'},
          depthStencil:{format:'depth24plus', depthWriteEnabled:true, depthCompare:'less-equal'}
        });
      },
      cache
    );
  },

  getPickingPipeline(device, bindGroupLayout, cache) {
    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout]
    });
    return getOrCreatePipeline(
      device,
      "EllipsoidBoundsPicking",
      () => {
        return device.createRenderPipeline({
          layout: pipelineLayout,
          vertex:{
            module: device.createShaderModule({ code: pickingVertCode }),
            entryPoint:'vs_bands',
            buffers:[
              // ring geometry
              {
                arrayStride:6*4,
                attributes:[
                  {shaderLocation:0, offset:0,   format:'float32x3'},
                  {shaderLocation:1, offset:3*4, format:'float32x3'}
                ]
              },
              // instance data
              {
                arrayStride:7*4,
                stepMode:'instance',
                attributes:[
                  {shaderLocation:2, offset:0,   format:'float32x3'},
                  {shaderLocation:3, offset:3*4, format:'float32x3'},
                  {shaderLocation:4, offset:6*4, format:'float32'}
                ]
              }
            ]
          },
          fragment:{
            module: device.createShaderModule({ code: pickingVertCode }),
            entryPoint:'fs_pick',
            targets:[{ format:'rgba8unorm' }]
          },
          primitive:{ topology:'triangle-list', cullMode:'back'},
          depthStencil:{ format:'depth24plus', depthWriteEnabled:true, depthCompare:'less-equal'}
        });
      },
      cache
    );
  },

  createRenderObject(device, pipeline, pickingPipeline, instanceVB, pickingInstanceVB, count, resources){
    if(!resources.ringGeo){
      throw new Error("No ring geometry available (not yet initialized).");
    }
    const { vb, ib, indexCount } = resources.ringGeo;
    return {
      pipeline,
      vertexBuffers: [vb, instanceVB!],
      indexBuffer: ib,
      indexCount,
      instanceCount: count,

      pickingPipeline,
      pickingVertexBuffers: [vb, pickingInstanceVB!],
      pickingIndexBuffer: ib,
      pickingIndexCount: indexCount,
      pickingInstanceCount: count,

      elementIndex: -1
    };
  }
};

/******************************************************
 * 2.4) Cuboid ExtendedSpec
 ******************************************************/
const cuboidExtendedSpec: ExtendedSpec<CuboidElementConfig> = {
  ...cuboidSpec,

  getRenderPipeline(device, bindGroupLayout, cache) {
    const format = navigator.gpu.getPreferredCanvasFormat();
    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout]
    });
    return getOrCreatePipeline(
      device,
      "CuboidShading",
      () => {
        return device.createRenderPipeline({
          layout: pipelineLayout,
          vertex:{
            module: device.createShaderModule({ code: cuboidVertCode }),
            entryPoint:'vs_main',
            buffers:[
              // cube geometry
              {
                arrayStride: 6*4,
                attributes:[
                  {shaderLocation:0, offset:0,   format:'float32x3'},
                  {shaderLocation:1, offset:3*4, format:'float32x3'}
                ]
              },
              // instance data
              {
                arrayStride: 10*4,
                stepMode:'instance',
                attributes:[
                  {shaderLocation:2, offset:0,    format:'float32x3'},
                  {shaderLocation:3, offset:3*4,  format:'float32x3'},
                  {shaderLocation:4, offset:6*4,  format:'float32x3'},
                  {shaderLocation:5, offset:9*4,  format:'float32'}
                ]
              }
            ]
          },
          fragment:{
            module: device.createShaderModule({ code: cuboidFragCode }),
            entryPoint:'fs_main',
            targets:[{
              format,
              blend:{
                color:{srcFactor:'src-alpha', dstFactor:'one-minus-src-alpha'},
                alpha:{srcFactor:'one', dstFactor:'one-minus-src-alpha'}
              }
            }]
          },
          primitive:{ topology:'triangle-list', cullMode:'none' },
          depthStencil:{format:'depth24plus', depthWriteEnabled:true, depthCompare:'less-equal'}
        });
      },
      cache
    );
  },

  getPickingPipeline(device, bindGroupLayout, cache) {
    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout]
    });
    return getOrCreatePipeline(
      device,
      "CuboidPicking",
      () => {
        return device.createRenderPipeline({
          layout: pipelineLayout,
          vertex:{
            module: device.createShaderModule({ code: pickingVertCode }),
            entryPoint:'vs_cuboid',
            buffers:[
              // cube geometry
              {
                arrayStride: 6*4,
                attributes:[
                  {shaderLocation:0, offset:0,   format:'float32x3'},
                  {shaderLocation:1, offset:3*4, format:'float32x3'}
                ]
              },
              // instance data
              {
                arrayStride: 7*4,
                stepMode:'instance',
                attributes:[
                  {shaderLocation:2, offset:0,   format:'float32x3'},
                  {shaderLocation:3, offset:3*4, format:'float32x3'},
                  {shaderLocation:4, offset:6*4, format:'float32'}
                ]
              }
            ]
          },
          fragment:{
            module: device.createShaderModule({ code: pickingVertCode }),
            entryPoint:'fs_pick',
            targets:[{ format:'rgba8unorm' }]
          },
          primitive:{ topology:'triangle-list', cullMode:'none'},
          depthStencil:{ format:'depth24plus', depthWriteEnabled:true, depthCompare:'less-equal'}
        });
      },
      cache
    );
  },

  createRenderObject(device, pipeline, pickingPipeline, instanceVB, pickingInstanceVB, count, resources){
    if(!resources.cubeGeo){
      throw new Error("No cube geometry available (not yet initialized).");
    }
    const { vb, ib, indexCount } = resources.cubeGeo;
    return {
      pipeline,
      vertexBuffers: [vb, instanceVB!],
      indexBuffer: ib,
      indexCount,
      instanceCount: count,

      pickingPipeline,
      pickingVertexBuffers: [vb, pickingInstanceVB!],
      pickingIndexBuffer: ib,
      pickingIndexCount: indexCount,
      pickingInstanceCount: count,

      elementIndex: -1
    };
  }
};

/******************************************************
 * 2.5) Put them all in a registry
 ******************************************************/
export type SceneElementConfig =
  | PointCloudElementConfig
  | EllipsoidElementConfig
  | EllipsoidBoundsElementConfig
  | CuboidElementConfig;

const primitiveRegistry: Record<SceneElementConfig['type'], ExtendedSpec<any>> = {
  PointCloud: pointCloudExtendedSpec,
  Ellipsoid: ellipsoidExtendedSpec,
  EllipsoidBounds: ellipsoidBoundsExtendedSpec,
  Cuboid: cuboidExtendedSpec,
};

/******************************************************
 * 3) Geometry creation helpers
 ******************************************************/
function createSphereGeometry(stacks=16, slices=24) {
  const verts:number[]=[];
  const idxs:number[]=[];
  for(let i=0;i<=stacks;i++){
    const phi=(i/stacks)*Math.PI;
    const sp=Math.sin(phi), cp=Math.cos(phi);
    for(let j=0;j<=slices;j++){
      const theta=(j/slices)*2*Math.PI;
      const st=Math.sin(theta), ct=Math.cos(theta);
      const x=sp*ct, y=cp, z=sp*st;
      verts.push(x,y,z, x,y,z); // pos + normal
    }
  }
  for(let i=0;i<stacks;i++){
    for(let j=0;j<slices;j++){
      const row1=i*(slices+1)+j;
      const row2=(i+1)*(slices+1)+j;
      idxs.push(row1,row2,row1+1, row1+1,row2,row2+1);
    }
  }
  return {
    vertexData: new Float32Array(verts),
    indexData: new Uint16Array(idxs)
  };
}

function createTorusGeometry(majorRadius:number, minorRadius:number, majorSegments:number, minorSegments:number) {
  const verts:number[]=[];
  const idxs:number[]=[];
  for(let j=0;j<=majorSegments;j++){
    const theta=(j/majorSegments)*2*Math.PI;
    const ct=Math.cos(theta), st=Math.sin(theta);
    for(let i=0;i<=minorSegments;i++){
      const phi=(i/minorSegments)*2*Math.PI;
      const cp=Math.cos(phi), sp=Math.sin(phi);
      const x=(majorRadius+minorRadius*cp)*ct;
      const y=(majorRadius+minorRadius*cp)*st;
      const z=minorRadius*sp;
      const nx=cp*ct, ny=cp*st, nz=sp;
      verts.push(x,y,z, nx,ny,nz);
    }
  }
  for(let j=0;j<majorSegments;j++){
    const row1=j*(minorSegments+1);
    const row2=(j+1)*(minorSegments+1);
    for(let i=0;i<minorSegments;i++){
      const a=row1+i, b=row1+i+1, c=row2+i, d=row2+i+1;
      idxs.push(a,b,c, b,d,c);
    }
  }
  return {
    vertexData: new Float32Array(verts),
    indexData: new Uint16Array(idxs)
  };
}

function createCubeGeometry() {
  // 6 faces => 24 verts, 36 indices
  const positions: number[] = [
    // +X face
    0.5, -0.5, -0.5,   0.5, -0.5,  0.5,   0.5,  0.5,  0.5,   0.5,  0.5, -0.5,
    // -X face
    -0.5, -0.5,  0.5,  -0.5, -0.5, -0.5,  -0.5,  0.5, -0.5,  -0.5,  0.5,  0.5,
    // +Y face
    -0.5,  0.5, -0.5,   0.5,  0.5, -0.5,   0.5,  0.5,  0.5,  -0.5,  0.5,  0.5,
    // -Y face
    -0.5, -0.5,  0.5,   0.5, -0.5,  0.5,   0.5, -0.5, -0.5,  -0.5, -0.5, -0.5,
    // +Z face
    -0.5, -0.5,  0.5,   0.5, -0.5,  0.5,   0.5,  0.5,  0.5,  -0.5,  0.5,  0.5,
    // -Z face
     0.5, -0.5, -0.5,  -0.5, -0.5, -0.5,  -0.5,  0.5, -0.5,   0.5,  0.5, -0.5,
  ];
  const normals: number[] = [
    // +X
    1,0,0, 1,0,0, 1,0,0, 1,0,0,
    // -X
    -1,0,0, -1,0,0, -1,0,0, -1,0,0,
    // +Y
    0,1,0, 0,1,0, 0,1,0, 0,1,0,
    // -Y
    0,-1,0, 0,-1,0, 0,-1,0, 0,-1,0,
    // +Z
    0,0,1, 0,0,1, 0,0,1, 0,0,1,
    // -Z
    0,0,-1, 0,0,-1, 0,0,-1, 0,0,-1,
  ];
  const indices: number[] = [];
  for(let face=0; face<6; face++){
    const base = face*4;
    indices.push(base+0, base+2, base+1, base+0, base+3, base+2);
  }
  // Interleave
  const vertexData = new Float32Array(positions.length*2);
  for(let i=0; i<positions.length/3; i++){
    vertexData[i*6+0] = positions[i*3+0];
    vertexData[i*6+1] = positions[i*3+1];
    vertexData[i*6+2] = positions[i*3+2];
    vertexData[i*6+3] = normals[i*3+0];
    vertexData[i*6+4] = normals[i*3+1];
    vertexData[i*6+5] = normals[i*3+2];
  }
  return {
    vertexData,
    indexData: new Uint16Array(indices),
  };
}

/******************************************************
 * 4) Scene + SceneWrapper
 ******************************************************/
interface SceneProps {
  elements: SceneElementConfig[];
  width?: number;
  height?: number;
  aspectRatio?: number;
  camera?: CameraState;  // Controlled mode
  defaultCamera?: CameraState;  // Uncontrolled mode
  onCameraChange?: (camera: CameraState) => void;
}

export function Scene({ elements, width, height, aspectRatio = 1, camera, defaultCamera, onCameraChange }: SceneProps) {
  const [containerRef, measuredWidth] = useContainerWidth(1);
  const dimensions = useMemo(
    () => computeCanvasDimensions(measuredWidth, width, height, aspectRatio),
    [measuredWidth, width, height, aspectRatio]
  );

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      {dimensions && (
        <SceneInner
          containerWidth={dimensions.width}
          containerHeight={dimensions.height}
          style={dimensions.style}
          elements={elements}
          camera={camera}
          defaultCamera={defaultCamera}
          onCameraChange={onCameraChange}
        />
      )}
    </div>
  );
}

/******************************************************
 * 4.1) The Scene Component
 ******************************************************/
function SceneInner({
  elements,
  containerWidth,
  containerHeight,
  style,
  camera: controlledCamera,
  defaultCamera,
  onCameraChange
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
    elementBaseId: number[];
    idToElement: {elementIdx: number, instanceIdx: number}[];
    pipelineCache: Map<string, PipelineCacheEntry>;  // Add this
    resources: {  // Add this
      sphereGeo: { vb: GPUBuffer; ib: GPUBuffer; indexCount: number } | null;
      ringGeo: { vb: GPUBuffer; ib: GPUBuffer; indexCount: number } | null;
      billboardQuad: { vb: GPUBuffer; ib: GPUBuffer; } | null;
      cubeGeo: { vb: GPUBuffer; ib: GPUBuffer; indexCount: number } | null;
    };
  } | null>(null);

  const [isReady, setIsReady] = useState(false);

  // Initialize camera with proper state structure
  const [internalCamera, setInternalCamera] = useState(() =>
    defaultCamera ?? DEFAULT_CAMERA
  );

  // Use controlled camera if provided, otherwise use internal state
  const camera = controlledCamera ?? internalCamera;

  // Unified camera update handler
  const handleCameraUpdate = useCallback((updateFn: (camera: CameraState) => CameraState) => {
    const newCamera = updateFn(camera);

    if (controlledCamera) {
      // In controlled mode, only notify parent
      onCameraChange?.(newCamera);
    } else {
      // In uncontrolled mode, update internal state
      setInternalCamera(newCamera);
      // Optionally notify parent
      onCameraChange?.(newCamera);
    }
  }, [camera, controlledCamera, onCameraChange]);

  // We'll also track a picking lock
  const pickingLockRef = useRef(false);

  // Add hover state tracking
  const lastHoverState = useRef<{elementIdx: number, instanceIdx: number} | null>(null);

  // ---------- Minimal math utils ----------
  function mat4Multiply(a: Float32Array, b: Float32Array) {
    const out = new Float32Array(16);
    for (let i=0; i<4; i++) {
      for (let j=0; j<4; j++) {
        out[j*4+i] =
          a[i+0]*b[j*4+0] + a[i+4]*b[j*4+1] + a[i+8]*b[j*4+2] + a[i+12]*b[j*4+3];
      }
    }
    return out;
  }
  function mat4Perspective(fov: number, aspect: number, near: number, far: number) {
    const out = new Float32Array(16);
    const f = 1.0 / Math.tan(fov/2);
    out[0]=f/aspect; out[5]=f;
    out[10]=(far+near)/(near-far); out[11] = -1;
    out[14]=(2*far*near)/(near-far);
    return out;
  }
  function mat4LookAt(eye:[number,number,number], target:[number,number,number], up:[number,number,number]) {
    const zAxis = normalize([eye[0]-target[0],eye[1]-target[1],eye[2]-target[2]]);
    const xAxis = normalize(cross(up,zAxis));
    const yAxis = cross(zAxis,xAxis);
    const out=new Float32Array(16);
    out[0]=xAxis[0]; out[1]=yAxis[0]; out[2]=zAxis[0]; out[3]=0;
    out[4]=xAxis[1]; out[5]=yAxis[1]; out[6]=zAxis[1]; out[7]=0;
    out[8]=xAxis[2]; out[9]=yAxis[2]; out[10]=zAxis[2]; out[11]=0;
    out[12]=-dot(xAxis,eye); out[13]=-dot(yAxis,eye); out[14]=-dot(zAxis,eye); out[15]=1;
    return out;
  }
  function cross(a:[number,number,number], b:[number,number,number]){
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]] as [number,number,number];
  }
  function dot(a:[number,number,number], b:[number,number,number]){
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
  }
  function normalize(v:[number,number,number]){
    const len=Math.sqrt(dot(v,v));
    if(len>1e-6) return [v[0]/len, v[1]/len, v[2]/len] as [number,number,number];
    return [0,0,0];
  }

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
        elementBaseId: [],
        idToElement: [],
        pipelineCache: new Map(),
        resources: {
          sphereGeo: null,
          ringGeo: null,
          billboardQuad: null,
          cubeGeo: null
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

    const dpr = window.devicePixelRatio || 1;
    const displayWidth = Math.floor(containerWidth * dpr);
    const displayHeight = Math.floor(containerHeight * dpr);

    if(depthTexture) depthTexture.destroy();
    const dt = device.createTexture({
      size: [displayWidth, displayHeight],
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT
    });
    gpuRef.current.depthTexture = dt;
  }, [containerWidth, containerHeight]);

  const createOrUpdatePickTextures = useCallback(() => {
    if(!gpuRef.current || !canvasRef.current) return;
    const { device, pickTexture, pickDepthTexture } = gpuRef.current;

    const dpr = window.devicePixelRatio || 1;
    const displayWidth = Math.floor(containerWidth * dpr);
    const displayHeight = Math.floor(containerHeight * dpr);

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
  }, [containerWidth, containerHeight]);

  /******************************************************
   * C) Building the RenderObjects (no if/else)
   ******************************************************/
  function buildRenderObjects(elements: SceneElementConfig[]): RenderObject[] {
    if(!gpuRef.current) return [];
    const { device, bindGroupLayout, pipelineCache, resources } = gpuRef.current;

    // Initialize elementBaseId and idToElement arrays
    gpuRef.current.elementBaseId = [];
    gpuRef.current.idToElement = [null];  // First ID (0) is reserved
    let currentID = 1;

    return elements.map((elem, i) => {
      const spec = primitiveRegistry[elem.type];
      if(!spec) {
        gpuRef.current!.elementBaseId[i] = 0;  // No valid IDs for invalid elements
        return {
          pipeline: null,
          vertexBuffers: [],
          indexBuffer: null,
          vertexCount: 0,
          indexCount: 0,
          instanceCount: 0,
          pickingPipeline: null,
          pickingVertexBuffers: [],
          pickingIndexBuffer: null,
          pickingVertexCount: 0,
          pickingIndexCount: 0,
          pickingInstanceCount: 0,
          elementIndex: i
        };
      }

      const count = spec.getCount(elem);
      gpuRef.current!.elementBaseId[i] = currentID;

      // Expand global ID table
      for(let j=0; j<count; j++){
        gpuRef.current!.idToElement[currentID + j] = { elementIdx: i, instanceIdx: j };
      }
      currentID += count;

      const renderData = spec.buildRenderData(elem);
      const pickData   = spec.buildPickingData(elem, gpuRef.current!.elementBaseId[i]);

      let vb: GPUBuffer|null = null;
      if(renderData && renderData.length>0){
        vb = device.createBuffer({
          size: renderData.byteLength,
          usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });
        device.queue.writeBuffer(vb,0,renderData);
      }
      let pickVB: GPUBuffer|null = null;
      if(pickData && pickData.length>0){
        pickVB = device.createBuffer({
          size: pickData.byteLength,
          usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });
        device.queue.writeBuffer(pickVB,0,pickData);
      }

      const pipeline = spec.getRenderPipeline(device, bindGroupLayout, pipelineCache);
      const pickingPipeline = spec.getPickingPipeline(device, bindGroupLayout, pipelineCache);

      return spec.createRenderObject(
        device,
        pipeline,
        pickingPipeline,
        vb,
        pickVB,
        count,
        resources
      );
    });
  }

  /******************************************************
   * D) Render pass (single call, no loop)
   ******************************************************/
  const renderFrame = useCallback((camState: CameraState) => {
    if(!gpuRef.current) return;
    const { device, context, uniformBuffer, uniformBindGroup, depthTexture, renderObjects } = gpuRef.current;
    if(!depthTexture) return;

    const aspect = containerWidth/containerHeight;
    const proj = mat4Perspective(camState.fov, aspect, camState.near, camState.far);

    // Calculate camera position from orbit coordinates
    const cx = camState.orbitRadius * Math.sin(camState.orbitPhi) * Math.sin(camState.orbitTheta);
    const cy = camState.orbitRadius * Math.cos(camState.orbitPhi);
    const cz = camState.orbitRadius * Math.sin(camState.orbitPhi) * Math.cos(camState.orbitTheta);
    const eye: [number,number,number] = [cx, cy, cz];
    const target: [number,number,number] = [camState.panX, camState.panY, 0];
    const up: [number,number,number] = [0, 1, 0];
    const view = mat4LookAt(eye, target, up);

    const mvp = mat4Multiply(proj, view);

    // compute "light dir" in camera-ish space
    const forward = normalize([target[0]-eye[0], target[1]-eye[1], target[2]-eye[2]]);
    const right = normalize(cross(up, forward));
    const camUp = cross(right, forward);
    const lightDir = normalize([
      right[0]*LIGHTING.DIRECTION.RIGHT + camUp[0]*LIGHTING.DIRECTION.UP + forward[0]*LIGHTING.DIRECTION.FORWARD,
      right[1]*LIGHTING.DIRECTION.RIGHT + camUp[1]*LIGHTING.DIRECTION.UP + forward[1]*LIGHTING.DIRECTION.FORWARD,
      right[2]*LIGHTING.DIRECTION.RIGHT + camUp[2]*LIGHTING.DIRECTION.UP + forward[2]*LIGHTING.DIRECTION.FORWARD,
    ]);

    // write uniform
    const data=new Float32Array(32);
    data.set(mvp,0);
    data[16] = right[0]; data[17] = right[1]; data[18] = right[2];
    data[20] = camUp[0]; data[21] = camUp[1]; data[22] = camUp[2];
    data[24] = lightDir[0]; data[25] = lightDir[1]; data[26] = lightDir[2];
    device.queue.writeBuffer(uniformBuffer,0,data);

    let tex:GPUTexture;
    try {
      tex = context.getCurrentTexture();
    } catch(e){
      return; // If canvas is resized or lost, bail
    }
    const passDesc: GPURenderPassDescriptor = {
      colorAttachments:[{
        view: tex.createView(),
        loadOp:'clear',
        storeOp:'store',
        clearValue:{r:0.15,g:0.15,b:0.15,a:1}
      }],
      depthStencilAttachment:{
        view: depthTexture.createView(),
        depthLoadOp:'clear',
        depthStoreOp:'store',
        depthClearValue:1.0
      }
    };

    const cmd = device.createCommandEncoder();
    const pass = cmd.beginRenderPass(passDesc);

    // draw each object
    for(const ro of renderObjects){
      pass.setPipeline(ro.pipeline);
      pass.setBindGroup(0, uniformBindGroup);
      ro.vertexBuffers.forEach((vb,i)=> pass.setVertexBuffer(i,vb));
      if(ro.indexBuffer){
        pass.setIndexBuffer(ro.indexBuffer,'uint16');
        pass.drawIndexed(ro.indexCount ?? 0, ro.instanceCount ?? 1);
      } else {
        pass.draw(ro.vertexCount ?? 0, ro.instanceCount ?? 1);
      }
    }
    pass.end();
    device.queue.submit([cmd.finish()]);
  }, [containerWidth, containerHeight]);

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
        uniformBindGroup, renderObjects, idToElement
      } = gpuRef.current;
      if(!pickTexture || !pickDepthTexture || !readbackBuffer) return;
      if (currentPickingId !== pickingId) return;

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
      for(const ro of renderObjects){
        pass.setPipeline(ro.pickingPipeline);
        ro.pickingVertexBuffers.forEach((vb,i)=>{
          pass.setVertexBuffer(i, vb);
        });
        if(ro.pickingIndexBuffer){
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
    const { idToElement } = gpuRef.current;

    // Get new hover state
    const newHoverState = idToElement[pickedID] || null;

    // If hover state hasn't changed, do nothing
    if ((!lastHoverState.current && !newHoverState) ||
        (lastHoverState.current && newHoverState &&
         lastHoverState.current.elementIdx === newHoverState.elementIdx &&
         lastHoverState.current.instanceIdx === newHoverState.instanceIdx)) {
      return;
    }

    // Clear previous hover if it exists
    if (lastHoverState.current) {
      const prevElement = elements[lastHoverState.current.elementIdx];
      prevElement?.onHover?.(null);
    }

    // Set new hover if it exists
    if (newHoverState) {
      const { elementIdx, instanceIdx } = newHoverState;
      if (elementIdx >= 0 && elementIdx < elements.length) {
        elements[elementIdx].onHover?.(instanceIdx);
      }
    }

    // Update last hover state
    lastHoverState.current = newHoverState;
  }

  function handleClickID(pickedID:number){
    if(!gpuRef.current) return;
    const {idToElement} = gpuRef.current;
    const rec = idToElement[pickedID];
    if(!rec) return;
    const {elementIdx, instanceIdx} = rec;
    if(elementIdx<0||elementIdx>=elements.length) return;
    elements[elementIdx].onClick?.(instanceIdx);
  }

  /******************************************************
   * F) Mouse Handling
   ******************************************************/
  interface MouseState {
    type: 'idle'|'dragging';
    button?: number;
    startX?: number;
    startY?: number;
    lastX?: number;
    lastY?: number;
    isShiftDown?: boolean;
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
  const handleScene3dMouseMove = useCallback((e: ReactMouseEvent) => {
    if(!canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const st = mouseState.current;
    if(st.type === 'dragging' && st.lastX !== undefined && st.lastY !== undefined) {
      const dx = e.clientX - st.lastX;
      const dy = e.clientY - st.lastY;
      st.dragDistance=(st.dragDistance||0)+Math.sqrt(dx*dx+dy*dy);
      if(st.button===2 || st.isShiftDown){
        // Pan by keeping the point under the mouse cursor fixed relative to mouse movement
        handleCameraUpdate(cam => {
          // Calculate camera position and vectors
          const cx = cam.orbitRadius * Math.sin(cam.orbitPhi) * Math.sin(cam.orbitTheta);
          const cy = cam.orbitRadius * Math.cos(cam.orbitPhi);
          const cz = cam.orbitRadius * Math.sin(cam.orbitPhi) * Math.cos(cam.orbitTheta);
          const eye: [number,number,number] = [cx, cy, cz];
          const target: [number,number,number] = [cam.panX, cam.panY, 0];
          const up: [number,number,number] = [0, 1, 0];

          // Calculate view-aligned right and up vectors
          const forward = normalize([target[0]-eye[0], target[1]-eye[1], target[2]-eye[2]]);
          const right = normalize(cross(up, forward));
          const actualUp = normalize(cross(forward, right));

          // Scale movement by distance from target
          const scale = cam.orbitRadius * 0.002;

          // Calculate pan offset in view space
          const dx_view = dx * scale;
          const dy_view = dy * scale;

          return {
            ...cam,
            panX: cam.panX + right[0] * dx_view + actualUp[0] * dy_view,
            panY: cam.panY + right[1] * dx_view + actualUp[1] * dy_view
          };
        });
      } else if(st.button===0){
        // Orbit
        handleCameraUpdate(cam => {
          const newPhi = Math.max(0.1, Math.min(Math.PI-0.1, cam.orbitPhi - dy * 0.01));
          return {
            ...cam,
            orbitTheta: cam.orbitTheta - dx * 0.01,
            orbitPhi: newPhi
          };
        });
      }
      st.lastX=e.clientX;
      st.lastY=e.clientY;
    } else if(st.type === 'idle') {
      throttledPickAtScreenXY(x, y, 'hover');
    }
  }, [handleCameraUpdate, throttledPickAtScreenXY]);

  const handleScene3dMouseDown = useCallback((e: ReactMouseEvent) => {
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

  const handleScene3dMouseUp = useCallback((e: ReactMouseEvent) => {
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
          // Cleanup instance resources
          resources.sphereGeo?.vb.destroy();
          resources.sphereGeo?.ib.destroy();
          // ... cleanup other geometries ...

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
      renderFrame(camera);  // Always render with current camera (controlled or internal)
    }
  }, [isReady, camera, renderFrame]); // Watch the camera value

  // Update canvas size effect
  useEffect(() => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const dpr = window.devicePixelRatio || 1;
    const displayWidth = Math.floor(containerWidth * dpr);
    const displayHeight = Math.floor(containerHeight * dpr);

    if (canvas.width !== displayWidth || canvas.height !== displayHeight) {
      canvas.width = displayWidth;
      canvas.height = displayHeight;
      createOrUpdateDepthTexture();
      createOrUpdatePickTextures();
      renderFrame(camera);
    }
  }, [containerWidth, containerHeight, createOrUpdateDepthTexture, createOrUpdatePickTextures, renderFrame, camera]);

  // Update elements effect
  useEffect(() => {
    if (isReady && gpuRef.current) {
      const ros = buildRenderObjects(elements);
      gpuRef.current.renderObjects = ros;
      renderFrame(camera);
    }
  }, [isReady, elements, renderFrame, camera]);

  // Wheel handling
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const handleWheel = (e: WheelEvent) => {
      if (mouseState.current.type === 'idle') {
        e.preventDefault();
        handleCameraUpdate(cam => ({
          ...cam,
          orbitRadius: Math.max(0.1, cam.orbitRadius * Math.exp(e.deltaY * 0.001))
        }));
      }
    };

    canvas.addEventListener('wheel', handleWheel, { passive: false });
    return () => canvas.removeEventListener('wheel', handleWheel);
  }, [handleCameraUpdate]);

  return (
    <div style={{ width: '100%', border: '1px solid #ccc' }}>
      <canvas
        ref={canvasRef}
        style={style}
        onMouseMove={handleScene3dMouseMove}
        onMouseDown={handleScene3dMouseDown}
        onMouseUp={handleScene3dMouseUp}
        onMouseLeave={handleScene3dMouseLeave}
      />
    </div>
  );
}


/******************************************************
 * 6) Minimal WGSL code
 ******************************************************/
const billboardVertCode = /*wgsl*/`
struct Camera {
  mvp: mat4x4<f32>,
  cameraRight: vec3<f32>,
  _pad1: f32,
  cameraUp: vec3<f32>,
  _pad2: f32,
  lightDir: vec3<f32>,
  _pad3: f32,
};
@group(0) @binding(0) var<uniform> camera : Camera;

struct VSOut {
  @builtin(position) Position: vec4<f32>,
  @location(0) color: vec3<f32>,
  @location(1) alpha: f32
};

@vertex
fn vs_main(
  @location(0) corner: vec2<f32>,
  @location(1) pos: vec3<f32>,
  @location(2) col: vec3<f32>,
  @location(3) alpha: f32,
  @location(4) scaleX: f32,
  @location(5) scaleY: f32
)-> VSOut {
  let offset = camera.cameraRight*(corner.x*scaleX) + camera.cameraUp*(corner.y*scaleY);
  let worldPos = vec4<f32>(pos + offset, 1.0);
  var out: VSOut;
  out.Position = camera.mvp * worldPos;
  out.color = col;
  out.alpha = alpha;
  return out;
}
`;

const billboardFragCode = /*wgsl*/`
@fragment
fn fs_main(@location(0) color: vec3<f32>, @location(1) alpha: f32)-> @location(0) vec4<f32> {
  return vec4<f32>(color, alpha);
}
`;

const ellipsoidVertCode = /*wgsl*/`
struct Camera {
  mvp: mat4x4<f32>,
  cameraRight: vec3<f32>,
  _pad1: f32,
  cameraUp: vec3<f32>,
  _pad2: f32,
  lightDir: vec3<f32>,
  _pad3: f32,
};
@group(0) @binding(0) var<uniform> camera : Camera;

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
  out.worldPos = worldPos.xyz;
  return out;
}
`;

const ellipsoidFragCode = /*wgsl*/`
struct Camera {
  mvp: mat4x4<f32>,
  cameraRight: vec3<f32>,
  _pad1: f32,
  cameraUp: vec3<f32>,
  _pad2: f32,
  lightDir: vec3<f32>,
  _pad3: f32,
};
@group(0) @binding(0) var<uniform> camera : Camera;

@fragment
fn fs_main(
  @location(1) normal: vec3<f32>,
  @location(2) baseColor: vec3<f32>,
  @location(3) alpha: f32,
  @location(4) worldPos: vec3<f32>
)-> @location(0) vec4<f32> {
  let N = normalize(normal);
  let L = normalize(camera.lightDir);
  let lambert = max(dot(N,L), 0.0);
  let ambient = ${LIGHTING.AMBIENT_INTENSITY};
  var color = baseColor * (ambient + lambert*${LIGHTING.DIFFUSE_INTENSITY});

  let V = normalize(-worldPos);
  let H = normalize(L + V);
  let spec = pow(max(dot(N,H),0.0), ${LIGHTING.SPECULAR_POWER});
  color += vec3<f32>(1.0,1.0,1.0)*spec*${LIGHTING.SPECULAR_INTENSITY};

  return vec4<f32>(color, alpha);
}
`;

const ringVertCode = /*wgsl*/`
struct Camera {
  mvp: mat4x4<f32>,
  cameraRight: vec3<f32>,
  _pad1: f32,
  cameraUp: vec3<f32>,
  _pad2: f32,
  lightDir: vec3<f32>,
  _pad3: f32,
};
@group(0) @binding(0) var<uniform> camera : Camera;

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
}
`;

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
}
`;

const cuboidVertCode = /*wgsl*/`
struct Camera {
  mvp: mat4x4<f32>,
  cameraRight: vec3<f32>,
  _pad1: f32,
  cameraUp: vec3<f32>,
  _pad2: f32,
  lightDir: vec3<f32>,
  _pad3: f32,
};
@group(0) @binding(0) var<uniform> camera: Camera;

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
}
`;

const cuboidFragCode = /*wgsl*/`
struct Camera {
  mvp: mat4x4<f32>,
  cameraRight: vec3<f32>,
  _pad1: f32,
  cameraUp: vec3<f32>,
  _pad2: f32,
  lightDir: vec3<f32>,
  _pad3: f32,
};
@group(0) @binding(0) var<uniform> camera: Camera;

@fragment
fn fs_main(
  @location(1) normal: vec3<f32>,
  @location(2) baseColor: vec3<f32>,
  @location(3) alpha: f32,
  @location(4) worldPos: vec3<f32>
)-> @location(0) vec4<f32> {
  let N = normalize(normal);
  let L = normalize(camera.lightDir);
  let lambert = max(dot(N,L), 0.0);
  let ambient = ${LIGHTING.AMBIENT_INTENSITY};
  var color = baseColor * (ambient + lambert*${LIGHTING.DIFFUSE_INTENSITY});

  let V = normalize(-worldPos);
  let H = normalize(L + V);
  let spec = pow(max(dot(N,H),0.0), ${LIGHTING.SPECULAR_POWER});
  color += vec3<f32>(1.0,1.0,1.0)*spec*${LIGHTING.SPECULAR_INTENSITY};

  return vec4<f32>(color, alpha);
}
`;

const pickingVertCode = /*wgsl*/`
struct Camera {
  mvp: mat4x4<f32>,
  cameraRight: vec3<f32>,
  _pad1: f32,
  cameraUp: vec3<f32>,
  _pad2: f32,
  lightDir: vec3<f32>,
  _pad3: f32,
};
@group(0) @binding(0) var<uniform> camera: Camera;

struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) pickID: f32
};

@vertex
fn vs_pointcloud(
  @location(0) corner: vec2<f32>,
  @location(1) pos: vec3<f32>,
  @location(2) pickID: f32,
  @location(3) scaleX: f32,
  @location(4) scaleY: f32
)-> VSOut {
  let offset = camera.cameraRight*(corner.x*scaleX) + camera.cameraUp*(corner.y*scaleY);
  let worldPos = vec4<f32>(pos + offset, 1.0);
  var out: VSOut;
  out.pos = camera.mvp * worldPos;
  out.pickID = pickID;
  return out;
}

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
}

@vertex
fn vs_bands(
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
}

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
}

@fragment
fn fs_pick(@location(0) pickID: f32)-> @location(0) vec4<f32> {
  let iID = u32(pickID);
  let r = f32(iID & 255u)/255.0;
  let g = f32((iID>>8)&255u)/255.0;
  let b = f32((iID>>16)&255u)/255.0;
  return vec4<f32>(r,g,b,1.0);
}
`;

// Add this at the top with other imports
function throttle<T extends (...args: any[]) => void>(
  func: T,
  limit: number
): (...args: Parameters<T>) => void {
  let inThrottle = false;
  return function(this: any, ...args: Parameters<T>) {
    if (!inThrottle) {
      func.apply(this, args);
      inThrottle = true;
      setTimeout(() => (inThrottle = false), limit);
    }
  };
}

// Add this helper function near the top with other utility functions
function computeCanvasDimensions(containerWidth: number, width?: number, height?: number, aspectRatio = 1) {
    if (!containerWidth && !width) return;

    // Determine final width from explicit width or container width
    const finalWidth = width || containerWidth;

    // Determine final height from explicit height or aspect ratio
    const finalHeight = height || finalWidth / aspectRatio;

    return {
        width: finalWidth,
        height: finalHeight,
        style: {
            width: width ? `${width}px` : '100%',
            height: `${finalHeight}px`
        }
    };
}
