/// <reference path="./webgpu.d.ts" />
/// <reference types="react" />

import React, {
  useRef, useEffect, useState, useCallback, MouseEvent as ReactMouseEvent
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
    cache: Map<string, GPURenderPipeline>
  ): GPURenderPipeline;

  /** Return (or lazily create) the GPU picking pipeline for this type. */
  getPickingPipeline(
    device: GPUDevice,
    bindGroupLayout: GPUBindGroupLayout,
    cache: Map<string, GPURenderPipeline>
  ): GPURenderPipeline;

  /** Create a RenderObject that references geometry + the two pipelines. */
  createRenderObject(
    device: GPUDevice,
    pipeline: GPURenderPipeline,
    pickingPipeline: GPURenderPipeline,
    instanceVB: GPUBuffer|null,
    pickingVB: GPUBuffer|null,
    instanceCount: number
  ): RenderObject;
}

/******************************************************
 * 1.6) Pipeline Cache Helper
 ******************************************************/
const pipelineCache = new Map<string, GPURenderPipeline>();
function getOrCreatePipeline(
  key: string,
  createFn: () => GPURenderPipeline,
  cache: Map<string, GPURenderPipeline>
): GPURenderPipeline {
  if (cache.has(key)) {
    return cache.get(key)!;
  }
  const pipeline = createFn();
  cache.set(key, pipeline);
  return pipeline;
}

/******************************************************
 * 2) Common Resources: geometry, layout, etc.
 ******************************************************/
const CommonResources = {
  /** Track which device created our resources */
  currentDevice: null as GPUDevice | null,

  /** Sphere geometry for ellipsoids */
  sphereGeo: null as { vb: GPUBuffer; ib: GPUBuffer; indexCount: number } | null,

  /** Torus geometry for ellipsoid bounds */
  ringGeo: null as { vb: GPUBuffer; ib: GPUBuffer; indexCount: number } | null,

  /** Quad geometry for point clouds (billboard) */
  billboardQuad: null as { vb: GPUBuffer; ib: GPUBuffer; } | null,

  /** Cube geometry for cuboids */
  cubeGeo: null as { vb: GPUBuffer; ib: GPUBuffer; indexCount: number } | null,

  init(device: GPUDevice) {
    // If device changed, destroy old resources
    if (this.currentDevice && this.currentDevice !== device) {
      this.sphereGeo?.vb.destroy();
      this.sphereGeo?.ib.destroy();
      this.sphereGeo = null;

      this.ringGeo?.vb.destroy();
      this.ringGeo?.ib.destroy();
      this.ringGeo = null;

      this.billboardQuad?.vb.destroy();
      this.billboardQuad?.ib.destroy();
      this.billboardQuad = null;

      this.cubeGeo?.vb.destroy();
      this.cubeGeo?.ib.destroy();
      this.cubeGeo = null;
    }

    this.currentDevice = device;

    // Create geometry resources for this device
    if(!this.sphereGeo) {
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
      this.sphereGeo = { vb, ib, indexCount: indexData.length };
    }

    // create ring geometry
    if(!this.ringGeo){
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
      this.ringGeo = { vb, ib, indexCount: indexData.length };
    }

    // create billboard quad geometry
    if(!this.billboardQuad){
      const quadVerts = new Float32Array([-0.5,-0.5,  0.5,-0.5,  -0.5,0.5,  0.5,0.5]);
      const quadIdx   = new Uint16Array([0,1,2,2,1,3]);
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
      this.billboardQuad = { vb, ib };
    }

    // create cube geometry
    if(!this.cubeGeo){
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
      this.cubeGeo = { vb, ib, indexCount: cube.indexData.length };
    }
  }
};

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

    return getOrCreatePipeline("PointCloudShading", () => {
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
    }, cache);
  },

  getPickingPipeline(device, bindGroupLayout, cache) {
    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout]
    });
    return getOrCreatePipeline("PointCloudPicking", () => {
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
    }, cache);
  },

  createRenderObject(device, pipeline, pickingPipeline, instanceVB, pickingInstanceVB, count){
    if(!CommonResources.billboardQuad){
      throw new Error("No billboard geometry available (not yet initialized).");
    }
    return {
      pipeline,
      vertexBuffers: [CommonResources.billboardQuad.vb, instanceVB!],
      indexBuffer: CommonResources.billboardQuad.ib,
      indexCount: 6,
      instanceCount: count,

      pickingPipeline,
      pickingVertexBuffers: [CommonResources.billboardQuad.vb, pickingInstanceVB!],
      pickingIndexBuffer: CommonResources.billboardQuad.ib,
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
    return getOrCreatePipeline("EllipsoidShading", () => {
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
    }, cache);
  },

  getPickingPipeline(device, bindGroupLayout, cache) {
    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout]
    });
    return getOrCreatePipeline("EllipsoidPicking", () => {
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
    }, cache);
  },

  createRenderObject(device, pipeline, pickingPipeline, instanceVB, pickingInstanceVB, count){
    if(!CommonResources.sphereGeo) {
      throw new Error("No sphere geometry available (not yet initialized).");
    }
    const { vb, ib, indexCount } = CommonResources.sphereGeo;
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
    return getOrCreatePipeline("EllipsoidBoundsShading", () => {
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
    }, cache);
  },

  getPickingPipeline(device, bindGroupLayout, cache) {
    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout]
    });
    return getOrCreatePipeline("EllipsoidBoundsPicking", () => {
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
    }, cache);
  },

  createRenderObject(device, pipeline, pickingPipeline, instanceVB, pickingInstanceVB, count){
    if(!CommonResources.ringGeo){
      throw new Error("No ring geometry available (not yet initialized).");
    }
    const { vb, ib, indexCount } = CommonResources.ringGeo;
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
    return getOrCreatePipeline("CuboidShading", () => {
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
    }, cache);
  },

  getPickingPipeline(device, bindGroupLayout, cache) {
    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout]
    });
    return getOrCreatePipeline("CuboidPicking", () => {
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
    }, cache);
  },

  createRenderObject(device, pipeline, pickingPipeline, instanceVB, pickingInstanceVB, count){
    if(!CommonResources.cubeGeo){
      throw new Error("No cube geometry available (not yet initialized).");
    }
    const { vb, ib, indexCount } = CommonResources.cubeGeo;
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
  containerWidth: number;
}

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
 * 4.1) The Scene Component
 ******************************************************/
function Scene({ elements, containerWidth }: SceneProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const safeWidth = containerWidth > 0 ? containerWidth : 300;
  const canvasWidth = safeWidth;
  const canvasHeight = safeWidth;

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
  } | null>(null);

  const [isReady, setIsReady] = useState(false);

  // camera
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
  const [camera, setCamera] = useState<CameraState>({
    orbitRadius: 1.5,
    orbitTheta: 0.2,
    orbitPhi: 1.0,
    panX: 0,
    panY: 0,
    fov: Math.PI/3,
    near: 0.01,
    far: 100.0
  });

  // We'll also track a picking lock
  const pickingLockRef = useRef(false);

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

      // Initialize common geometry resources
      CommonResources.init(device);

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
        idToElement: []
      };

      setIsReady(true);
    } catch(err){
      console.error("initWebGPU error:", err);
    }
  },[]);

  /******************************************************
   * B) Depth & Pick textures
   ******************************************************/
  const createOrUpdateDepthTexture=useCallback(()=>{
    if(!gpuRef.current) return;
    const { device, depthTexture } = gpuRef.current;
    if(depthTexture) depthTexture.destroy();
    const dt = device.createTexture({
      size: [canvasWidth, canvasHeight],
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT
    });
    gpuRef.current.depthTexture = dt;
  },[canvasWidth, canvasHeight]);

  const createOrUpdatePickTextures=useCallback(()=>{
    if(!gpuRef.current) return;
    const { device, pickTexture, pickDepthTexture } = gpuRef.current;
    if(pickTexture) pickTexture.destroy();
    if(pickDepthTexture) pickDepthTexture.destroy();
    const colorTex = device.createTexture({
      size:[canvasWidth, canvasHeight],
      format:'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC
    });
    const depthTex = device.createTexture({
      size:[canvasWidth, canvasHeight],
      format:'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT
    });
    gpuRef.current.pickTexture = colorTex;
    gpuRef.current.pickDepthTexture = depthTex;
  },[canvasWidth, canvasHeight]);

  /******************************************************
   * C) Building the RenderObjects (no if/else)
   ******************************************************/
  function buildRenderObjects(sceneElements: SceneElementConfig[]): RenderObject[] {
    if(!gpuRef.current) return [];
    const { device, bindGroupLayout, elementBaseId, idToElement } = gpuRef.current;

    elementBaseId.length = 0;
    idToElement.length = 1;
    let currentID = 1;

    const result: RenderObject[] = [];

    sceneElements.forEach((elem, eIdx) => {
      const spec = primitiveRegistry[elem.type];
      if(!spec) {
        elementBaseId[eIdx] = 0;
        return;
      }
      const count = spec.getCount(elem);
      elementBaseId[eIdx] = currentID;

      // Expand global ID table
      for(let i=0; i<count; i++){
        idToElement[currentID + i] = { elementIdx: eIdx, instanceIdx: i };
      }
      currentID += count;

      const renderData = spec.buildRenderData(elem);
      const pickData   = spec.buildPickingData(elem, elementBaseId[eIdx]);

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

      const ro = spec.createRenderObject(device, pipeline, pickingPipeline, vb, pickVB, count);
      ro.elementIndex = eIdx;
      result.push(ro);
    });

    return result;
  }

  /******************************************************
   * D) Render pass (single call, no loop)
   ******************************************************/
  const renderFrame = useCallback((camState:any)=>{
    if(!gpuRef.current) return;
    const { device, context, uniformBuffer, uniformBindGroup, depthTexture, renderObjects } = gpuRef.current;
    if(!depthTexture) return;

    // camera matrix
    const aspect = canvasWidth/canvasHeight;
    const proj = mat4Perspective(camState.fov, aspect, camState.near, camState.far);
    const cx=camState.orbitRadius*Math.sin(camState.orbitPhi)*Math.sin(camState.orbitTheta);
    const cy=camState.orbitRadius*Math.cos(camState.orbitPhi);
    const cz=camState.orbitRadius*Math.sin(camState.orbitPhi)*Math.cos(camState.orbitTheta);
    const eye:[number,number,number]=[cx,cy,cz];
    const target:[number,number,number]=[camState.panX,camState.panY,0];
    const up:[number,number,number] = [0,1,0];
    const view = mat4LookAt(eye,target,up);

    const mvp = mat4Multiply(proj, view);

    // compute "light dir" in camera-ish space
    const forward = normalize([target[0]-eye[0], target[1]-eye[1], target[2]-eye[2]]);
    const right = normalize(cross(forward, up));
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
  },[canvasWidth, canvasHeight]);

  /******************************************************
   * E) Pick pass (on hover/click)
   ******************************************************/
  async function pickAtScreenXY(screenX:number, screenY:number, mode:'hover'|'click'){
    if(!gpuRef.current || pickingLockRef.current) return;
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

      const pickX = Math.floor(screenX);
      const pickY = Math.floor(screenY);
      if(pickX<0 || pickY<0 || pickX>=canvasWidth || pickY>=canvasHeight){
        if(mode==='hover') handleHoverID(0);
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

  function handleHoverID(pickedID:number){
    if(!gpuRef.current) return;
    const {idToElement} = gpuRef.current;
    if(!idToElement[pickedID]){
      // no object => clear hover
      for(const e of elements) e.onHover?.(null);
      return;
    }
    const {elementIdx, instanceIdx} = idToElement[pickedID];
    if(elementIdx<0||elementIdx>=elements.length){
      for(const e of elements) e.onHover?.(null);
      return;
    }
    elements[elementIdx].onHover?.(instanceIdx);
    // clear hover on others
    for(let i=0;i<elements.length;i++){
      if(i!==elementIdx){
        elements[i].onHover?.(null);
      }
    }
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

  const handleMouseMove=useCallback((e:ReactMouseEvent)=>{
    if(!canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const st=mouseState.current;
    if(st.type==='dragging' && st.lastX!==undefined && st.lastY!==undefined){
      const dx = e.clientX - st.lastX;
      const dy = e.clientY - st.lastY;
      st.dragDistance=(st.dragDistance||0)+Math.sqrt(dx*dx+dy*dy);
      if(st.button===2 || st.isShiftDown){
        setCamera(cam => ({
          ...cam,
          panX:cam.panX - dx*0.002,
          panY:cam.panY + dy*0.002
        }));
      } else if(st.button===0){
        setCamera(cam=>{
          const newPhi = Math.max(0.1, Math.min(Math.PI-0.1, cam.orbitPhi - dy*0.01));
          return {
            ...cam,
            orbitTheta: cam.orbitTheta - dx*0.01,
            orbitPhi: newPhi
          };
        });
      }
      st.lastX=e.clientX;
      st.lastY=e.clientY;
    } else if(st.type==='idle'){
      // picking => hover
      pickAtScreenXY(x,y,'hover');
    }
  },[pickAtScreenXY]);

  const handleMouseDown=useCallback((e:ReactMouseEvent)=>{
    mouseState.current={
      type:'dragging',
      button:e.button,
      startX:e.clientX,
      startY:e.clientY,
      lastX:e.clientX,
      lastY:e.clientY,
      isShiftDown:e.shiftKey,
      dragDistance:0
    };
    e.preventDefault();
  },[]);

  const handleMouseUp=useCallback((e:ReactMouseEvent)=>{
    const st=mouseState.current;
    if(st.type==='dragging' && st.startX!==undefined && st.startY!==undefined){
      if(!canvasRef.current) return;
      const rect=canvasRef.current.getBoundingClientRect();
      const x=e.clientX-rect.left, y=e.clientY-rect.top;
      // if we haven't dragged far => treat as click
      if((st.dragDistance||0) < 4){
        pickAtScreenXY(x,y,'click');
      }
    }
    mouseState.current={type:'idle'};
  },[pickAtScreenXY]);

  const handleMouseLeave=useCallback(()=>{
    mouseState.current={type:'idle'};
  },[]);

  const onWheel=useCallback((e:WheelEvent)=>{
    if(mouseState.current.type==='idle'){
      e.preventDefault();
      const d=e.deltaY*0.01;
      setCamera(cam=>({
        ...cam,
        orbitRadius:Math.max(0.01, cam.orbitRadius + d)
      }));
    }
  },[]);

  /******************************************************
   * G) Lifecycle & Render-on-demand
   ******************************************************/
  // Init once
  useEffect(()=>{
    initWebGPU();
    return () => {
      // On unmount, wait for GPU to be idle before cleanup
      if (gpuRef.current) {
        const { device } = gpuRef.current;

        // Wait for GPU to complete all work
        device.queue.onSubmittedWorkDone().then(() => {
          // Now safe to cleanup resources
          if (gpuRef.current) {
            gpuRef.current.depthTexture?.destroy();
            gpuRef.current.pickTexture?.destroy();
            gpuRef.current.pickDepthTexture?.destroy();
            gpuRef.current.readbackBuffer.destroy();
            gpuRef.current.uniformBuffer.destroy();

            // Clear the cache
            pipelineCache.clear();

            // Destroy geometry resources
            CommonResources.sphereGeo?.vb.destroy();
            CommonResources.sphereGeo?.ib.destroy();
            CommonResources.ringGeo?.vb.destroy();
            CommonResources.ringGeo?.ib.destroy();
            CommonResources.billboardQuad?.vb.destroy();
            CommonResources.billboardQuad?.ib.destroy();
            CommonResources.cubeGeo?.vb.destroy();
            CommonResources.cubeGeo?.ib.destroy();

            // Reset all references
            CommonResources.currentDevice = null;
            CommonResources.sphereGeo = null;
            CommonResources.ringGeo = null;
            CommonResources.billboardQuad = null;
            CommonResources.cubeGeo = null;
          }
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
  },[isReady, canvasWidth, canvasHeight, createOrUpdateDepthTexture, createOrUpdatePickTextures]);

  // Set actual canvas size
  useEffect(()=>{
    if(canvasRef.current){
      canvasRef.current.width=canvasWidth;
      canvasRef.current.height=canvasHeight;
    }
  },[canvasWidth, canvasHeight]);

  // Whenever elements change, or we become ready, rebuild GPU buffers
  useEffect(()=>{
    if(isReady && gpuRef.current){
      const ros = buildRenderObjects(elements);
      gpuRef.current.renderObjects = ros;
      // After building new objects, render once
      renderFrame(camera);
    }
  },[isReady, elements, renderFrame, camera]);

  // Also re-render if the camera changes
  useEffect(()=>{
    if(isReady){
      renderFrame(camera);
    }
  },[isReady, camera, renderFrame]);

  // Wheel handling
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const handleWheel = (e: WheelEvent) => {
      if (mouseState.current.type === 'idle') {
        e.preventDefault();
        const d = e.deltaY * 0.01;
        setCamera(cam => ({
          ...cam,
          orbitRadius: Math.max(0.01, cam.orbitRadius + d)
        }));
      }
    };
    canvas.addEventListener('wheel', handleWheel, { passive: false });
    return () => {
      canvas.removeEventListener('wheel', handleWheel);
    };
  }, []);

  return (
    <div style={{width:'100%', border:'1px solid #ccc'}}>
      <canvas
        ref={canvasRef}
        style={{ width:'100%', height:canvasHeight }}
        onMouseMove={handleMouseMove}
        onMouseDown={handleMouseDown}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
        onWheel={onWheel}
      />
    </div>
  );
}

/******************************************************
 * 5) Example: App
 ******************************************************/
function generateSpherePointCloud(numPoints:number, radius:number){
  const positions=new Float32Array(numPoints*3);
  const colors=new Float32Array(numPoints*3);
  for(let i=0;i<numPoints;i++){
    const theta=Math.random()*2*Math.PI;
    const phi=Math.acos(2*Math.random()-1);
    const x=radius*Math.sin(phi)*Math.cos(theta);
    const y=radius*Math.sin(phi)*Math.sin(theta);
    const z=radius*Math.cos(phi);
    positions[i*3+0] = x;
    positions[i*3+1] = y;
    positions[i*3+2] = z;
    colors[i*3+0] = Math.random();
    colors[i*3+1] = Math.random();
    colors[i*3+2] = Math.random();
  }
  return {positions, colors};
}

const pcData = generateSpherePointCloud(500, 0.5);

// Example ellipsoid data
const numEllipsoids=3;
const eCenters=new Float32Array(numEllipsoids*3);
const eRadii=new Float32Array(numEllipsoids*3);
const eColors=new Float32Array(numEllipsoids*3);
// example "snowman"
eCenters.set([0,0,0.6, 0,0,0.3, 0,0,0]);
eRadii.set([0.1,0.1,0.1, 0.15,0.15,0.15, 0.2,0.2,0.2]);
eColors.set([1,1,1, 1,1,1, 1,1,1]);

const numBounds=2;
const boundCenters=new Float32Array(numBounds*3);
const boundRadii=new Float32Array(numBounds*3);
const boundColors=new Float32Array(numBounds*3);
boundCenters.set([0.8,0,0.3,  -0.8,0,0.4]);
boundRadii.set([0.2,0.1,0.1,  0.1,0.2,0.2]);
boundColors.set([0.7,0.3,0.3, 0.3,0.3,0.9]);

// Cuboid examples
const numCuboids1 = 3;
const cCenters1 = new Float32Array(numCuboids1*3);
const cSizes1   = new Float32Array(numCuboids1*3);
const cColors1  = new Float32Array(numCuboids1*3);
// stack some cubes
cCenters1.set([1.0, 0, 0,   1.0, 0.3, 0,   1.0, 0.6, 0]);
cSizes1.set([0.2,0.2,0.2, 0.2,0.2,0.2, 0.2,0.2,0.2]);
cColors1.set([0.9,0.2,0.2, 0.2,0.9,0.2, 0.2,0.2,0.9]);

const numCuboids2 = 2;
const cCenters2 = new Float32Array(numCuboids2*3);
const cSizes2   = new Float32Array(numCuboids2*3);
const cColors2  = new Float32Array(numCuboids2*3);
cCenters2.set([-1.0,0,0, -1.0, 0.4, 0.4]);
cSizes2.set([0.3,0.1,0.2, 0.2,0.2,0.2]);
cColors2.set([0.8,0.4,0.6, 0.4,0.6,0.8]);

export function App(){
  const [hoveredIndices, setHoveredIndices] = useState({
    pc1: null as number | null,
    pc2: null as number | null,
    ellipsoid1: null as number | null,
    ellipsoid2: null as number | null,
    bounds1: null as number | null,
    bounds2: null as number | null,
    cuboid1: null as number | null,
    cuboid2: null as number | null,
  });

  const pcData1 = React.useMemo(() => generateSpherePointCloud(500, 0.5), []);
  const pcData2 = generateSpherePointCloud(300, 0.3);

  const pcElement1: PointCloudElementConfig = {
    type: 'PointCloud',
    data: pcData1,
    decorations: hoveredIndices.pc1 === null ? undefined : [
      {indexes: [hoveredIndices.pc1], color: [1, 1, 0], alpha: 1, minSize: 0.05}
    ],
    onHover: (i) => setHoveredIndices(prev => ({...prev, pc1: i})),
    onClick: (i) => console.log("Clicked point in cloud 1 #", i)
  };

  const pcElement2: PointCloudElementConfig = {
    type: 'PointCloud',
    data: pcData2,
    decorations: hoveredIndices.pc2 === null ? undefined : [
      {indexes: [hoveredIndices.pc2], color: [0, 1, 0], alpha: 1, minSize: 0.05}
    ],
    onHover: (i) => setHoveredIndices(prev => ({...prev, pc2: i})),
    onClick: (i) => console.log("Clicked point in cloud 2 #", i)
  };

  // Example Ellipsoids
  const eCenters1 = new Float32Array([0, 0, 0.6, 0, 0, 0.3, 0, 0, 0]);
  const eRadii1 =  new Float32Array([0.1,0.1,0.1, 0.15,0.15,0.15, 0.2,0.2,0.2]);
  const eColors1 = new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);

  const eCenters2 = new Float32Array([0.5,0.5,0.5, -0.5,-0.5,-0.5, 0.5,-0.5,0.5]);
  const eRadii2  = new Float32Array([0.2,0.2,0.2, 0.1,0.1,0.1, 0.15,0.15,0.15]);
  const eColors2 = new Float32Array([0,1,1, 1,0,1, 1,1,0]);

  const ellipsoidElement1: EllipsoidElementConfig = {
    type: 'Ellipsoid',
    data: {centers: eCenters1, radii: eRadii1, colors: eColors1},
    decorations: hoveredIndices.ellipsoid1 === null ? undefined : [
      {indexes: [hoveredIndices.ellipsoid1], color: [1, 0, 1], alpha: 0.8}
    ],
    onHover: (i) => setHoveredIndices(prev => ({...prev, ellipsoid1: i})),
    onClick: (i) => console.log("Clicked ellipsoid in group 1 #", i)
  };

  const ellipsoidElement2: EllipsoidElementConfig = {
    type: 'Ellipsoid',
    data: {centers: eCenters2, radii: eRadii2, colors: eColors2},
    decorations: hoveredIndices.ellipsoid2 === null ? undefined : [
      {indexes: [hoveredIndices.ellipsoid2], color: [0, 1, 0], alpha: 0.8}
    ],
    onHover: (i) => setHoveredIndices(prev => ({...prev, ellipsoid2: i})),
    onClick: (i) => console.log("Clicked ellipsoid in group 2 #", i)
  };

  // Ellipsoid Bounds
  const boundsElement1: EllipsoidBoundsElementConfig = {
    type: 'EllipsoidBounds',
    data: {centers: boundCenters, radii: boundRadii, colors: boundColors},
    decorations: hoveredIndices.bounds1 === null ? undefined : [
      {indexes: [hoveredIndices.bounds1], color: [0, 1, 1], alpha: 1}
    ],
    onHover: (i) => setHoveredIndices(prev => ({...prev, bounds1: i})),
    onClick: (i) => console.log("Clicked bounds in group 1 #", i)
  };

  const boundsElement2: EllipsoidBoundsElementConfig = {
    type: 'EllipsoidBounds',
    data: {
      centers: new Float32Array([0.2,0.2,0.2, -0.2,-0.2,-0.2]),
      radii: new Float32Array([0.15,0.15,0.15, 0.1,0.1,0.1]),
      colors: new Float32Array([0.5,0.5,0.5, 0.2,0.2,0.2])
    },
    decorations: hoveredIndices.bounds2 === null ? undefined : [
      {indexes: [hoveredIndices.bounds2], color: [1, 0, 0], alpha: 1}
    ],
    onHover: (i) => setHoveredIndices(prev => ({...prev, bounds2: i})),
    onClick: (i) => console.log("Clicked bounds in group 2 #", i)
  };

  // Cuboid Examples
  const cuboidElement1: CuboidElementConfig = {
    type: 'Cuboid',
    data: { centers: cCenters1, sizes: cSizes1, colors: cColors1 },
    decorations: hoveredIndices.cuboid1 === null ? undefined : [
      {indexes: [hoveredIndices.cuboid1], color: [1,1,0], alpha: 1}
    ],
    onHover: (i) => setHoveredIndices(prev => ({...prev, cuboid1: i})),
    onClick: (i) => console.log("Clicked cuboid in group 1 #", i)
  };

  const cuboidElement2: CuboidElementConfig = {
    type: 'Cuboid',
    data: { centers: cCenters2, sizes: cSizes2, colors: cColors2 },
    decorations: hoveredIndices.cuboid2 === null ? undefined : [
      {indexes: [hoveredIndices.cuboid2], color: [1,0,1], alpha: 1}
    ],
    onHover: (i) => setHoveredIndices(prev => ({...prev, cuboid2: i})),
    onClick: (i) => console.log("Clicked cuboid in group 2 #", i)
  };

  const elements: SceneElementConfig[] = [
    pcElement1,
    pcElement2,
    ellipsoidElement1,
    ellipsoidElement2,
    boundsElement1,
    boundsElement2,
    cuboidElement1,
    cuboidElement2
  ];

  return <SceneWrapper elements={elements} />;
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
