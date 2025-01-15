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

const GEOMETRY = {
  SPHERE: {
    STACKS: 16,
    SLICES: 24,
    MIN_STACKS: 8,
    MIN_SLICES: 12,
  }
} as const;

/******************************************************
 * 1) Data Structures
 ******************************************************/
interface PointCloudData {
  positions: Float32Array;
  colors?: Float32Array;
  scales?: Float32Array;
}

interface EllipsoidData {
  centers: Float32Array;
  radii: Float32Array;
  colors?: Float32Array;
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
  onHover?: (index: number|null) => void;
  onClick?: (index: number) => void;
}

interface EllipsoidElementConfig {
  type: 'Ellipsoid';
  data: EllipsoidData;
  decorations?: Decoration[];
  onHover?: (index: number|null) => void;
  onClick?: (index: number) => void;
}

interface EllipsoidBoundsElementConfig {
  type: 'EllipsoidBounds';
  data: EllipsoidData;
  decorations?: Decoration[];
  onHover?: (index: number|null) => void;
  onClick?: (index: number) => void;
}

interface LineData {
  segments: Float32Array;
  thickness: number;
}
interface LineElement {
  type: 'Lines';
  data: LineData;
  onHover?: (index: number|null) => void;
  onClick?: (index: number) => void;
}

export type SceneElementConfig =
  | PointCloudElementConfig
  | EllipsoidElementConfig
  | EllipsoidBoundsElementConfig
  | LineElement;

/******************************************************
 * 2) Camera State
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
 * 3) React Props
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

  /******************************************************
   * 5a) GPU references
   ******************************************************/
  const gpuRef = useRef<{
    device: GPUDevice;
    context: GPUCanvasContext;

    // main pipelines
    billboardPipeline: GPURenderPipeline;
    billboardQuadVB: GPUBuffer;
    billboardQuadIB: GPUBuffer;
    ellipsoidPipeline: GPURenderPipeline;
    sphereVB: GPUBuffer;
    sphereIB: GPUBuffer;
    sphereIndexCount: number;
    ellipsoidBandPipeline: GPURenderPipeline;
    ringVB: GPUBuffer;
    ringIB: GPUBuffer;
    ringIndexCount: number;

    // uniform
    uniformBuffer: GPUBuffer;
    uniformBindGroup: GPUBindGroup;

    // instance data
    pcInstanceBuffer: GPUBuffer | null;
    pcInstanceCount: number;
    ellipsoidInstanceBuffer: GPUBuffer | null;
    ellipsoidInstanceCount: number;
    bandInstanceBuffer: GPUBuffer | null;
    bandInstanceCount: number;

    // Depth
    depthTexture: GPUTexture | null;

    // [GPU PICKING ADDED]
    pickTexture: GPUTexture | null;
    pickDepthTexture: GPUTexture | null;
    pickPipelineBillboard: GPURenderPipeline;
    pickPipelineEllipsoid: GPURenderPipeline;
    pickPipelineBands: GPURenderPipeline;
    readbackBuffer: GPUBuffer;

    // ID mapping
    idToElement: { elementIdx: number; instanceIdx: number }[];
    elementBaseId: number[];
  } | null>(null);

  const rafIdRef = useRef<number>(0);
  const [isReady, setIsReady] = useState(false);

  // Camera
  const [camera, setCamera] = useState<CameraState>({
    orbitRadius: 1.5,
    orbitTheta: 0.2,
    orbitPhi: 1.0,
    panX: 0,
    panY: 0,
    fov: Math.PI / 3,
    near: 0.01,
    far: 100.0,
  });

  // Add at the top of the Scene component:
  const pickingLockRef = useRef(false);

  /******************************************************
   * A) Minimal Math
   ******************************************************/
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
    const f = 1.0 / Math.tan(fov / 2);
    out[0] = f / aspect;
    out[5] = f;
    out[10] = (far + near) / (near - far);
    out[11] = -1;
    out[14] = (2*far*near)/(near-far);
    return out;
  }

  function mat4LookAt(eye:[number, number, number], target:[number, number, number], up:[number, number, number]) {
    const zAxis = normalize([eye[0]-target[0], eye[1]-target[1], eye[2]-target[2]]);
    const xAxis = normalize(cross(up, zAxis));
    const yAxis = cross(zAxis, xAxis);
    const out = new Float32Array(16);
    out[0]=xAxis[0];  out[1]=yAxis[0];  out[2]=zAxis[0];  out[3]=0;
    out[4]=xAxis[1];  out[5]=yAxis[1];  out[6]=zAxis[1];  out[7]=0;
    out[8]=xAxis[2];  out[9]=yAxis[2];  out[10]=zAxis[2]; out[11]=0;
    out[12]=-dot(xAxis, eye);
    out[13]=-dot(yAxis, eye);
    out[14]=-dot(zAxis, eye);
    out[15]=1;
    return out;
  }

  function cross(a:[number, number, number], b:[number, number, number]) {
    return [
      a[1]*b[2] - a[2]*b[1],
      a[2]*b[0] - a[0]*b[2],
      a[0]*b[1] - a[1]*b[0]
    ] as [number, number, number];
  }

  function dot(a:[number, number, number], b:[number, number, number]) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
  }

  function normalize(v:[number, number, number]) {
    const len=Math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
    if(len>1e-6) {
      return [v[0]/len, v[1]/len, v[2]/len] as [number, number, number];
    }
    return [0,0,0];
  }

  /******************************************************
   * B) Create Sphere + Torus Geometry
   ******************************************************/
  function createSphereGeometry() {
    // typical logic creating sphere => pos+normal
    const stacks=GEOMETRY.SPHERE.STACKS, slices=GEOMETRY.SPHERE.SLICES;
    const verts:number[]=[];
    const idxs:number[]=[];
    for(let i=0;i<=stacks;i++){
      const phi=(i/stacks)*Math.PI;
      const sp=Math.sin(phi), cp=Math.cos(phi);
      for(let j=0;j<=slices;j++){
        const theta=(j/slices)*2*Math.PI;
        const st=Math.sin(theta), ct=Math.cos(theta);
        const x=sp*ct, y=cp, z=sp*st;
        verts.push(x,y,z, x,y,z); // pos, normal
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
      vertexData:new Float32Array(verts),
      indexData:new Uint16Array(idxs)
    };
  }

  function createTorusGeometry(majorRadius:number, minorRadius:number, majorSegments:number, minorSegments:number) {
    // typical torus geometry => pos+normal
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
        const a=row1+i;
        const b=row1+i+1;
        const c=row2+i;
        const d=row2+i+1;
        idxs.push(a,b,c, b,d,c);
      }
    }
    return {
      vertexData:new Float32Array(verts),
      indexData:new Uint16Array(idxs)
    };
  }

  /******************************************************
   * C) Initialize WebGPU
   ******************************************************/
  const initWebGPU=useCallback(async()=>{
    if(!canvasRef.current) return;
    if(!navigator.gpu){
      console.error("WebGPU not supported.");
      return;
    }
    try {
      const adapter=await navigator.gpu.requestAdapter();
      if(!adapter) throw new Error("Failed to get GPU adapter");
      const device=await adapter.requestDevice();
      const context=canvasRef.current.getContext('webgpu') as GPUCanvasContext;
      const format=navigator.gpu.getPreferredCanvasFormat();
      context.configure({device, format, alphaMode:'premultiplied'});

      // create geometry...
      const sphereGeo=createSphereGeometry();
      const sphereVB=device.createBuffer({
        size:sphereGeo.vertexData.byteLength,
        usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST
      });
      device.queue.writeBuffer(sphereVB,0,sphereGeo.vertexData);
      const sphereIB=device.createBuffer({
        size:sphereGeo.indexData.byteLength,
        usage:GPUBufferUsage.INDEX|GPUBufferUsage.COPY_DST
      });
      device.queue.writeBuffer(sphereIB,0,sphereGeo.indexData);

      const torusGeo=createTorusGeometry(1.0,0.03,40,12);
      const ringVB=device.createBuffer({
        size:torusGeo.vertexData.byteLength,
        usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST
      });
      device.queue.writeBuffer(ringVB,0,torusGeo.vertexData);
      const ringIB=device.createBuffer({
        size:torusGeo.indexData.byteLength,
        usage:GPUBufferUsage.INDEX|GPUBufferUsage.COPY_DST
      });
      device.queue.writeBuffer(ringIB,0,torusGeo.indexData);

      // billboard
      const QUAD_VERTS=new Float32Array([-0.5,-0.5, 0.5,-0.5, -0.5,0.5, 0.5,0.5]);
      const QUAD_IDX=new Uint16Array([0,1,2,2,1,3]);
      const billboardQuadVB=device.createBuffer({
        size:QUAD_VERTS.byteLength,
        usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST
      });
      device.queue.writeBuffer(billboardQuadVB,0,QUAD_VERTS);
      const billboardQuadIB=device.createBuffer({
        size:QUAD_IDX.byteLength,
        usage:GPUBufferUsage.INDEX|GPUBufferUsage.COPY_DST
      });
      device.queue.writeBuffer(billboardQuadIB,0,QUAD_IDX);

      // uniform
      const uniformBufferSize=128;
      const uniformBuffer=device.createBuffer({
        size:uniformBufferSize,
        usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST
      });
      const uniformBindGroupLayout=device.createBindGroupLayout({
        entries:[{
          binding:0,
          visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,
          buffer:{type:'uniform'}
        }]
      });
      const pipelineLayout=device.createPipelineLayout({
        bindGroupLayouts:[uniformBindGroupLayout]
      });
      const uniformBindGroup=device.createBindGroup({
        layout:uniformBindGroupLayout,
        entries:[{binding:0, resource:{buffer:uniformBuffer}}]
      });

      // pipeline for billboards (with shading)
      const billboardPipeline=device.createRenderPipeline({
        layout:pipelineLayout,
        vertex:{
          module:device.createShaderModule({
            code: `
struct Camera {
  mvp: mat4x4<f32>,
  cameraRight: vec3<f32>,
  pad1: f32,
  cameraUp: vec3<f32>,
  pad2: f32,
  lightDir: vec3<f32>,
  pad3: f32,
};
@group(0) @binding(0) var<uniform> camera : Camera;

struct VSOut {
  @builtin(position) Position: vec4<f32>,
  @location(0) color: vec3<f32>,
  @location(1) alpha: f32,
};

@vertex
fn vs_main(
  @location(0) corner : vec2<f32>,
  @location(1) pos : vec3<f32>,
  @location(2) col : vec3<f32>,
  @location(3) alpha : f32,
  @location(4) scaleX: f32,
  @location(5) scaleY: f32
)->VSOut {
  let offset=camera.cameraRight*(corner.x*scaleX) + camera.cameraUp*(corner.y*scaleY);
  let worldPos=vec4<f32>(pos+offset,1.0);
  var outData:VSOut;
  outData.Position=camera.mvp*worldPos;
  outData.color=col;
  outData.alpha=alpha;
  return outData;
}

@fragment
fn fs_main(
  @location(0) color: vec3<f32>,
  @location(1) alpha: f32
)->@location(0) vec4<f32>{
  return vec4<f32>(color, alpha);
}`
          }),
          entryPoint:'vs_main',
          buffers:[
            {
              arrayStride:2*4,
              attributes:[{shaderLocation:0, offset:0, format:'float32x2'}]
            },
            {
              arrayStride:9*4,
              stepMode:'instance',
              attributes:[
                {shaderLocation:1, offset:0, format:'float32x3'},
                {shaderLocation:2, offset:3*4, format:'float32x3'},
                {shaderLocation:3, offset:6*4, format:'float32'},
                {shaderLocation:4, offset:7*4, format:'float32'},
                {shaderLocation:5, offset:8*4, format:'float32'},
              ]
            }
          ]
        },
        fragment:{
          module:device.createShaderModule({
            code:`@fragment fn fs_main(@location(0) c: vec3<f32>, @location(1) a: f32)->@location(0) vec4<f32>{
  return vec4<f32>(c,a);
}`
          }),
          entryPoint:'fs_main',
          targets:[{
            format,
            blend:{
              color:{srcFactor:'src-alpha', dstFactor:'one-minus-src-alpha', operation:'add'},
              alpha:{srcFactor:'one', dstFactor:'one-minus-src-alpha', operation:'add'}
            }
          }]
        },
        primitive:{topology:'triangle-list', cullMode:'back'},
        depthStencil:{
          format:'depth24plus',
          depthWriteEnabled:true,
          depthCompare:'less-equal'
        }
      });

      // pipeline for ellipsoid shading
      const ellipsoidPipeline=device.createRenderPipeline({
        layout:pipelineLayout,
        vertex:{
          module:device.createShaderModule({
            code:`
struct Camera {
  mvp: mat4x4<f32>,
  cameraRight: vec3<f32>,
  pad1: f32,
  cameraUp: vec3<f32>,
  pad2: f32,
  lightDir: vec3<f32>,
  pad3: f32,
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
  @location(0) inPos : vec3<f32>,
  @location(1) inNorm: vec3<f32>,
  @location(2) iPos: vec3<f32>,
  @location(3) iScale: vec3<f32>,
  @location(4) iColor: vec3<f32>,
  @location(5) iAlpha: f32
)->VSOut {
  let worldPos = iPos + (inPos * iScale);
  let scaledNorm = normalize(inNorm / iScale);

  var outData:VSOut;
  outData.pos=camera.mvp*vec4<f32>(worldPos,1.0);
  outData.normal=scaledNorm;
  outData.color=iColor;
  outData.alpha=iAlpha;
  outData.worldPos=worldPos;
  return outData;
}`
          }),
          entryPoint:'vs_main',
          buffers:[
            {
              // Vertex buffer - positions and normals
              arrayStride:6*4,
              attributes:[
                {shaderLocation:0, offset:0, format:'float32x3'},  // inPos
                {shaderLocation:1, offset:3*4, format:'float32x3'} // inNorm
              ]
            },
            {
              // Instance buffer - position, scale, color, alpha
              arrayStride:10*4,
              stepMode:'instance',
              attributes:[
                {shaderLocation:2, offset:0*4, format:'float32x3'},  // iPos
                {shaderLocation:3, offset:3*4, format:'float32x3'},  // iScale
                {shaderLocation:4, offset:6*4, format:'float32x3'},  // iColor
                {shaderLocation:5, offset:9*4, format:'float32'}     // iAlpha
              ]
            }
          ]
        },
        fragment:{
          module:device.createShaderModule({
            code:`
struct Camera {
  mvp: mat4x4<f32>,
  cameraRight: vec3<f32>,
  pad1: f32,
  cameraUp: vec3<f32>,
  pad2: f32,
  lightDir: vec3<f32>,
  pad3: f32,
};
@group(0) @binding(0) var<uniform> camera : Camera;

@fragment
fn fs_main(
  @location(1) normal: vec3<f32>,
  @location(2) baseColor: vec3<f32>,
  @location(3) alpha: f32,
  @location(4) worldPos: vec3<f32>
)->@location(0) vec4<f32> {
  let N=normalize(normal);
  let L=normalize(camera.lightDir);
  let lambert=max(dot(N,L),0.0);

  let ambient=${LIGHTING.AMBIENT_INTENSITY};
  var color=baseColor*(ambient+lambert*${LIGHTING.DIFFUSE_INTENSITY});

  let V=normalize(-worldPos);
  let H=normalize(L+V);
  let spec=pow(max(dot(N,H),0.0), ${LIGHTING.SPECULAR_POWER});
  color+=vec3<f32>(1.0,1.0,1.0)*spec*${LIGHTING.SPECULAR_INTENSITY};

  return vec4<f32>(color, alpha);
}`
          }),
          entryPoint:'fs_main',
          targets:[{
            format,
            blend:{
              color:{srcFactor:'src-alpha', dstFactor:'one-minus-src-alpha', operation:'add'},
              alpha:{srcFactor:'one', dstFactor:'one-minus-src-alpha', operation:'add'}
            }
          }]
        },
        primitive:{topology:'triangle-list', cullMode:'back'},
        depthStencil:{
          format:'depth24plus',
          depthWriteEnabled:true,
          depthCompare:'less-equal'
        }
      });

      // pipeline for "bounds" torus
      const ellipsoidBandPipeline=device.createRenderPipeline({
        layout:pipelineLayout,
        vertex:{
          module:device.createShaderModule({
            code:`
struct Camera {
  mvp: mat4x4<f32>,
  cameraRight: vec3<f32>,
  pad1: f32,
  cameraUp: vec3<f32>,
  pad2: f32,
  lightDir: vec3<f32>,
  pad3: f32,
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
  @builtin(instance_index) instIdx:u32,
  @location(0) inPos: vec3<f32>,
  @location(1) inNorm: vec3<f32>,
  @location(2) iCenter: vec3<f32>,
  @location(3) iScale: vec3<f32>,
  @location(4) iColor: vec3<f32>,
  @location(5) iAlpha: f32
)->VSOut {
  let ringIndex=i32(instIdx%3u);
  var lp=inPos;
  var ln=inNorm;

  if(ringIndex==1){
    let px=lp.x; lp.x=lp.y; lp.y=-px;
    let nx=ln.x; ln.x=ln.y; ln.y=-nx;
  } else if(ringIndex==2){
    let pz=lp.z; lp.z=-lp.x; lp.x=pz;
    let nz=ln.z; ln.z=-ln.x; ln.x=nz;
  }
  lp*=iScale;
  ln=normalize(ln/iScale);
  let wp=lp+iCenter;

  var outData:VSOut;
  outData.pos=camera.mvp*vec4<f32>(wp,1.0);
  outData.normal=ln;
  outData.color=iColor;
  outData.alpha=iAlpha;
  outData.worldPos=wp;
  return outData;
}

@fragment
fn fs_main(
  @location(1) n: vec3<f32>,
  @location(2) baseColor: vec3<f32>,
  @location(3) alpha: f32,
  @location(4) wp: vec3<f32>
)->@location(0) vec4<f32>{
  // same shading logic
  return vec4<f32>(baseColor,alpha);
}`
          }),
          entryPoint:'vs_main',
          buffers:[
            {
              arrayStride:6*4,
              attributes:[
                {shaderLocation:0, offset:0, format:'float32x3'},
                {shaderLocation:1, offset:3*4, format:'float32x3'}
              ]
            },
            {
              arrayStride:10*4,
              stepMode:'instance',
              attributes:[
                {shaderLocation:2, offset:0, format:'float32x3'},
                {shaderLocation:3, offset:3*4, format:'float32x3'},
                {shaderLocation:4, offset:6*4, format:'float32x3'},
                {shaderLocation:5, offset:9*4, format:'float32'}
              ]
            }
          ]
        },
        fragment:{
          module:device.createShaderModule({
            code:`@fragment fn fs_main(
  @location(1) n:vec3<f32>,
  @location(2) col:vec3<f32>,
  @location(3) a:f32,
  @location(4) wp:vec3<f32>
)->@location(0) vec4<f32>{
  return vec4<f32>(col,a);
}`
          }),
          entryPoint:'fs_main',
          targets:[{
            format,
            blend:{
              color:{srcFactor:'src-alpha', dstFactor:'one-minus-src-alpha', operation:'add'},
              alpha:{srcFactor:'one', dstFactor:'one-minus-src-alpha', operation:'add'}
            }
          }]
        },
        primitive:{topology:'triangle-list', cullMode:'back'},
        depthStencil:{format:'depth24plus', depthWriteEnabled:true, depthCompare:'less-equal'}
      });

      /******************************************************
       * [GPU PICKING ADDED] - create pipelines for "ID color"
       ******************************************************/
      const pickVert = device.createShaderModule({
        code: `
struct Camera {
  mvp: mat4x4<f32>,
  cameraRight: vec3<f32>,
  pad1: f32,
  cameraUp: vec3<f32>,
  pad2: f32,
  lightDir: vec3<f32>,
  pad3: f32,
};
@group(0) @binding(0) var<uniform> camera : Camera;

struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) pickID: f32,
};

@vertex
fn vs_pc(
  @location(0) corner: vec2<f32>,
  @location(1) pos: vec3<f32>,
  @location(2) pickID: f32,
  @location(3) scaleX: f32,
  @location(4) scaleY: f32
)->VSOut {
  let offset = camera.cameraRight*(corner.x*scaleX) + camera.cameraUp*(corner.y*scaleY);
  let worldPos = vec4<f32>(pos + offset, 1.0);
  var outData: VSOut;
  outData.pos = camera.mvp*worldPos;
  outData.pickID = pickID;
  return outData;
}

@vertex
fn vs_ellipsoid(
  @location(0) inPos: vec3<f32>,
  @location(1) inNorm: vec3<f32>,
  @location(2) iPos: vec3<f32>,
  @location(3) iScale: vec3<f32>,
  @location(4) pickID: f32
)->VSOut {
  let worldPos = iPos + (inPos * iScale);
  var outData: VSOut;
  outData.pos = camera.mvp*vec4<f32>(worldPos, 1.0);
  outData.pickID = pickID;
  return outData;
}

@vertex
fn vs_bands(
  @builtin(instance_index) instIdx: u32,
  @location(0) inPos: vec3<f32>,
  @location(1) inNorm: vec3<f32>,
  @location(2) center: vec3<f32>,
  @location(3) scale: vec3<f32>,
  @location(4) pickID: f32
)->VSOut {
  let ringIndex = i32(instIdx%3u);
  var lp = inPos;
  if(ringIndex == 1) {
    let px = lp.x; lp.x = lp.y; lp.y = -px;
  } else if(ringIndex == 2) {
    let pz = lp.z; lp.z = -lp.x; lp.x = pz;
  }
  lp *= scale;
  let wp = lp + center;
  var outData: VSOut;
  outData.pos = camera.mvp*vec4<f32>(wp, 1.0);
  outData.pickID = pickID;
  return outData;
}

@fragment
fn fs_pick(@location(0) pickID: f32)->@location(0) vec4<f32> {
  let iID = u32(pickID);
  let r = f32(iID & 255u)/255.0;
  let g = f32((iID>>8)&255u)/255.0;
  let b = f32((iID>>16)&255u)/255.0;
  return vec4<f32>(r,g,b,1.0);
}`
      });

      const pickPipelineBillboard=device.createRenderPipeline({
        layout:pipelineLayout,
        vertex:{
          module: pickVert,
          entryPoint:'vs_pc',
          buffers:[
            {
              arrayStride:2*4,
              attributes:[{shaderLocation:0, offset:0, format:'float32x2'}]
            },
            {
              arrayStride:6*4,
              stepMode:'instance',
              attributes:[
                {shaderLocation:1, offset:0, format:'float32x3'},
                {shaderLocation:2, offset:3*4, format:'float32'},
                {shaderLocation:3, offset:4*4, format:'float32'},
                {shaderLocation:4, offset:5*4, format:'float32'},
              ]
            }
          ]
        },
        fragment:{
          module: pickVert,
          entryPoint:'fs_pick',
          targets:[{ format:'rgba8unorm' }]
        },
        primitive:{topology:'triangle-list', cullMode:'back'},
        depthStencil:{format:'depth24plus', depthWriteEnabled:true, depthCompare:'less-equal'}
      });

      const pickPipelineEllipsoid=device.createRenderPipeline({
        layout:pipelineLayout,
        vertex:{
          module: pickVert,
          entryPoint:'vs_ellipsoid',
          buffers:[
            {
              // Vertex buffer - positions and normals
              arrayStride:6*4,
              attributes:[
                {shaderLocation:0, offset:0, format:'float32x3'},  // inPos
                {shaderLocation:1, offset:3*4, format:'float32x3'} // inNorm
              ]
            },
            {
              // Instance buffer - pos, scale, pickID
              arrayStride:7*4,  // Changed from 10*4 to 7*4
              stepMode:'instance',
              attributes:[
                {shaderLocation:2, offset:0, format:'float32x3'},  // iPos
                {shaderLocation:3, offset:3*4, format:'float32x3'}, // iScale
                {shaderLocation:4, offset:6*4, format:'float32'}    // pickID
              ]
            }
          ]
        },
        fragment:{
          module: pickVert,
          entryPoint:'fs_pick',
          targets:[{format:'rgba8unorm'}]
        },
        primitive:{topology:'triangle-list', cullMode:'back'},
        depthStencil:{format:'depth24plus', depthWriteEnabled:true, depthCompare:'less-equal'}
      });

      const pickPipelineBands=device.createRenderPipeline({
        layout:pipelineLayout,
        vertex:{
          module: pickVert,
          entryPoint:'vs_bands',
          buffers:[
            {
              // Vertex buffer - positions and normals
              arrayStride:6*4,
              attributes:[
                {shaderLocation:0, offset:0, format:'float32x3'},  // inPos
                {shaderLocation:1, offset:3*4, format:'float32x3'} // inNorm
              ]
            },
            {
              // Instance buffer - pos, scale, pickID
              arrayStride:7*4,  // Changed from 10*4 to 7*4
              stepMode:'instance',
              attributes:[
                {shaderLocation:2, offset:0, format:'float32x3'},  // iPos
                {shaderLocation:3, offset:3*4, format:'float32x3'}, // iScale
                {shaderLocation:4, offset:6*4, format:'float32'}    // pickID
              ]
            }
          ]
        },
        fragment:{
          module: pickVert,
          entryPoint:'fs_pick',
          targets:[{ format:'rgba8unorm'}]
        },
        primitive:{topology:'triangle-list', cullMode:'back'},
        depthStencil:{format:'depth24plus', depthWriteEnabled:true, depthCompare:'less-equal'}
      });

      // store
      gpuRef.current={
        device, context,
        billboardPipeline, billboardQuadVB, billboardQuadIB,
        ellipsoidPipeline, sphereVB, sphereIB,
        sphereIndexCount:sphereGeo.indexData.length,
        ellipsoidBandPipeline, ringVB, ringIB,
        ringIndexCount:torusGeo.indexData.length,
        uniformBuffer, uniformBindGroup,

        pcInstanceBuffer:null,
        pcInstanceCount:0,
        ellipsoidInstanceBuffer:null,
        ellipsoidInstanceCount:0,
        bandInstanceBuffer:null,
        bandInstanceCount:0,
        depthTexture:null,

        pickTexture:null,
        pickDepthTexture:null,
        pickPipelineBillboard,
        pickPipelineEllipsoid,
        pickPipelineBands,

        readbackBuffer: device.createBuffer({
          size: 256,  // Changed from 4 to 256
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        }),
        idToElement:[],
        elementBaseId:[]
      };

      setIsReady(true);
    } catch(err){
      console.error("Error init WebGPU:",err);
    }
  },[]);

  /******************************************************
   * D) Depth texture
   ******************************************************/
  const createOrUpdateDepthTexture=useCallback(()=>{
    if(!gpuRef.current) return;
    const {device, depthTexture}=gpuRef.current;
    if(depthTexture) depthTexture.destroy();
    const dt=device.createTexture({
      size:[canvasWidth, canvasHeight],
      format:'depth24plus',
      usage:GPUTextureUsage.RENDER_ATTACHMENT
    });
    gpuRef.current.depthTexture=dt;
  },[canvasWidth, canvasHeight]);

  // [GPU PICKING ADDED] => color+depth for picking
  const createOrUpdatePickTextures=useCallback(()=>{
    if(!gpuRef.current)return;
    const {device, pickTexture, pickDepthTexture}=gpuRef.current;
    if(pickTexture) pickTexture.destroy();
    if(pickDepthTexture) pickDepthTexture.destroy();
    const colorTex=device.createTexture({
      size:[canvasWidth,canvasHeight],
      format:'rgba8unorm',
      usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.COPY_SRC
    });
    const depthTex=device.createTexture({
      size:[canvasWidth,canvasHeight],
      format:'depth24plus',
      usage:GPUTextureUsage.RENDER_ATTACHMENT
    });
    gpuRef.current.pickTexture=colorTex;
    gpuRef.current.pickDepthTexture=depthTex;
  },[canvasWidth, canvasHeight]);

  /******************************************************
   * E) Render loop
   ******************************************************/
  const renderFrame=useCallback(()=>{
    if(rafIdRef.current) cancelAnimationFrame(rafIdRef.current);
    if(!gpuRef.current){
      rafIdRef.current=requestAnimationFrame(renderFrame);
      return;
    }
    const {
      device, context,
      billboardPipeline, billboardQuadVB, billboardQuadIB,
      ellipsoidPipeline, sphereVB, sphereIB, sphereIndexCount,
      ellipsoidBandPipeline, ringVB, ringIB, ringIndexCount,
      uniformBuffer, uniformBindGroup,
      pcInstanceBuffer, pcInstanceCount,
      ellipsoidInstanceBuffer, ellipsoidInstanceCount,
      bandInstanceBuffer, bandInstanceCount,
      depthTexture
    }=gpuRef.current;
    if(!depthTexture){
      rafIdRef.current=requestAnimationFrame(renderFrame);
      return;
    }

    // camera MVP
    const aspect=canvasWidth/canvasHeight;
    const proj=mat4Perspective(camera.fov,aspect,camera.near,camera.far);
    const cx=camera.orbitRadius*Math.sin(camera.orbitPhi)*Math.sin(camera.orbitTheta);
    const cy=camera.orbitRadius*Math.cos(camera.orbitPhi);
    const cz=camera.orbitRadius*Math.sin(camera.orbitPhi)*Math.cos(camera.orbitTheta);
    const eye:[number,number,number]=[cx,cy,cz];
    const target:[number,number,number]=[camera.panX,camera.panY,0];
    const up:[number,number,number]=[0,1,0];
    const view=mat4LookAt(eye,target,up);
    const forward=normalize([target[0]-eye[0], target[1]-eye[1], target[2]-eye[2]]);
    const cameraRight=normalize(cross(forward,up));
    const cameraUp=cross(cameraRight,forward);
    const mvp=mat4Multiply(proj,view);

    // uniform
    const data=new Float32Array(32);
    data.set(mvp,0);
    data[16]=cameraRight[0];
    data[17]=cameraRight[1];
    data[18]=cameraRight[2];
    data[20]=cameraUp[0];
    data[21]=cameraUp[1];
    data[22]=cameraUp[2];

    // light dir
    const light=normalize([
      cameraRight[0]*LIGHTING.DIRECTION.RIGHT + cameraUp[0]*LIGHTING.DIRECTION.UP + forward[0]*LIGHTING.DIRECTION.FORWARD,
      cameraRight[1]*LIGHTING.DIRECTION.RIGHT + cameraUp[1]*LIGHTING.DIRECTION.UP + forward[1]*LIGHTING.DIRECTION.FORWARD,
      cameraRight[2]*LIGHTING.DIRECTION.RIGHT + cameraUp[2]*LIGHTING.DIRECTION.UP + forward[2]*LIGHTING.DIRECTION.FORWARD
    ]);
    data[24]=light[0];
    data[25]=light[1];
    data[26]=light[2];
    device.queue.writeBuffer(uniformBuffer,0,data);

    // get swapchain
    let texture:GPUTexture;
    try {
      texture=context.getCurrentTexture();
    }catch(e){
      rafIdRef.current=requestAnimationFrame(renderFrame);
      return;
    }

    const passDesc: GPURenderPassDescriptor={
      colorAttachments:[{
        view: texture.createView(),
        clearValue:{r:0.15,g:0.15,b:0.15,a:1},
        loadOp:'clear',
        storeOp:'store'
      }],
      depthStencilAttachment:{
        view: depthTexture.createView(),
        depthClearValue:1.0,
        depthLoadOp:'clear',
        depthStoreOp:'store'
      }
    };
    const cmdEncoder=device.createCommandEncoder();
    const pass=cmdEncoder.beginRenderPass(passDesc);

    // draw
    if(pcInstanceBuffer&&pcInstanceCount>0){
      pass.setPipeline(billboardPipeline);
      pass.setBindGroup(0,uniformBindGroup);
      pass.setVertexBuffer(0,billboardQuadVB);
      pass.setIndexBuffer(billboardQuadIB,'uint16');
      pass.setVertexBuffer(1,pcInstanceBuffer);
      pass.drawIndexed(6, pcInstanceCount);
    }
    if(ellipsoidInstanceBuffer&&ellipsoidInstanceCount>0){
      pass.setPipeline(ellipsoidPipeline);
      pass.setBindGroup(0,uniformBindGroup);
      pass.setVertexBuffer(0,sphereVB);
      pass.setIndexBuffer(sphereIB,'uint16');
      pass.setVertexBuffer(1,ellipsoidInstanceBuffer);
      pass.drawIndexed(sphereIndexCount, ellipsoidInstanceCount);
    }
    if(bandInstanceBuffer&&bandInstanceCount>0){
      pass.setPipeline(ellipsoidBandPipeline);
      pass.setBindGroup(0,uniformBindGroup);
      pass.setVertexBuffer(0, ringVB);
      pass.setIndexBuffer(ringIB,'uint16');
      pass.setVertexBuffer(1, bandInstanceBuffer);
      pass.drawIndexed(ringIndexCount, bandInstanceCount);
    }

    pass.end();
    device.queue.submit([cmdEncoder.finish()]);
    rafIdRef.current=requestAnimationFrame(renderFrame);
  },[camera, canvasWidth, canvasHeight]);

  /******************************************************
   * F) Building Instance Data (Missing from snippet)
   ******************************************************/

  function buildPCInstanceData(
    positions: Float32Array,
    colors?: Float32Array,
    scales?: Float32Array,
    decorations?: Decoration[],
  ) {
    const count=positions.length/3;
    const data=new Float32Array(count*9);
    // (px,py,pz, r,g,b, alpha, scaleX, scaleY)
    for(let i=0;i<count;i++){
      data[i*9+0]=positions[i*3+0];
      data[i*9+1]=positions[i*3+1];
      data[i*9+2]=positions[i*3+2];
      if(colors&&colors.length===count*3){
        data[i*9+3]=colors[i*3+0];
        data[i*9+4]=colors[i*3+1];
        data[i*9+5]=colors[i*3+2];
      } else {
        data[i*9+3]=1; data[i*9+4]=1; data[i*9+5]=1;
      }
      data[i*9+6]=1.0; // alpha
      const s = scales? scales[i]: 0.02;
      data[i*9+7]=s;
      data[i*9+8]=s;
    }
    if(decorations){
      for(const dec of decorations){
        const {indexes, color, alpha, scale, minSize}=dec;
        for(const idx of indexes){
          if(idx<0||idx>=count) continue;
          if(color){
            data[idx*9+3]=color[0];
            data[idx*9+4]=color[1];
            data[idx*9+5]=color[2];
          }
          if(alpha!==undefined){
            data[idx*9+6]=alpha;
          }
          if(scale!==undefined){
            data[idx*9+7]*=scale;
            data[idx*9+8]*=scale;
          }
          if(minSize!==undefined){
            if(data[idx*9+7]<minSize) data[idx*9+7]=minSize;
            if(data[idx*9+8]<minSize) data[idx*9+8]=minSize;
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
    decorations?: Decoration[]
  ){
    const count=centers.length/3;
    const data=new Float32Array(count*10);
    // (pos.x,pos.y,pos.z, scale.x,scale.y,scale.z, col.r,col.g,col.b, alpha)
    for(let i=0;i<count;i++){
      data[i*10+0]=centers[i*3+0];
      data[i*10+1]=centers[i*3+1];
      data[i*10+2]=centers[i*3+2];

      data[i*10+3]=radii[i*3+0]||0.1;
      data[i*10+4]=radii[i*3+1]||0.1;
      data[i*10+5]=radii[i*3+2]||0.1;

      if(colors&&colors.length===count*3){
        data[i*10+6]=colors[i*3+0];
        data[i*10+7]=colors[i*3+1];
        data[i*10+8]=colors[i*3+2];
      } else {
        data[i*10+6]=1; data[i*10+7]=1; data[i*10+8]=1;
      }
      data[i*10+9]=1.0;
    }
    if(decorations){
      for(const dec of decorations){
        const {indexes, color, alpha, scale, minSize}=dec;
        for(const idx of indexes){
          if(idx<0||idx>=count) continue;
          if(color){
            data[idx*10+6]=color[0];
            data[idx*10+7]=color[1];
            data[idx*10+8]=color[2];
          }
          if(alpha!==undefined){
            data[idx*10+9]=alpha;
          }
          if(scale!==undefined){
            data[idx*10+3]*=scale;
            data[idx*10+4]*=scale;
            data[idx*10+5]*=scale;
          }
          if(minSize!==undefined){
            if(data[idx*10+3]<minSize) data[idx*10+3]=minSize;
            if(data[idx*10+4]<minSize) data[idx*10+4]=minSize;
            if(data[idx*10+5]<minSize) data[idx*10+5]=minSize;
          }
        }
      }
    }
    return data;
  }

  function buildEllipsoidBoundsInstanceData(
    centers: Float32Array,
    radii: Float32Array,
    colors?: Float32Array,
    decorations?: Decoration[]
  ){
    const count=centers.length/3;
    const ringCount=count*3;
    const data=new Float32Array(ringCount*10);
    for(let i=0;i<count;i++){
      const cx=centers[i*3+0], cy=centers[i*3+1], cz=centers[i*3+2];
      const rx=radii[i*3+0]||0.1, ry=radii[i*3+1]||0.1, rz=radii[i*3+2]||0.1;

      let cr=1, cg=1, cb=1;
      if(colors&&colors.length===count*3){
        cr=colors[i*3+0];
        cg=colors[i*3+1];
        cb=colors[i*3+2];
      }
      let alpha=1.0;

      // decorations
      if(decorations){
        for(const dec of decorations){
          if(dec.indexes.includes(i)){
            if(dec.color){
              cr=dec.color[0];
              cg=dec.color[1];
              cb=dec.color[2];
            }
            if(dec.alpha!==undefined){
              alpha=dec.alpha;
            }
          }
        }
      }
      // 3 rings
      for(let ring=0; ring<3; ring++){
        const idx=i*3+ring;
        data[idx*10+0]=cx;
        data[idx*10+1]=cy;
        data[idx*10+2]=cz;
        data[idx*10+3]=rx; data[idx*10+4]=ry; data[idx*10+5]=rz;
        data[idx*10+6]=cr; data[idx*10+7]=cg; data[idx*10+8]=cb;
        data[idx*10+9]=alpha;
      }
    }
    return data;
  }

  /******************************************************
   * G) Update Buffers
   ******************************************************/
  const pickPCVertexBufferRef=useRef<GPUBuffer|null>(null);
  const pickPCCountRef=useRef<number>(0);
  const pickEllipsoidVBRef=useRef<GPUBuffer|null>(null);
  const pickEllipsoidCountRef=useRef<number>(0);
  const pickBandsVBRef=useRef<GPUBuffer|null>(null);
  const pickBandsCountRef=useRef<number>(0);

  const updateBuffers=useCallback((sceneElements: SceneElementConfig[])=>{
    if(!gpuRef.current)return;
    const {device, idToElement, elementBaseId}=gpuRef.current;

    // 1) build normal instance data
    let pcInstData:Float32Array|null=null, pcCount=0;
    let elInstData:Float32Array|null=null, elCount=0;
    let bandInstData:Float32Array|null=null, bandCount=0;

    // 2) figure out ID ranges
    // start from 1 => ID=0 => "no object"
    let totalIDs=1;
    for(let e=0;e<sceneElements.length;e++){
      const elem=sceneElements[e];
      let count=0;
      if(elem.type==='PointCloud'){
        count=elem.data.positions.length/3;
      } else if(elem.type==='Ellipsoid'){
        count=elem.data.centers.length/3;
      } else if(elem.type==='EllipsoidBounds'){
        count=elem.data.centers.length/3;
      }
      elementBaseId[e]=totalIDs;
      totalIDs+=count;
    }
    idToElement.length=totalIDs;
    for(let i=0;i<idToElement.length;i++){
      idToElement[i]={elementIdx:-1, instanceIdx:-1};
    }

    // We'll store picking data
    let pickPCData:Float32Array|null=null;
    let pickEllipsoidData:Float32Array|null=null;
    let pickBandData:Float32Array|null=null;

    function buildPickingPCData(positions:Float32Array, scales?:Float32Array, baseID:number){
      const count=positions.length/3;
      // layout => pos(3), pickID(1), scaleX(1), scaleY(1) => total 6 floats
      const arr=new Float32Array(count*6);
      for(let i=0;i<count;i++){
        arr[i*6+0]=positions[i*3+0];
        arr[i*6+1]=positions[i*3+1];
        arr[i*6+2]=positions[i*3+2];
        const thisID=baseID+i;
        arr[i*6+3]=thisID;
        const s=scales?scales[i]:0.02;
        arr[i*6+4]=s;
        arr[i*6+5]=s;
        idToElement[thisID]={elementIdx:-1, instanceIdx:i};
      }
      return arr;
    }

    function buildPickingEllipsoidData(centers:Float32Array, radii:Float32Array, baseID:number){
      const count=centers.length/3;
      // layout => pos(3), scale(3), pickID(1) => 7 floats
      const arr=new Float32Array(count*7);
      for(let i=0;i<count;i++){
        arr[i*7+0]=centers[i*3+0];
        arr[i*7+1]=centers[i*3+1];
        arr[i*7+2]=centers[i*3+2];
        arr[i*7+3]=radii[i*3+0]||0.1;
        arr[i*7+4]=radii[i*3+1]||0.1;
        arr[i*7+5]=radii[i*3+2]||0.1;
        const thisID=baseID+i;
        arr[i*7+6]=thisID;
        idToElement[thisID]={elementIdx:-1, instanceIdx:i};
      }
      return arr;
    }

    function buildPickingBoundsData(centers:Float32Array, radii:Float32Array, baseID:number){
      const count=centers.length/3;
      const ringCount=count*3;
      const arr=new Float32Array(ringCount*7);
      for(let i=0;i<count;i++){
        const cx=centers[i*3+0], cy=centers[i*3+1], cz=centers[i*3+2];
        const rx=radii[i*3+0]||0.1, ry=radii[i*3+1]||0.1, rz=radii[i*3+2]||0.1;
        const thisID=baseID+i;
        idToElement[thisID]={elementIdx:-1, instanceIdx:i};
        for(let ring=0;ring<3;ring++){
          const idx=i*3+ring;
          arr[idx*7+0]=cx;
          arr[idx*7+1]=cy;
          arr[idx*7+2]=cz;
          arr[idx*7+3]=rx; arr[idx*7+4]=ry; arr[idx*7+5]=rz;
          arr[idx*7+6]=thisID;
        }
      }
      return arr;
    }

    // 3) for each element
    for(let e=0;e<sceneElements.length;e++){
      const elem=sceneElements[e];
      const baseID=elementBaseId[e];
      if(elem.type==='PointCloud'){
        const {positions, colors, scales}=elem.data;
        if(positions.length>0){
          pcInstData=buildPCInstanceData(positions, colors, scales, elem.decorations);
          pcCount=positions.length/3;
          pickPCData=buildPickingPCData(positions, scales, baseID);
          for(let i=0;i<pcCount;i++){
            const thisID=baseID+i;
            idToElement[thisID].elementIdx=e;
          }
        }
      }
      else if(elem.type==='Ellipsoid'){
        const {centers,radii,colors}=elem.data;
        if(centers.length>0){
          elInstData=buildEllipsoidInstanceData(centers,radii,colors, elem.decorations);
          elCount=centers.length/3;
          pickEllipsoidData=buildPickingEllipsoidData(centers,radii, baseID);
          for(let i=0;i<elCount;i++){
            const thisID=baseID+i;
            idToElement[thisID].elementIdx=e;
          }
        }
      }
      else if(elem.type==='EllipsoidBounds'){
        const {centers,radii,colors}=elem.data;
        if(centers.length>0){
          bandInstData=buildEllipsoidBoundsInstanceData(centers,radii,colors, elem.decorations);
          bandCount=(centers.length/3)*3;
          pickBandData=buildPickingBoundsData(centers,radii, baseID);
          const c=centers.length/3;
          for(let i=0;i<c;i++){
            const thisID=baseID+i;
            idToElement[thisID].elementIdx=e;
          }
        }
      }
    }

    // 4) create normal GPU buffers
    if(pcInstData&&pcCount>0){
      const b=device.createBuffer({size:pcInstData.byteLength, usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST});
      device.queue.writeBuffer(b,0,pcInstData);
      gpuRef.current.pcInstanceBuffer?.destroy();
      gpuRef.current.pcInstanceBuffer=b;
      gpuRef.current.pcInstanceCount=pcCount;
    } else {
      gpuRef.current.pcInstanceBuffer?.destroy();
      gpuRef.current.pcInstanceBuffer=null;
      gpuRef.current.pcInstanceCount=0;
    }

    if(elInstData&&elCount>0){
      const b=device.createBuffer({size:elInstData.byteLength, usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST});
      device.queue.writeBuffer(b,0,elInstData);
      gpuRef.current.ellipsoidInstanceBuffer?.destroy();
      gpuRef.current.ellipsoidInstanceBuffer=b;
      gpuRef.current.ellipsoidInstanceCount=elCount;
    } else {
      gpuRef.current.ellipsoidInstanceBuffer?.destroy();
      gpuRef.current.ellipsoidInstanceBuffer=null;
      gpuRef.current.ellipsoidInstanceCount=0;
    }

    if(bandInstData&&bandCount>0){
      const b=device.createBuffer({size:bandInstData.byteLength, usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST});
      device.queue.writeBuffer(b,0,bandInstData);
      gpuRef.current.bandInstanceBuffer?.destroy();
      gpuRef.current.bandInstanceBuffer=b;
      gpuRef.current.bandInstanceCount=bandCount;
    } else {
      gpuRef.current.bandInstanceBuffer?.destroy();
      gpuRef.current.bandInstanceBuffer=null;
      gpuRef.current.bandInstanceCount=0;
    }

    // 5) create picking GPU buffers
    if(pickPCData&&pickPCData.length>0){
      const b=device.createBuffer({size:pickPCData.byteLength, usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST});
      device.queue.writeBuffer(b,0,pickPCData);
      pickPCVertexBufferRef.current=b;
      pickPCCountRef.current=pickPCData.length/6;
    } else {
      pickPCVertexBufferRef.current=null;
      pickPCCountRef.current=0;
    }

    if(pickEllipsoidData&&pickEllipsoidData.length>0){
      const b=device.createBuffer({size:pickEllipsoidData.byteLength, usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST});
      device.queue.writeBuffer(b,0,pickEllipsoidData);
      pickEllipsoidVBRef.current=b;
      pickEllipsoidCountRef.current=pickEllipsoidData.length/7;
    } else {
      pickEllipsoidVBRef.current=null;
      pickEllipsoidCountRef.current=0;
    }

    if(pickBandData&&pickBandData.length>0){
      const b=device.createBuffer({size:pickBandData.byteLength, usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST});
      device.queue.writeBuffer(b,0,pickBandData);
      pickBandsVBRef.current=b;
      pickBandsCountRef.current=pickBandData.length/7;
    } else {
      pickBandsVBRef.current=null;
      pickBandsCountRef.current=0;
    }
  },[]);

  /******************************************************
   * H) Mouse
   ******************************************************/
  interface MouseState {
    type:'idle'|'dragging';
    button?: number;
    startX?: number;
    startY?: number;
    lastX?: number;
    lastY?: number;
    isShiftDown?:boolean;
    dragDistance?: number;
  }
  const mouseState=useRef<MouseState>({type:'idle'});
  const lastHoverID=useRef<number>(0);

  /******************************************************
   * H.1) GPU picking pass
   ******************************************************/
  const pickAtScreenXY=useCallback(async (screenX:number, screenY:number, mode:'hover'|'click')=>{
    if(!gpuRef.current || pickingLockRef.current) return;
    pickingLockRef.current = true;

    try {
      const {
        device, pickTexture, pickDepthTexture, readbackBuffer,
        pickPipelineBillboard, pickPipelineEllipsoid, pickPipelineBands,
        billboardQuadVB, billboardQuadIB,
        sphereVB, sphereIB, sphereIndexCount,
        ringVB, ringIB, ringIndexCount,
        uniformBuffer, uniformBindGroup,
        idToElement
      }=gpuRef.current;
      if(!pickTexture||!pickDepthTexture) return;

      // Convert screen coordinates to pick coordinates
      const pickX = Math.floor(screenX);
      const pickY = Math.floor(screenY); // Remove the flip, as Y is already in correct space

      // Add bounds checking
      if(pickX < 0 || pickY < 0 || pickX >= canvasWidth || pickY >= canvasHeight) {
        if(mode === 'hover' && lastHoverID.current !== 0) {
          lastHoverID.current = 0;
          handleHoverID(0);
        }
        return;
      }

      const cmd= device.createCommandEncoder();
      const passDesc: GPURenderPassDescriptor={
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
      const pass=cmd.beginRenderPass(passDesc);
      pass.setBindGroup(0, uniformBindGroup);

      // A) point cloud
      if(pickPCVertexBufferRef.current&&pickPCCountRef.current>0){
        pass.setPipeline(pickPipelineBillboard);
        pass.setVertexBuffer(0,billboardQuadVB);
        pass.setIndexBuffer(billboardQuadIB,'uint16');
        pass.setVertexBuffer(1, pickPCVertexBufferRef.current);
        pass.drawIndexed(6, pickPCCountRef.current);
      }
      // B) ellipsoids
      if(pickEllipsoidVBRef.current&&pickEllipsoidCountRef.current>0){
        pass.setPipeline(pickPipelineEllipsoid);
        pass.setVertexBuffer(0,sphereVB);
        pass.setIndexBuffer(sphereIB,'uint16');
        pass.setVertexBuffer(1, pickEllipsoidVBRef.current);
        pass.drawIndexed(sphereIndexCount, pickEllipsoidCountRef.current);
      }
      // C) bands
      if(pickBandsVBRef.current&&pickBandsCountRef.current>0){
        pass.setPipeline(pickPipelineBands);
        pass.setVertexBuffer(0, ringVB);
        pass.setIndexBuffer(ringIB,'uint16');
        pass.setVertexBuffer(1, pickBandsVBRef.current);
        pass.drawIndexed(ringIndexCount, pickBandsCountRef.current);
      }
      pass.end();

      // copy that pixel
      cmd.copyTextureToBuffer(
        {
          texture: pickTexture,
          origin: { x: pickX, y: pickY }
        },
        {
          buffer: readbackBuffer,
          bytesPerRow: 256,  // Changed from 4 to 256
          rowsPerImage: 1
        },
        [1, 1, 1]
      );

      device.queue.submit([cmd.finish()]);

      try {
        await readbackBuffer.mapAsync(GPUMapMode.READ);
        const arr = new Uint8Array(readbackBuffer.getMappedRange());
        const r = arr[0], g = arr[1], b = arr[2];
        readbackBuffer.unmap();
        const pickedID = (b<<16)|(g<<8)|r;

        if(mode === 'hover') {
          if(pickedID !== lastHoverID.current) {
            lastHoverID.current = pickedID;
            handleHoverID(pickedID);
          }
        } else if(mode === 'click') {
          handleClickID(pickedID);
        }
      } catch(e) {
        console.error('Error during buffer mapping:', e);
      }
    } finally {
      pickingLockRef.current = false;
    }
  },[canvasWidth, canvasHeight]);

  function handleHoverID(pickedID:number){
    if(!gpuRef.current) return;
    const {idToElement}=gpuRef.current;
    if(pickedID<=0||pickedID>=idToElement.length){
      // no object
      for(const elem of elements){
        if(elem.onHover) elem.onHover(null);
      }
      return;
    }
    const {elementIdx, instanceIdx}=idToElement[pickedID];
    if(elementIdx<0||instanceIdx<0||elementIdx>=elements.length){
      // no object
      for(const elem of elements){
        if(elem.onHover) elem.onHover(null);
      }
      return;
    }
    // call that one, then null on others
    const elem=elements[elementIdx];
    elem.onHover?.(instanceIdx);
    for(let e=0;e<elements.length;e++){
      if(e!==elementIdx){
        elements[e].onHover?.(null);
      }
    }
  }
  function handleClickID(pickedID:number){
    if(!gpuRef.current) return;
    const {idToElement}=gpuRef.current;
    if(pickedID<=0||pickedID>=idToElement.length) return;
    const {elementIdx, instanceIdx}=idToElement[pickedID];
    if(elementIdx<0||instanceIdx<0||elementIdx>=elements.length) return;
    elements[elementIdx].onClick?.(instanceIdx);
  }

  /******************************************************
   * H.2) Mouse
   ******************************************************/
  const handleMouseMove=useCallback((e:ReactMouseEvent)=>{
    if(!canvasRef.current) return;
    const rect=canvasRef.current.getBoundingClientRect();
    // Get coordinates relative to canvas element
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const st=mouseState.current;
    if(st.type==='dragging'&&st.lastX!==undefined&&st.lastY!==undefined){
      const dx=e.clientX-st.lastX, dy=e.clientY-st.lastY;
      st.dragDistance=(st.dragDistance||0)+Math.sqrt(dx*dx+dy*dy);
      if(st.button===2||st.isShiftDown){
        setCamera(cam=>({
          ...cam,
          panX:cam.panX-dx*0.002,
          panY:cam.panY+dy*0.002
        }));
      } else if(st.button===0){
        setCamera(cam=>{
          const newPhi=Math.max(0.1, Math.min(Math.PI-0.1, cam.orbitPhi-dy*0.01));
          return {
            ...cam,
            orbitTheta:cam.orbitTheta-dx*0.01,
            orbitPhi:newPhi
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
    if(!canvasRef.current)return;
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
    if(st.type==='dragging'&&st.startX!==undefined&&st.startY!==undefined){
      if(!canvasRef.current)return;
      const rect=canvasRef.current.getBoundingClientRect();
      const x=e.clientX-rect.left, y=e.clientY-rect.top;
      if((st.dragDistance||0)<4){
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
        orbitRadius:Math.max(0.01, cam.orbitRadius+d)
      }));
    }
  },[]);

  /******************************************************
   * I) Effects
   ******************************************************/
  useEffect(()=>{
    initWebGPU();
    return ()=>{
      if(gpuRef.current){
        const {
          billboardQuadVB,billboardQuadIB,
          sphereVB,sphereIB,
          ringVB,ringIB,
          uniformBuffer,
          pcInstanceBuffer,ellipsoidInstanceBuffer,bandInstanceBuffer,
          depthTexture,
          pickTexture,pickDepthTexture,
          readbackBuffer
        }=gpuRef.current;
        billboardQuadVB.destroy();
        billboardQuadIB.destroy();
        sphereVB.destroy();
        sphereIB.destroy();
        ringVB.destroy();
        ringIB.destroy();
        uniformBuffer.destroy();
        pcInstanceBuffer?.destroy();
        ellipsoidInstanceBuffer?.destroy();
        bandInstanceBuffer?.destroy();
        depthTexture?.destroy();
        pickTexture?.destroy();
        pickDepthTexture?.destroy();
        readbackBuffer.destroy();
      }
      if(rafIdRef.current) cancelAnimationFrame(rafIdRef.current);
    };
  },[initWebGPU]);

  useEffect(()=>{
    if(isReady){
      createOrUpdateDepthTexture();
      createOrUpdatePickTextures();
    }
  },[isReady, canvasWidth, canvasHeight, createOrUpdateDepthTexture, createOrUpdatePickTextures]);

  useEffect(()=>{
    if(canvasRef.current){
      canvasRef.current.width=canvasWidth;
      canvasRef.current.height=canvasHeight;
    }
  },[canvasWidth, canvasHeight]);

  useEffect(()=>{
    if(isReady){
      renderFrame();
      rafIdRef.current=requestAnimationFrame(renderFrame);
    }
    return ()=>{
      if(rafIdRef.current) cancelAnimationFrame(rafIdRef.current);
    };
  },[isReady, renderFrame]);

  useEffect(()=>{
    if(isReady){
      updateBuffers(elements);
    }
  },[isReady,elements,updateBuffers]);

  return (
    <div style={{ width:'100%', border:'1px solid #ccc' }}>
      <canvas
        ref={canvasRef}
        style={{ width:'100%', height:canvasHeight }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
        onWheel={onWheel}
        onWheelCapture={(e) => e.preventDefault()}
      />
    </div>
  );
}

/******************************************************
 * 6) Example: App
 ******************************************************/

// generate sample data outside of the component

  // Generate a sphere point cloud
  function generateSpherePointCloud(numPoints: number, radius: number){
    const positions=new Float32Array(numPoints*3);
    const colors=new Float32Array(numPoints*3);
    for(let i=0;i<numPoints;i++){
      const theta=Math.random()*2*Math.PI;
      const phi=Math.acos(2*Math.random()-1);
      const x=radius*Math.sin(phi)*Math.cos(theta);
      const y=radius*Math.sin(phi)*Math.sin(theta);
      const z=radius*Math.cos(phi);
      positions[i*3]=x; positions[i*3+1]=y; positions[i*3+2]=z;
      colors[i*3]=Math.random();
      colors[i*3+1]=Math.random();
      colors[i*3+2]=Math.random();
    }
    return {positions, colors};
  }

const numPoints=500;
const radius=0.5;
const {positions:spherePositions, colors:sphereColors}=generateSpherePointCloud(numPoints, radius);

export function App() {
  const [highlightIdx, setHighlightIdx] = useState<number|null>(null);
  const [hoveredEllipsoid, setHoveredEllipsoid] = useState<{type:'Ellipsoid'|'EllipsoidBounds', index:number}|null>(null);




  // Define a point cloud
  const pcElement: PointCloudElementConfig={
    type:'PointCloud',
    data:{ positions: spherePositions, colors: sphereColors },
    decorations: highlightIdx==null?undefined:[
      {indexes:[highlightIdx], color:[1,0,1], alpha:1, minSize:0.05}
    ],
    onHover:(i)=>{
      if(i===null){
        setHighlightIdx(null);
        setHoveredEllipsoid(null);
      } else {
        setHighlightIdx(i);
        setHoveredEllipsoid(null);
      }
    },
    onClick:(i)=> alert(`Clicked point #${i}`)
  };

  // Some ellipsoids
  const eCenters=new Float32Array([0,0,0, 0.5,0.2,-0.2]);
  const eRadii=new Float32Array([0.2,0.3,0.15, 0.1,0.25,0.2]);
  const eColors=new Float32Array([0.8,0.2,0.2, 0.2,0.8,0.2]);

  const ellipsoidElement: EllipsoidElementConfig={
    type:'Ellipsoid',
    data:{ centers:eCenters, radii:eRadii, colors:eColors },
    decorations: hoveredEllipsoid?.type==='Ellipsoid' ? [
      {indexes:[hoveredEllipsoid.index], color:[1,1,0], alpha:0.8}
    ]: undefined,
    onHover:(i)=>{
      if(i===null){
        setHoveredEllipsoid(null);
      } else {
        setHoveredEllipsoid({type:'Ellipsoid', index:i});
        setHighlightIdx(null);
      }
    },
    onClick:(i)=> alert(`Click ellipsoid #${i}`)
  };

  // Some bounds
  const boundCenters=new Float32Array([
    -0.4,0.4,0,
    0.3,-0.4,0.3,
    -0.3,-0.3,0.2
  ]);
  const boundRadii=new Float32Array([
    0.25,0.25,0.25,
    0.4,0.2,0.15,
    0.15,0.35,0.25
  ]);
  const boundColors=new Float32Array([
    1.0,0.7,0.2,
    0.2,0.7,1.0,
    0.8,0.3,1.0
  ]);

  const boundElement: EllipsoidBoundsElementConfig={
    type:'EllipsoidBounds',
    data:{ centers: boundCenters, radii: boundRadii, colors: boundColors },
    decorations:[
      {indexes:[0], alpha:1},
      {indexes:[1], alpha:1},
      {indexes:[2], alpha:1},
      ...(hoveredEllipsoid?.type==='EllipsoidBounds'
        ? [{indexes:[hoveredEllipsoid.index], color:[1,1,0], alpha:0.8}]
        : [])
    ],
    onHover:(i)=>{
      if(i===null){
        setHoveredEllipsoid(null);
      } else {
        setHoveredEllipsoid({type:'EllipsoidBounds', index:i});
        setHighlightIdx(null);
      }
    },
    onClick:(i)=> alert(`Click bounds #${i}`)
  };

  const elements: SceneElementConfig[] = [
    pcElement,
    ellipsoidElement,
    boundElement
  ];

  return <SceneWrapper elements={elements}/>;
}

/** convenience export */
export function Torus(props:{ elements:SceneElementConfig[] }){
  return <SceneWrapper {...props}/>;
}
