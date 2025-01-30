
export function createSphereGeometry(stacks=16, slices=24) {
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
      // Reverse winding order by swapping vertices
      idxs.push(row1,row1+1,row2, row1+1,row2+1,row2);  // Changed from (row1,row2,row1+1, row1+1,row2,row2+1)
    }
  }
  return {
    vertexData: new Float32Array(verts),
    indexData: new Uint16Array(idxs)
  };
}

export function createTorusGeometry(majorRadius:number, minorRadius:number, majorSegments:number, minorSegments:number) {
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

export function createCubeGeometry() {
  // 6 faces => 24 verts, 36 indices
  const positions: number[] = [
    // +X face (right) - when looking at it from right side
    0.5, -0.5, -0.5,   0.5, -0.5,  0.5,   0.5,  0.5, -0.5,   0.5,  0.5,  0.5,  // reordered: BL,BR,TL,TR
    // -X face (left) - when looking at it from left side
    -0.5, -0.5,  0.5,  -0.5, -0.5, -0.5,  -0.5,  0.5,  0.5,  -0.5,  0.5, -0.5,  // reordered: BL,BR,TL,TR
    // +Y face (top) - when looking down at it
    -0.5,  0.5, -0.5,   0.5,  0.5, -0.5,  -0.5,  0.5,  0.5,   0.5,  0.5,  0.5,  // reordered: BL,BR,TL,TR
    // -Y face (bottom) - when looking up at it
    -0.5, -0.5,  0.5,   0.5, -0.5,  0.5,  -0.5, -0.5, -0.5,   0.5, -0.5, -0.5,  // reordered: BL,BR,TL,TR
    // +Z face (front) - when looking at front
    -0.5, -0.5,  0.5,   0.5, -0.5,  0.5,  -0.5,  0.5,  0.5,   0.5,  0.5,  0.5,  // reordered: BL,BR,TL,TR
    // -Z face (back) - when looking at it from behind
     0.5, -0.5, -0.5,  -0.5, -0.5, -0.5,   0.5,  0.5, -0.5,  -0.5,  0.5, -0.5,  // reordered: BL,BR,TL,TR
  ];

  // Normals stay the same as they define face orientation
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

  // For each face, define triangles in CCW order when viewed from outside
  const indices: number[] = [];
  for(let face=0; face<6; face++){
    const base = face*4;
    // All faces use same pattern: BL->BR->TL, BR->TR->TL
    indices.push(
      base+0, base+1, base+2,  // first triangle: BL->BR->TL
      base+1, base+3, base+2   // second triangle: BR->TR->TL
    );
  }

  // Interleave positions and normals
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
 * createCylinderGeometry
 * Returns a "unit cylinder" from z=0..1, radius=1.
 * No caps. 'segments' sets how many radial divisions.
 ******************************************************/
export function createCylinderGeometry(segments: number=8) {
  const vertexData: number[] = [];
  const indexData: number[]  = [];

  // Build two rings: z=0 and z=1
  for (let z of [0, 1]) {
    for (let s = 0; s < segments; s++) {
      const theta = 2 * Math.PI * s / segments;
      const x = Math.cos(theta);
      const y = Math.sin(theta);
      // position + normal
      vertexData.push(x, y, z,  x, y, 0); // (pos.x, pos.y, pos.z, n.x, n.y, n.z)
    }
  }
  // Connect ring 0..segments-1 to ring segments..2*segments-1
  for (let s = 0; s < segments; s++) {
    const sNext = (s + 1) % segments;
    const i0 = s;
    const i1 = sNext;
    const i2 = s + segments;
    const i3 = sNext + segments;
    // Two triangles: (i0->i2->i1), (i1->i2->i3)
    indexData.push(i0, i2, i1,  i1, i2, i3);
  }

  return {
    vertexData: new Float32Array(vertexData),
    indexData:  new Uint16Array(indexData)
  };
}
