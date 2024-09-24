import { Box, Line, OrbitControls, PerspectiveCamera, Plane, PointMaterial, Points, Sphere, Text, useTexture } from '@react-three/drei';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import * as React from "react";
import { useRef } from 'react';
import * as THREE from 'three';


// Define ViewCoordinates constants with human-friendly names
const ViewCoordinates = {
  RIGHT_UP_FORWARD: 'RUF',
  RIGHT_UP_BACKWARD: 'RUB',
  RIGHT_DOWN_FORWARD: 'RDF',
  RIGHT_DOWN_BACKWARD: 'RDB',
  LEFT_UP_FORWARD: 'LUF',
  LEFT_UP_BACKWARD: 'LUB',
  LEFT_DOWN_FORWARD: 'LDF',
  LEFT_DOWN_BACKWARD: 'LDB'
}

// Define rotation matrices for each coordinate system
const ViewCoordinateMatrices = {
  RUF: new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler(0, 0, 0)),
  RUB: new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler(0, Math.PI, 0)),
  RDF: new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler(Math.PI, 0, 0)),
  RDB: new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler(Math.PI, Math.PI, 0)),
  LUF: new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler(0, 0, Math.PI)),
  LUB: new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler(0, Math.PI, Math.PI)),
  LDF: new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler(Math.PI, 0, Math.PI)),
  LDB: new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler(Math.PI, Math.PI, Math.PI)),
}

// Helper function to get the matrix for a given coordinate system
function getViewCoordinateMatrix(viewCoordinate) {
  const abbreviation = ViewCoordinates[viewCoordinate] || viewCoordinate
  return ViewCoordinateMatrices[abbreviation] || ViewCoordinateMatrices.RUF
}

function CameraControls() {
  const { camera, gl } = useThree()
  return <OrbitControls args={[camera, gl.domElement]} />
}

/**
 * Creates a point light in the scene.
 * @param {number[]} position - The [x, y, z] position of the light.
 * @param {number} [intensity=1] - The intensity of the light.
 * @param {number} [distance=0] - Maximum range of the light.
 * @param {number} [decay=1] - The amount the light dims along the distance of the light.
 */
export function PointLight({ position, intensity = 1, distance = 0, decay = 1 }) {
  return <pointLight position={position} intensity={intensity} distance={distance} decay={decay} />
}

/**
 * Creates a 3D scene with camera controls and lighting.
 * @param {Object} props - The component props.
 * @param {React.ReactNode} props.children - The child components to render in the scene.
 * @param {string} [props.viewCoordinates='RIGHT_UP_FORWARD'] - The view coordinate system.
 * @param {number} [props.ambientLightIntensity=0.5] - The intensity of the ambient light.
 * @param {number} [props.size=500] - The size of the square canvas in pixels.
 */
export function Scene({ children, viewCoordinates = 'RIGHT_UP_FORWARD', ambientLightIntensity = 0.5, size = 500 }) {
  return (
    <Canvas style={{ width: size, height: size }}>
      <PerspectiveCamera makeDefault position={[5, 5, 5]} />
      <CameraControls />
      <ambientLight intensity={ambientLightIntensity} />
      <pointLight position={[10, 10, 10]} intensity={1} />
      <SceneContent viewCoordinates={viewCoordinates}>{children}</SceneContent>
    </Canvas>
  )
}

function SceneContent({ children, viewCoordinates }) {
  const matrix = getViewCoordinateMatrix(viewCoordinates)
  return <group matrix={matrix}>{children}</group>
}

/**
 * Creates 3D points in the scene.
 * @param {Object} props - The component props.
 * @param {number[]} props.positions - Array of [x, y, z] positions for each point.
 * @param {number[]} [props.colors] - Array of [r, g, b] colors for each point.
 * @param {number[]} [props.radii] - Array of radii for each point.
 * @param {function} [props.onClick] - Click event handler for the points.
 */
export function Points3D({ positions, colors, radii, onClick }) {
  const positionArray = new Float32Array(positions.flat());
  const colorArray = colors ? new Float32Array(colors.flat()) : undefined;

  return (
    <points onClick={onClick}>
      <bufferGeometry>
        <bufferAttribute attachObject={['attributes', 'position']} count={positionArray.length / 3} array={positionArray} itemSize={3} />
        {colorArray && <bufferAttribute attachObject={['attributes', 'color']} count={colorArray.length / 3} array={colorArray} itemSize={3} />}
      </bufferGeometry>
      <pointsMaterial vertexColors={!!colors} size={radii} />
    </points>
  )
}

/**
 * Creates 3D line strips in the scene.
 * @param {Object} props - The component props.
 * @param {number[][][]} props.strips - Array of line strips, each an array of [x, y, z] positions.
 * @param {number[][]} [props.colors] - Array of [r, g, b] colors for each strip.
 * @param {number[]} [props.radii] - Array of radii for each strip.
 * @param {string[]} [props.labels] - Array of labels for each strip.
 */
export function LineStrips3D({ strips, colors, radii, labels }) {
  return (
    <group>
      {strips.map((strip, index) => (
        <Line
          key={index}
          points={strip}
          color={colors ? colors[index] : 'white'}
          lineWidth={radii ? radii[index] : 1}
        />
      ))}
      {labels && labels.map((label, index) => (
        <Text key={index} position={strips[index][strips[index].length - 1]} fontSize={0.1} color="white">
          {label}
        </Text>
      ))}
    </group>
  )
}

/**
 * Creates 3D boxes in the scene.
 * @param {Object} props - The component props.
 * @param {number[][]} [props.sizes] - Array of [width, height, depth] sizes for each box.
 * @param {number[][]} [props.half_sizes] - Array of [halfWidth, halfHeight, halfDepth] sizes for each box.
 * @param {number[][]} props.centers - Array of [x, y, z] positions for the center of each box.
 * @param {number[][]} props.rotations - Array of [x, y, z] rotation angles for each box.
 * @param {string[]} [props.colors] - Array of color strings for each box.
 * @param {string[]} [props.labels] - Array of labels for each box.
 * @param {function} [props.onClick] - Click event handler for the boxes.
 */
export function Boxes3D({ sizes, half_sizes, centers, rotations, colors, labels, onClick }) {
  const boxSizes = sizes || half_sizes.map(hs => hs.map(v => v * 2));
  return (
    <group>
      {boxSizes.map((size, index) => {
        const position = ensureVector3(centers[index]);
        const rotation = ensureVector3(rotations[index]);
        const color = ensureColor(colors ? colors[index] : 'white');
        return (
          <group key={index}>
            <Box args={size} position={position} rotation={rotation} onClick={onClick}>
              <meshStandardMaterial color={color} />
            </Box>
            {labels && (
              <Text position={position} fontSize={0.1} color="white">
                {labels[index]}
              </Text>
            )}
          </group>
        );
      })}
    </group>
  )
}

/**
 * Creates 3D ellipsoids in the scene.
 * @param {Object} props - The component props.
 * @param {number[][]} props.radii - Array of [rx, ry, rz] radii for each ellipsoid.
 * @param {number[][]} props.centers - Array of [x, y, z] positions for the center of each ellipsoid.
 * @param {number[][]} props.rotations - Array of [x, y, z] rotation angles for each ellipsoid.
 * @param {string[]} [props.colors] - Array of color strings for each ellipsoid.
 * @param {string[]} [props.labels] - Array of labels for each ellipsoid.
 * @param {function} [props.onClick] - Click event handler for the ellipsoids.
 */
export function Ellipsoids3D({ radii, centers, rotations, colors, labels, onClick }) {
  return (
    <group>
      {radii.map((radius, index) => (
        <group key={index}>
          <Sphere args={[1, 32, 32]} position={centers[index]} rotation={rotations[index]} scale={radius} onClick={onClick}>
            <meshStandardMaterial color={colors ? colors[index] : 'white'} />
          </Sphere>
          {labels && (
            <Text position={centers[index]} fontSize={0.1} color="white">
              {labels[index]}
            </Text>
          )}
        </group>
      ))}
    </group>
  )
}

/**
 * Creates a 2D image in the 3D scene.
 * @param {Object} props - The component props.
 * @param {HTMLImageElement} props.data - The image data to display.
 */
export function Image({ data }) {
  const texture = useTexture(data)
  return (
    <Plane>
      <meshBasicMaterial map={texture} />
    </Plane>
  )
}

/**
 * Creates 3D lines in the scene.
 * @param {Object} props - The component props.
 * @param {number[]} props.positions - Array of [x, y, z] positions for the line vertices.
 * @param {number[]} [props.colors] - Array of [r, g, b] colors for the line vertices.
 * @param {number} [props.radii] - The width of the lines.
 * @param {function} [props.onClick] - Click event handler for the lines.
 */
export function Lines3D({ positions, colors, radii, onClick }) {
  return (
    <Line
      points={positions}
      color={colors ? colors[0] : 'white'}
      lineWidth={radii}
      onClick={onClick}
    />
  )
}

/**
 * Creates a 3D mesh in the scene.
 * @param {Object} props - The component props.
 * @param {number[]} props.vertex_positions - Array of [x, y, z] positions for the mesh vertices.
 * @param {number[]} [props.vertex_normals] - Array of [nx, ny, nz] normals for the mesh vertices.
 * @param {number[]} [props.vertex_colors] - Array of [r, g, b] colors for the mesh vertices.
 * @param {number[]} [props.indices] - Array of indices defining the mesh triangles.
 * @param {number[]} [props.vertex_uvs] - Array of [u, v] texture coordinates for the mesh vertices.
 * @param {function} [props.onClick] - Click event handler for the mesh.
 */
export function Mesh3D({ vertex_positions, vertex_normals, vertex_colors, indices, vertex_uvs, onClick }) {
  return (
    <mesh onClick={onClick}>
      <bufferGeometry>
        <bufferAttribute attachObject={['attributes', 'position']} count={vertex_positions.length / 3} array={new Float32Array(vertex_positions)} itemSize={3} />
        {vertex_normals && <bufferAttribute attachObject={['attributes', 'normal']} count={vertex_normals.length / 3} array={new Float32Array(vertex_normals)} itemSize={3} />}
        {vertex_colors && <bufferAttribute attachObject={['attributes', 'color']} count={vertex_colors.length / 3} array={new Float32Array(vertex_colors)} itemSize={3} />}
        {vertex_uvs && <bufferAttribute attachObject={['attributes', 'uv']} count={vertex_uvs.length / 2} array={new Float32Array(vertex_uvs)} itemSize={2} />}
        {indices && <bufferAttribute attach="index" count={indices.length} array={new Uint16Array(indices)} itemSize={1} />}
      </bufferGeometry>
      <meshStandardMaterial vertexColors={!!vertex_colors} />
    </mesh>
  )
}

/**
 * Creates 3D arrows in the scene.
 * @param {Object} props - The component props.
 * @param {number[][]} props.origins - Array of [x, y, z] positions for the origin of each arrow.
 * @param {number[][]} props.vectors - Array of [dx, dy, dz] vectors defining the direction and length of each arrow.
 * @param {number[]} [props.colors] - Array of colors for each arrow.
 * @param {function} [props.onClick] - Click event handler for the arrows.
 */
export function Arrows3D({ origins, vectors, colors, onClick }) {
  return (
    <group>
      {origins.map((origin, index) => {
        const originVector = ensureVector3(origin);
        const directionVector = ensureVector3(vectors[index]);
        const color = ensureColor(colors ? colors[index] : 0xffff00);
        return (
          <arrowHelper
            key={index}
            args={[
              directionVector.clone().normalize(),
              originVector,
              directionVector.length(),
              color
            ]}
            onClick={onClick}
          />
        );
      })}
    </group>
  )
}

/**
 * Creates coordinate axes in the scene.
 * @param {Object} props - The component props.
 * @param {number[]} props.origin - The [x, y, z] position of the origin of the coordinate system.
 * @param {number[][]} props.axes - Array of [x, y, z] vectors defining the axes.
 */
export function CoordinateAxes({ origin, axes }) {
  const originVector = ensureVector3(origin);
  return (
    <group position={originVector}>
      {axes.map((axis, index) => {
        const axisVector = ensureVector3(axis);
        return (
          <arrowHelper
            key={index}
            args={[
              axisVector.clone().normalize(),
              new THREE.Vector3(0, 0, 0),
              axisVector.length(),
              ['red', 'green', 'blue'][index]
            ]}
          />
        );
      })}
    </group>
  )
}

/**
 * Applies a transformation to its children.
 * @param {Object} props - The component props.
 * @param {number[]} [props.translation] - The [x, y, z] translation to apply.
 * @param {number[]} [props.rotation] - The [x, y, z] rotation to apply (in radians).
 * @param {number[]} [props.scale] - The [x, y, z] scale to apply.
 * @param {React.ReactNode} props.children - The child components to transform.
 */
export function Transform({ translation, rotation, scale, children }) {
  const translationVector = translation ? ensureVector3(translation) : undefined;
  const rotationVector = rotation ? ensureVector3(rotation) : undefined;
  const scaleVector = scale ? ensureVector3(scale) : undefined;
  return <group position={translationVector} rotation={rotationVector} scale={scaleVector}>{children}</group>
}

// Helper functions
function ensureVector3(value) {
  return value instanceof THREE.Vector3 ? value : new THREE.Vector3(...value);
}

function ensureColor(value) {
  return value instanceof THREE.Color ? value : new THREE.Color(value);
}

function ensureArray(value) {
  return Array.isArray(value) ? value : [value];
}

export function App() {
  const handleClick = (event) => {
    console.log('Clicked:', event.object);
  };
  console.log("Rendering App")
  return (
    <Scene viewCoordinates="RIGHT_UP_FORWARD" size={600}>
      <Transform translation={[0, 0, 0]}>
        <Points3D
          positions={[0, 0, 0, 2, 2, 2, -2, 0, 2]}
          colors={[1, 0, 0, 0, 1, 0, 0, 0, 1]}
          radii={[0.2, 0.3, 0.25]}
          onClick={handleClick}
        />
        <LineStrips3D
          strips={[[0, 0, 0, 2, 2, 2], [-2, 0, 2, 0, 2, 0]]}
          colors={[[1, 0, 0, 1], [0, 1, 0, 1]]}
          radii={[0.1, 0.1]}
          labels={['Strip 1', 'Strip 2']}
        />
        <Boxes3D
          sizes={[[1, 1, 1], [0.8, 0.8, 0.8]]}
          centers={[[0, 0, 0], [2, 2, 2]]}
          rotations={[[0, 0, 0], [Math.PI/4, 0, Math.PI/4]]}
          colors={['#4287f5', '#f5d142']}
          labels={['Box 1', 'Box 2']}
          onClick={handleClick}
        />
        <Arrows3D
          origins={[[0, 0, 0], [2, 2, 2]]}
          vectors={[[1, 0, 0], [0, 1, 0]]}
          colors={['#ff0000', '#00ff00']}
          onClick={handleClick}
        />
        <CoordinateAxes origin={[0, 0, 0]} axes={[[3, 0, 0], [0, 3, 0], [0, 0, 3]]} />
      </Transform>
    </Scene>
  )
}

function PointCloud(props) {
  const pointsRef = useRef()

  useFrame((state, delta) => {
    pointsRef.current.rotation.x += delta * 0.2
    pointsRef.current.rotation.y += delta * 0.1
  })

  const pointCount = 1000
  const positions = new Float32Array(pointCount * 3)

  for (let i = 0; i < pointCount; i++) {
    positions[i * 3] = (Math.random() - 0.5) * 2
    positions[i * 3 + 1] = (Math.random() - 0.5) * 2
    positions[i * 3 + 2] = (Math.random() - 0.5) * 2
  }

  return (
    <Points ref={pointsRef} positions={positions} {...props}>
      <PointMaterial transparent color="#ff88cc" size={0.05} sizeAttenuation={true} depthWrite={false} />
    </Points>
  )
}

export function PointCloudExample({ size = 600 }) {
  return (
    <Canvas style={{ width: size, height: size }}>
      <PerspectiveCamera position={[3, 3, 3]} fov={60} />
      <OrbitControls enableZoom={true} enablePan={true} enableRotate={true} />
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} intensity={0.8} />
      <PointCloud />
      <gridHelper args={[10, 10, '#888888', '#444444']} />
      <axesHelper args={[2]} />
    </Canvas>
  );
}
