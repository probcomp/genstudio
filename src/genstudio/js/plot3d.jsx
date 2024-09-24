import { Canvas, useThree, extend } from '@react-three/fiber'
import { OrbitControls, Text, Line, Sphere, Box, Plane, useTexture, Html } from '@react-three/drei'
import * as THREE from 'three'
import {React, ReactDOM} from "./imports"
console.log(React)

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
 */
export function Scene({ children, viewCoordinates = 'RIGHT_UP_FORWARD', ambientLightIntensity = 0.5 }) {
  return (
    <Canvas>
      <CameraControls />
      <ambientLight intensity={ambientLightIntensity} />
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
  return (
    <points onClick={onClick}>
      <bufferGeometry>
        <bufferAttribute attachObject={['attributes', 'position']} count={positions.length / 3} array={new Float32Array(positions)} itemSize={3} />
        {colors && <bufferAttribute attachObject={['attributes', 'color']} count={colors.length / 3} array={new Float32Array(colors)} itemSize={3} />}
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
      {boxSizes.map((size, index) => (
        <group key={index}>
          <Box args={size} position={centers[index]} rotation={rotations[index]} onClick={onClick}>
            <meshStandardMaterial color={colors ? colors[index] : 'white'} />
          </Box>
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
      {origins.map((origin, index) => (
        <arrowHelper
          key={index}
          args={[
            new THREE.Vector3(...vectors[index]).normalize(),
            new THREE.Vector3(...origin),
            vectors[index].length(),
            colors ? colors[index] : 0xffff00
          ]}
          onClick={onClick}
        />
      ))}
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
  return (
    <group position={origin}>
      {axes.map((axis, index) => (
        <arrowHelper key={index} args={[new THREE.Vector3(...axis).normalize(), new THREE.Vector3(0, 0, 0), axis.length(), ['red', 'green', 'blue'][index]]} />
      ))}
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
  return <group position={translation} rotation={rotation} scale={scale}>{children}</group>
}

export function App() {
  const handleClick = (event) => {
    console.log('Clicked:', event.object);
  };

  return (
    <Scene viewCoordinates="RIGHT_UP_FORWARD">
      <Transform translation={[0, 0, 0]}>
        <Points3D
          positions={[0, 0, 0, 1, 1, 1, -1, 0, 1]}
          colors={[1, 0, 0, 0, 1, 0, 0, 0, 1]}
          radii={[0.1, 0.2, 0.15]}
          onClick={handleClick}
        />
        <LineStrips3D
          strips={[[0, 0, 0, 1, 1, 1], [-1, 0, 1, 0, 1, 0]]}
          colors={[[1, 0, 0], [0, 1, 0]]}
          radii={[0.05, 0.1]}
          labels={['Strip 1', 'Strip 2']}
        />
        <Transform translation={[1, 1, 1]}>
          <Boxes3D
            sizes={[[1, 1, 1], [0.5, 0.5, 0.5]]}
            centers={[[0, 0, 0], [1, 1, 1]]}
            rotations={[[0, 0, 0], [Math.PI/4, 0, Math.PI/4]]}
            colors={[[0, 0, 1], [1, 1, 0]]}
            labels={['Box 1', 'Box 2']}
            onClick={handleClick}
          />
        </Transform>
        <Arrows3D
          origins={[[0, 0, 0], [1, 1, 1]]}
          vectors={[[1, 0, 0], [0, 1, 0]]}
          colors={[0xff0000, 0x00ff00]}
          onClick={handleClick}
        />
      </Transform>
    </Scene>
  )
}
