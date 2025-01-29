import React, { useMemo } from 'react';
import { SceneInner, ComponentConfig, PointCloudComponentConfig, EllipsoidComponentConfig, EllipsoidAxesComponentConfig, CuboidComponentConfig } from './impl3d';
import { CameraParams } from './camera3d';
import { useContainerWidth } from '../utils';

interface Decoration {
  indexes: number[];
  color?: [number, number, number];
  alpha?: number;
  scale?: number;
  minSize?: number;
}

export function deco(
  indexes: number | number[],
  options: {
    color?: [number, number, number],
    alpha?: number,
    scale?: number,
    minSize?: number
  } = {}
): Decoration {
  const indexArray = typeof indexes === 'number' ? [indexes] : indexes;
  return { indexes: indexArray, ...options };
}

export function PointCloud(props: PointCloudComponentConfig): PointCloudComponentConfig {

  const {
    positions,
    colors,
    sizes,
    color = [1, 1, 1],
    size = 0.02,
    decorations,
    onHover,
    onClick
  } = props
  console.log("P--", props)

  return {
    type: 'PointCloud',
    positions,
    colors,
    color,
    sizes,
    size,
    decorations,
    onHover,
    onClick
  };
}

export function Ellipsoid({
  centers,
  radii,
  radius = [1, 1, 1],
  colors,
  color = [1, 1, 1],
  decorations,
  onHover,
  onClick
}: EllipsoidComponentConfig): EllipsoidComponentConfig {

  const radiusTriple = typeof radius === 'number' ?
    [radius, radius, radius] as [number, number, number] : radius;

  return {
    type: 'Ellipsoid',
    centers,
    radii,
    radius: radiusTriple,
    colors,
    color,
    decorations,
    onHover,
    onClick
  };
}

export function EllipsoidAxes({
  centers,
  radii,
  radius = [1, 1, 1],
  colors,
  color = [1, 1, 1],
  decorations,
  onHover,
  onClick
}: EllipsoidAxesComponentConfig): EllipsoidAxesComponentConfig {

  const radiusTriple = typeof radius === 'number' ?
    [radius, radius, radius] as [number, number, number] : radius;

  return {
    type: 'EllipsoidAxes',
    centers,
    radii,
    radius: radiusTriple,
    colors,
    color,
    decorations,
    onHover,
    onClick
  };
}

export function Cuboid({
  centers,
  sizes,
  colors,
  color = [1, 1, 1],
  decorations,
  onHover,
  onClick
}: CuboidComponentConfig): CuboidComponentConfig {

  return {
    type: 'Cuboid',
    centers,
    sizes,
    colors,
    color,
    decorations,
    onHover,
    onClick
  };
}

// Add this helper function near the top with other utility functions
export function computeCanvasDimensions(containerWidth: number, width?: number, height?: number, aspectRatio = 1) {
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
interface SceneProps {
  components: ComponentConfig[];
  width?: number;
  height?: number;
  aspectRatio?: number;
  camera?: CameraParams;
  defaultCamera?: CameraParams;
  onCameraChange?: (camera: CameraParams) => void;
}

export function Scene({ components, width, height, aspectRatio = 1, camera, defaultCamera, onCameraChange }: SceneProps) {
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
          components={components}
          camera={camera}
          defaultCamera={defaultCamera}
          onCameraChange={onCameraChange}
        />
      )}
    </div>
  );
}
