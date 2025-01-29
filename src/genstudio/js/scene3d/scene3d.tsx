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
  {
    color,
    alpha,
    scale,
    minSize
  }: {
    color?: [number, number, number],
    alpha?: number,
    scale?: number,
    minSize?: number
  } = {}
): Decoration {
  const indexArray = typeof indexes === 'number' ? [indexes] : indexes;
  return { indexes: indexArray, color, alpha, scale, minSize };
}

export function PointCloud(
  positions: Float32Array,
  colors?: Float32Array,
  scales?: Float32Array,
  decorations?: Decoration[],
  onHover?: (index: number|null) => void,
  onClick?: (index: number|null) => void
): PointCloudComponentConfig {
  return {
    type: 'PointCloud',
    data: {
      positions,
      colors,
      scales
    },
    decorations,
    onHover,
    onClick
  };
}

export function Ellipsoid(
  centers: Float32Array,
  radii: Float32Array,
  colors?: Float32Array,
  decorations?: Decoration[],
  onHover?: (index: number|null) => void,
  onClick?: (index: number|null) => void
): EllipsoidComponentConfig {
  return {
    type: 'Ellipsoid',
    data: {
      centers,
      radii,
      colors
    },
    decorations,
    onHover,
    onClick
  };
}

export function EllipsoidAxes(
  centers: Float32Array,
  radii: Float32Array,
  colors?: Float32Array,
  decorations?: Decoration[],
  onHover?: (index: number|null) => void,
  onClick?: (index: number|null) => void
): EllipsoidAxesComponentConfig {
  return {
    type: 'EllipsoidAxes',
    data: {
      centers,
      radii,
      colors
    },
    decorations,
    onHover,
    onClick
  };
}

export function Cuboid(
  centers: Float32Array,
  sizes: Float32Array,
  colors?: Float32Array,
  decorations?: Decoration[],
  onHover?: (index: number|null) => void,
  onClick?: (index: number|null) => void
): CuboidComponentConfig {
  return {
    type: 'Cuboid',
    data: {
      centers,
      sizes,
      colors
    },
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
