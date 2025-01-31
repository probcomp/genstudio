import React, { useMemo } from 'react';
import { SceneInner, ComponentConfig, PointCloudComponentConfig, EllipsoidComponentConfig, EllipsoidAxesComponentConfig, CuboidComponentConfig, LineBeamsComponentConfig } from './impl3d';
import { CameraParams } from './camera3d';
import { useContainerWidth } from '../utils';
import { FPSCounter, useFPSCounter } from './fps';

interface Decoration {
  indexes: number[];
  color?: [number, number, number];
  alpha?: number;
  scale?: number;
}

export function deco(
  indexes: number | number[],
  options: {
    color?: [number, number, number],
    alpha?: number,
    scale?: number
  } = {}
): Decoration {
  const indexArray = typeof indexes === 'number' ? [indexes] : indexes;
  return { indexes: indexArray, ...options };
}

export function PointCloud(props: PointCloudComponentConfig): PointCloudComponentConfig {
  return {
    ...props,
    type: 'PointCloud',
  };
}

export function Ellipsoid(props: EllipsoidComponentConfig): EllipsoidComponentConfig {
  const radius = typeof props.radius === 'number' ?
    [props.radius, props.radius, props.radius] as [number, number, number] :
    props.radius;

  return {
    ...props,
    radius,
    type: 'Ellipsoid'
  };
}

export function EllipsoidAxes(props: EllipsoidAxesComponentConfig): EllipsoidAxesComponentConfig {
  const radius = typeof props.radius === 'number' ?
    [props.radius, props.radius, props.radius] as [number, number, number] :
    props.radius;

  return {
    ...props,
    radius,
    type: 'EllipsoidAxes'
  };
}

export function Cuboid(props: CuboidComponentConfig): CuboidComponentConfig {
  const size = typeof props.size === 'number' ?
    [props.size, props.size, props.size] as [number, number, number] :
    props.size;

  return {
    ...props,
    size,
    type: 'Cuboid'
  };
}

export function LineBeams(props: LineBeamsComponentConfig): LineBeamsComponentConfig {
  return {
    ...props,
    type: 'LineBeams'
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

  const { fpsDisplayRef, updateDisplay } = useFPSCounter();

  return (
    <div ref={containerRef} style={{ width: '100%', position: 'relative' }}>
      {dimensions && (
        <>
          <SceneInner
            containerWidth={dimensions.width}
            containerHeight={dimensions.height}
            style={dimensions.style}
            components={components}
            camera={camera}
            defaultCamera={defaultCamera}
            onCameraChange={onCameraChange}
            onFrameRendered={updateDisplay}
          />
          <FPSCounter fpsRef={fpsDisplayRef} />
        </>
      )}
    </div>
  );
}
