/**
 * @module scene3d
 * @description A high-level React component for rendering 3D scenes using WebGPU.
 * This module provides a declarative interface for 3D visualization, handling camera controls,
 * picking, and efficient rendering of various 3D primitives.
 *
 */

import React, { useMemo } from 'react';
import { SceneInner, ComponentConfig, PointCloudComponentConfig, EllipsoidComponentConfig, EllipsoidAxesComponentConfig, CuboidComponentConfig, LineBeamsComponentConfig } from './impl3d';
import { CameraParams } from './camera3d';
import { useContainerWidth } from '../utils';
import { FPSCounter, useFPSCounter } from './fps';

/**
 * @interface Decoration
 * @description Defines visual modifications that can be applied to specific instances of a primitive.
 */
interface Decoration {
  /** Array of instance indices to apply the decoration to */
  indexes: number[];
  /** Optional RGB color override */
  color?: [number, number, number];
  /** Optional alpha (opacity) override */
  alpha?: number;
  /** Optional scale multiplier override */
  scale?: number;
}

/**
 * Creates a decoration configuration for modifying the appearance of specific instances.
 * @param indexes - Single index or array of indices to apply decoration to
 * @param options - Optional visual modifications (color, alpha, scale)
 * @returns {Decoration} A decoration configuration object
 */
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

/**
 * Creates a point cloud component configuration.
 * @param props - Point cloud configuration properties
 * @returns {PointCloudComponentConfig} Configuration for rendering points in 3D space
 */
export function PointCloud(props: PointCloudComponentConfig): PointCloudComponentConfig {
  return {
    ...props,
    type: 'PointCloud',
  };
}

/**
 * Creates an ellipsoid component configuration.
 * @param props - Ellipsoid configuration properties
 * @returns {EllipsoidComponentConfig} Configuration for rendering ellipsoids in 3D space
 */
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

/**
 * Creates an ellipsoid axes component configuration.
 * @param props - Ellipsoid axes configuration properties
 * @returns {EllipsoidAxesComponentConfig} Configuration for rendering ellipsoid axes in 3D space
 */
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

/**
 * Creates a cuboid component configuration.
 * @param props - Cuboid configuration properties
 * @returns {CuboidComponentConfig} Configuration for rendering cuboids in 3D space
 */
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

/**
 * Creates a line beams component configuration.
 * @param props - Line beams configuration properties
 * @returns {LineBeamsComponentConfig} Configuration for rendering line beams in 3D space
 */
export function LineBeams(props: LineBeamsComponentConfig): LineBeamsComponentConfig {
  return {
    ...props,
    type: 'LineBeams'
  };
}

/**
 * Computes canvas dimensions based on container width and desired aspect ratio.
 * @param containerWidth - Width of the container element
 * @param width - Optional explicit width override
 * @param height - Optional explicit height override
 * @param aspectRatio - Desired aspect ratio (width/height), defaults to 1
 * @returns Canvas dimensions and style configuration
 */
export function computeCanvasDimensions(containerWidth: number, width?: number, height?: number, aspectRatio = 1) {
    if (!containerWidth && !width) return;

    const finalWidth = width || containerWidth;
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

/**
 * @interface SceneProps
 * @description Props for the Scene component
 */
interface SceneProps {
  /** Array of 3D components to render */
  components: ComponentConfig[];
  /** Optional explicit width */
  width?: number;
  /** Optional explicit height */
  height?: number;
  /** Desired aspect ratio (width/height) */
  aspectRatio?: number;
  /** Current camera parameters (for controlled mode) */
  camera?: CameraParams;
  /** Default camera parameters (for uncontrolled mode) */
  defaultCamera?: CameraParams;
  /** Callback fired when camera parameters change */
  onCameraChange?: (camera: CameraParams) => void;
}

/**
 * A React component for rendering 3D scenes.
 *
 * This component provides a high-level interface for 3D visualization, handling:
 * - WebGPU initialization and management
 * - Camera controls (orbit, pan, zoom)
 * - Mouse interaction and picking
 * - Efficient rendering of multiple primitive types
 *
 * @component
 * @example
 * ```tsx
 * <Scene
 *   components={[
 *     PointCloud({ positions: points, color: [1,0,0] }),
 *     Ellipsoid({ centers: centers, radius: 0.1 })
 *   ]}
 *   width={800}
 *   height={600}
 *   onCameraChange={handleCameraChange}
 * />
 * ```
 */
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
