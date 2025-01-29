import React, { useMemo } from 'react';
import { SceneInner, SceneElementConfig} from './impl';
import { CameraParams } from './camera3d';
import { useContainerWidth } from '../utils';

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
  elements: SceneElementConfig[];
  width?: number;
  height?: number;
  aspectRatio?: number;
  camera?: CameraParams;
  defaultCamera?: CameraParams;
  onCameraChange?: (camera: CameraParams) => void;
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
