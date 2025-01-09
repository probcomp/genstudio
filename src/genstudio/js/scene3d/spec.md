# Point Cloud & Scene Rendering API Specification (TypeScript)

This document describes a declarative, data-driven API for rendering and interacting with 3D content in a browser, written in TypeScript. The system manages a “scene” with multiple elements (e.g., point clouds, meshes, 2D overlays), all rendered using a single camera. A point cloud is one supported element type; more may be added later.

## Purpose and Overview

- Provide a TypeScript-based, declarative scene renderer for 3D data.
- Support multiple element types, such as:
  - **Point Clouds** (positions, per-point color/scale, decorations).
  - Other elements (e.g., meshes, images) in the future.
- Expose interactive camera controls: orbit, pan, zoom.
- Enable picking/hover/click interactions on supported elements (e.g., points).
- Allow styling subsets of points using a “decoration” mechanism.
- Automatically handle container resizing and device pixel ratio changes.
- Optionally track performance (e.g., FPS).

The API is **declarative**: the user calls a method to “set” an array of elements in the scene, and the underlying system updates/re-renders accordingly.

## Data and Configuration

### Scene and Elements

- **Scene**
  A central object that manages:
  - A camera.
  - An array of “elements,” each specifying data to be rendered.
  - Rendering behaviors (resizing, device pixel ratio, etc.).
  - Interaction callbacks (hover, click, camera changes).

- **Elements**
  Each element describes a renderable 3D object. Examples:
  - A **point cloud** with a set of 3D points.
  - (Planned) A **mesh** or other geometry.
  - (Planned) A **2D image** plane, etc.

### Point Cloud Element

```ts
interface PointCloudData {
  positions: Float32Array;  // [x1, y1, z1, x2, y2, z2, ...]
  colors?: Float32Array;    // [r1, g1, b1, r2, g2, b2, ...] (could be 0-255 or normalized)
  scales?: Float32Array;    // Per-point scale factors
}

interface Decoration {
  indexes: number[];                // Indices of points to style
  color?: [number, number, number]; // Override color if defined
  alpha?: number;                   // Override alpha in [0..1]
  scale?: number;                   // Additional scale multiplier
  minSize?: number;                 // Minimum size in screen pixels
}

interface PointCloudElementConfig {
  type: 'PointCloud';
  data: PointCloudData;
  pointSize?: number;       // Global baseline point size
  decorations?: Decoration[];
}
