# Multi-Element Scene Architecture

## Overview
Transform the PointCloudViewer into a general-purpose 3D scene viewer that can handle multiple types of elements:
- Point clouds
- Images/depth maps as textured quads
- Basic geometric primitives
- Support for element transforms and hierarchies

## Core Components

### 1. Scene Element System

    interface SceneElement {
        id: string;
        type: 'points' | 'image' | 'primitive';
        visible?: boolean;
        transform?: Transform3D;
    }

    interface Transform3D {
        position?: [number, number, number];
        rotation?: [number, number, number];
        scale?: [number, number, number];
    }

Element Types:

    interface PointCloudElement extends SceneElement {
        type: 'points';
        positions: Float32Array;
        colors?: Float32Array;
        decorations?: DecorationProperties[];
    }

    interface ImageElement extends SceneElement {
        type: 'image';
        data: Float32Array | Uint8Array;
        width: number;
        height: number;
        format: 'rgb' | 'rgba' | 'depth';
        blendMode?: 'normal' | 'multiply' | 'screen';
        opacity?: number;
    }

    interface PrimitiveElement extends SceneElement {
        type: 'primitive';
        shape: 'box' | 'sphere' | 'cylinder';
        dimensions: number[];
        color?: [number, number, number];
    }

### 2. Renderer Architecture

1. Base Renderer Class
   - Common WebGL setup
   - Matrix/transform management
   - Shared resources

2. Element-Specific Renderers
   - PointCloudRenderer
   - ImageRenderer
   - PrimitiveRenderer
   - Each handles its own shaders/buffers

3. Scene Graph
   - Transform hierarchies
   - Visibility culling
   - Pick/ray intersection

### 3. Implementation Phases

Phase 1: Core Architecture
- [x] Define base interfaces
- [ ] Set up renderer system
- [ ] Basic transform support
- [ ] Scene graph structure

Phase 2: Point Cloud Enhancement
- [ ] Port existing point cloud renderer
- [ ] Add decoration system
- [ ] Optimize for large datasets
- [ ] LOD support

Phase 3: Image Support
- [ ] Texture quad renderer
- [ ] Depth map visualization
- [ ] Blending modes
- [ ] UV mapping

Phase 4: Primitives
- [ ] Basic shapes
- [ ] Material system
- [ ] Instancing support
- [ ] Shadow casting

Phase 5: Advanced Features
- [ ] Picking system for all elements
- [ ] Scene serialization
- [ ] Animation system
- [ ] Custom shader support

## API Design

    interface SceneViewerProps {
        elements: SceneElement[];
        onElementClick?: (elementId: string, data: any) => void;
        camera?: CameraParams;
        backgroundColor?: [number, number, number];
        renderOptions?: {
            maxPoints?: number;
            frustumCulling?: boolean;
            shadows?: boolean;
        };
    }

    interface SceneViewerMethods {
        addElement(element: SceneElement): void;
        removeElement(elementId: string): void;
        updateElement(elementId: string, updates: Partial<SceneElement>): void;
        setCameraPosition(position: [number, number, number]): void;
        lookAt(target: [number, number, number]): void;
        getElementsAtPosition(x: number, y: number): SceneElement[];
        captureScreenshot(): Promise<Blob>;
    }

## Performance Considerations

1. Memory Management
   - WebGL buffer pooling
   - Texture atlas for multiple images
   - Geometry instancing
   - Efficient scene graph updates

2. Rendering Optimization
   - View frustum culling
   - LOD system for point clouds
   - Occlusion culling
   - Batching similar elements

3. Resource Loading
   - Progressive loading
   - Texture streaming
   - Background worker processing
   - Cache management

## Next Steps

1. Immediate Tasks
   - Create base renderer system
   - Implement transform hierarchy
   - Port existing point cloud renderer
   - Add basic image support

2. Testing Strategy
   - Unit tests for transforms
   - Visual regression tests
   - Performance benchmarks
   - Memory leak detection

3. Documentation
   - API documentation
   - Usage examples
   - Performance guidelines
   - Best practices

## Open Questions

1. Scene Graph
   - How to handle deep hierarchies efficiently?
   - Should we support instancing at the scene graph level?

2. Rendering Pipeline
   - How to handle transparent elements?
   - Should we support custom render passes?
   - How to manage shader variants?

3. Resource Management
   - How to handle large textures efficiently?
   - When to release GPU resources?
   - Caching strategy for elements?

4. API Design
   - How declarative should the API be?
   - Balance between flexibility and simplicity?
   - Error handling strategy?
