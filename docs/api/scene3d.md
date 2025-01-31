# genstudio.scene3d {: .api .api-title }

### Scene {: .api .api-member }

A 3D scene visualization component using WebGPU.

This class creates an interactive 3D scene that can contain multiple types of components:
- Point clouds
- Ellipsoids
- Ellipsoid bounds (wireframe)
- Cuboids

The visualization supports:
- Orbit camera control (left mouse drag)
- Pan camera control (shift + left mouse drag or middle mouse drag)
- Zoom control (mouse wheel)
- Component hover highlighting
- Component click selection



### PointCloud {: .api .api-member }

Create a point cloud element.

Parameters
{: .api .api-section }


- `positions` (ArrayLike): Nx3 array of point positions or flattened array

- `colors` (Optional[ArrayLike]): Nx3 array of RGB colors or flattened array (optional)

- `color` (Optional[ArrayLike]): Default RGB color [r,g,b] for all points if colors not provided

- `sizes` (Optional[ArrayLike]): N array of point sizes or flattened array (optional)

- `size` (Optional[NumberLike]): Default size for all points if sizes not provided

- `**kwargs` (Any): Additional arguments like decorations, onHover, onClick



### Ellipsoid {: .api .api-member }

Create an ellipsoid element.

Parameters
{: .api .api-section }


- `centers` (ArrayLike): Nx3 array of ellipsoid centers or flattened array

- `radii` (Optional[ArrayLike]): Nx3 array of radii (x,y,z) or flattened array (optional)

- `radius` (Optional[Union[NumberLike, ArrayLike]]): Default radius (sphere) or [x,y,z] radii (ellipsoid) if radii not provided

- `colors` (Optional[ArrayLike]): Nx3 array of RGB colors or flattened array (optional)

- `color` (Optional[ArrayLike]): Default RGB color [r,g,b] for all ellipsoids if colors not provided

- `**kwargs` (Any): Additional arguments like decorations, onHover, onClick



### EllipsoidAxes {: .api .api-member }

Create an ellipsoid bounds (wireframe) element.

Parameters
{: .api .api-section }


- `centers` (ArrayLike): Nx3 array of ellipsoid centers or flattened array

- `radii` (Optional[ArrayLike]): Nx3 array of radii (x,y,z) or flattened array (optional)

- `radius` (Optional[Union[NumberLike, ArrayLike]]): Default radius (sphere) or [x,y,z] radii (ellipsoid) if radii not provided

- `colors` (Optional[ArrayLike]): Nx3 array of RGB colors or flattened array (optional)

- `color` (Optional[ArrayLike]): Default RGB color [r,g,b] for all ellipsoids if colors not provided

- `**kwargs` (Any): Additional arguments like decorations, onHover, onClick



### Cuboid {: .api .api-member }

Create a cuboid element.

Parameters
{: .api .api-section }


- `centers` (ArrayLike): Nx3 array of cuboid centers or flattened array

- `sizes` (Optional[ArrayLike]): Nx3 array of sizes (width,height,depth) or flattened array (optional)

- `size` (Optional[Union[ArrayLike, NumberLike]]): Default size [w,h,d] for all cuboids if sizes not provided

- `colors` (Optional[ArrayLike]): Nx3 array of RGB colors or flattened array (optional)

- `color` (Optional[ArrayLike]): Default RGB color [r,g,b] for all cuboids if colors not provided

- `**kwargs` (Any): Additional arguments like decorations, onHover, onClick



### LineBeams {: .api .api-member }

Create a line beams element.

Parameters
{: .api .api-section }


- `positions` (ArrayLike): Array of quadruples [x,y,z,i, x,y,z,i, ...] where points sharing

          the same i value are connected in sequence

- `color` (Optional[ArrayLike]): Default RGB color [r,g,b] for all beams if segment_colors not provided

- `radius`: Default radius for all beams if segment_radii not provided

- `segment_colors`: Array of RGB colors per beam segment (optional)

- `segment_radii`: Array of radii per beam segment (optional)

- `**kwargs` (Any): Additional arguments like onHover, onClick

Returns
{: .api .api-section }


- A LineBeams scene component that renders connected beam segments. (SceneComponent)

- Points are connected in sequence within groups sharing the same i value. (SceneComponent)



### deco {: .api .api-member }

Create a decoration for scene components.

Parameters
{: .api .api-section }


- `indexes` (Union[int, np.integer, ArrayLike]): Single index or list of indices to decorate

- `color` (Optional[ArrayLike]): Optional RGB color override [r,g,b]

- `alpha` (Optional[NumberLike]): Optional opacity value (0-1)

- `scale` (Optional[NumberLike]): Optional scale factor

Returns
{: .api .api-section }


- Dictionary containing decoration settings (Decoration)
