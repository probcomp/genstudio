import uuid
from typing import Any, Optional, Sequence, Union

from genstudio.layout import JSCall, JSRef, LayoutItem


class Mark3DSpec:
    def __init__(self, name, data, options):
        self.id = str(uuid.uuid4())
        self.ast = JSCall("Mark3DSpec", [name, data, options])

    def ref_id(self):
        return self.id

    def for_json(self) -> Any:
        return self.ast


class Plot3DSpec(LayoutItem):
    def __init__(self, *specs: Union["Plot3DSpec", Mark3DSpec, dict], **kwargs: Any):
        super().__init__()
        self.layers = self._flatten_layers(specs)
        if kwargs:
            self.layers.append(kwargs)

    def __add__(self, other: Union["Plot3DSpec", Mark3DSpec, dict]) -> "Plot3DSpec":
        new_spec = Plot3DSpec()
        new_spec.layers = self.layers + self._flatten_layers([other])
        return new_spec

    def _flatten_layers(self, layers: Sequence[Any]) -> list[Any]:
        flattened = []
        for layer in layers:
            if isinstance(layer, (list, tuple)):
                flattened.extend(self._flatten_layers(layer))
            elif isinstance(layer, Plot3DSpec):
                flattened.extend(layer.layers)
            else:
                flattened.append(layer)
        return flattened

    def for_json(self) -> Any:
        return JSRef("Plot3DSpec")({"layers": self.layers})


def new(*specs: Any, **kwargs: Any) -> Plot3DSpec:
    return Plot3DSpec(*specs, **kwargs)


def Arrows3D(
    origins: Sequence[Sequence[float]],
    vectors: Sequence[Sequence[float]],
    colors: Optional[Sequence[Sequence[float]]] = None,
    **options,
):
    """
    Create 3D arrows.

    Args:
        origins: Sequence of [x, y, z] arrow origin positions
        vectors: Sequence of [dx, dy, dz] arrow vectors
        colors: Sequence of [r, g, b] or [r, g, b, a] arrow colors (optional)
        **options: Additional options
    """
    data = {"origins": origins, "vectors": vectors, "colors": colors}
    return Mark3DSpec("Arrows3D", data, options)


def Boxes3D(
    sizes: Optional[Sequence[Sequence[float]]] = None,
    half_sizes: Optional[Sequence[Sequence[float]]] = None,
    centers: Optional[Sequence[Sequence[float]]] = None,
    rotations: Optional[Sequence[Sequence[float]]] = None,
    colors: Optional[Sequence[Sequence[float]]] = None,
    labels: Optional[Sequence[str]] = None,
    **options,
):
    """
    Create 3D boxes.

    Args:
        sizes: Full extents in x/y/z. Specify this instead of half_sizes
        half_sizes: All half-extents that make up the batch of boxes. Specify this instead of sizes
        centers: Optional center positions of the boxes
        rotations: Optional rotations for the boxes
        colors: Optional colors for the boxes
        labels: Optional text labels for the boxes
        **options: Additional options
    """
    data = {
        "sizes": sizes,
        "half_sizes": half_sizes,
        "centers": centers,
        "rotations": rotations,
        "colors": colors,
        "labels": labels,
    }
    return Mark3DSpec("Boxes3D", data, options)


def DepthImage(data: Sequence[Sequence[float]], **options):
    """
    Create a depth image.

    Args:
        data: Depth data as a 2D array
        **options: Additional options
    """
    return Mark3DSpec("DepthImage", {"data": data}, options)


def Ellipsoids3D(
    radii: Sequence[Sequence[float]],
    centers: Optional[Sequence[Sequence[float]]] = None,
    rotations: Optional[Sequence[Sequence[float]]] = None,
    colors: Optional[Sequence[Sequence[float]]] = None,
    labels: Optional[Sequence[str]] = None,
    **options,
):
    """
    Create 3D ellipsoids or spheres.

    Args:
        radii: Radii of the ellipsoids
        centers: Optional center positions of the ellipsoids
        rotations: Optional rotations for the ellipsoids
        colors: Optional colors for the ellipsoids
        labels: Optional text labels for the ellipsoids
        **options: Additional options
    """
    data = {
        "radii": radii,
        "centers": centers,
        "rotations": rotations,
        "colors": colors,
        "labels": labels,
    }
    return Mark3DSpec("Ellipsoids3D", data, options)


def Image(data: Sequence[Sequence[Sequence[int]]], **options):
    """
    Create a 2D image.

    Args:
        data: Image data as a 3D array (height, width, channels)
        **options: Additional options
    """
    return Mark3DSpec("Image", {"data": data}, options)


def Lines3D(
    positions: Sequence[Sequence[float]],
    colors: Optional[Sequence[Sequence[float]]] = None,
    radii: Optional[Sequence[float]] = None,
    **options,
):
    """
    Create 3D lines.

    Args:
        positions: Sequence of [x, y, z] line vertex positions
        colors: Sequence of [r, g, b] or [r, g, b, a] line colors (optional)
        radii: Sequence of line widths (optional)
        **options: Additional options
    """
    data = {"positions": positions, "colors": colors, "radii": radii}
    return Mark3DSpec("Lines3D", data, options)


def LineStrips3D(
    strips: Sequence[Sequence[Sequence[float]]],
    colors: Optional[Sequence[Sequence[float]]] = None,
    radii: Optional[Sequence[float]] = None,
    labels: Optional[Sequence[str]] = None,
    **options,
):
    """
    Create 3D line strips.

    Args:
        strips: Sequence of line strips, each a sequence of 3D points
        colors: Optional colors for the line strips
        radii: Optional radii for the line strips
        labels: Optional text labels for the line strips
        **options: Additional options
    """
    data = {"strips": strips, "colors": colors, "radii": radii, "labels": labels}
    return Mark3DSpec("LineStrips3D", data, options)


def Mesh3D(
    vertex_positions: Sequence[Sequence[float]],
    vertex_normals: Optional[Sequence[Sequence[float]]] = None,
    vertex_colors: Optional[Sequence[Sequence[float]]] = None,
    indices: Optional[Sequence[Sequence[int]]] = None,
    vertex_uvs: Optional[Sequence[Sequence[float]]] = None,
    **options,
):
    """
    Create a 3D mesh.

    Args:
        vertex_positions: Sequence of [x, y, z] vertex positions
        vertex_normals: Sequence of [nx, ny, nz] vertex normals (optional)
        vertex_colors: Sequence of [r, g, b] or [r, g, b, a] vertex colors (optional)
        indices: Sequence of [i, j, k] triangle indices (optional)
        vertex_uvs: Sequence of [u, v] vertex UVs (optional)
        **options: Additional options
    """
    data = {
        "vertex_positions": vertex_positions,
        "vertex_normals": vertex_normals,
        "vertex_colors": vertex_colors,
        "indices": indices,
        "vertex_uvs": vertex_uvs,
    }
    return Mark3DSpec("Mesh3D", data, options)


def Points3D(
    positions: Sequence[Sequence[float]],
    colors: Optional[Sequence[Sequence[float]]] = None,
    radii: Optional[Sequence[float]] = None,
    **options,
):
    """
    Create 3D points.

    Args:
        positions: Sequence of [x, y, z] point positions
        colors: Sequence of [r, g, b] or [r, g, b, a] point colors (optional)
        radii: Sequence of point radii (optional)
        **options: Additional options
    """
    data = {"positions": positions, "colors": colors, "radii": radii}
    return Mark3DSpec("Points3D", data, options)


# def ViewCoordinates(origin: Sequence[float], axes: Sequence[Sequence[float]], **options):
#     """
#     Set view coordinates.

#     Args:
#         origin: [x, y, z] point specifying the origin of the coordinate system
#         axes: Sequence of 3 [x, y, z] vectors specifying the axes of the coordinate system
#         **options: Additional options
#     """
#     data = {
#         "origin": origin,
#         "axes": axes
#     }
#     return Mark3DSpec("ViewCoordinates", data, options)
