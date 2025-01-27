import genstudio.plot as Plot
from typing import Any, Dict, Union
from colorsys import hsv_to_rgb

import numpy as np


def make_torus_knot(n_points: int):
    # Create a torus knot
    t = np.linspace(0, 4 * np.pi, n_points)
    p, q = 3, 2  # Parameters that determine knot shape
    R, r = 2, 1  # Major and minor radii

    # Torus knot parametric equations
    x = (R + r * np.cos(q * t)) * np.cos(p * t)
    y = (R + r * np.cos(q * t)) * np.sin(p * t)
    z = r * np.sin(q * t)

    # Add Gaussian noise to create volume
    noise_scale = 0.1
    x += np.random.normal(0, noise_scale, n_points)
    y += np.random.normal(0, noise_scale, n_points)
    z += np.random.normal(0, noise_scale, n_points)

    # Base color from position
    angle = np.arctan2(y, x)
    height = (z - z.min()) / (z.max() - z.min())
    radius = np.sqrt(x * x + y * y)
    radius_norm = (radius - radius.min()) / (radius.max() - radius.min())

    # Create hue that varies with angle and height
    hue = (angle / (2 * np.pi) + height) % 1.0
    # Saturation that varies with radius
    saturation = 0.8 + radius_norm * 0.2
    # Value/brightness
    value = 0.8 + np.random.uniform(0, 0.2, n_points)

    # Convert HSV to RGB
    colors = np.array([hsv_to_rgb(h, s, v) for h, s, v in zip(hue, saturation, value)])
    # Reshape colors to match xyz structure before flattening
    rgb = (colors.reshape(-1, 3) * 255).astype(np.uint8).flatten()

    # Prepare point cloud coordinates
    xyz = np.column_stack([x, y, z]).astype(np.float32).flatten()

    return xyz, rgb


def deco(
    indexes: Any,
    *,
    color: Any = None,
    alpha: Any = None,
    scale: Any = None,
    min_size: Any = None,
) -> Dict[str, Any]:
    """Create a decoration for scene elements.

    Args:
        indexes: Single index or list of indices to decorate
        color: Optional RGB color override [r,g,b]
        alpha: Optional opacity value (0-1)
        scale: Optional scale factor
        min_size: Optional minimum size in pixels

    Returns:
        Dictionary containing decoration settings
    """
    # Convert single index to list
    if isinstance(indexes, (int, np.integer)):
        indexes = np.array([indexes])

    # Create base decoration dict
    decoration = {"indexes": indexes}

    # Add optional parameters if provided
    if color is not None:
        decoration["color"] = color
    if alpha is not None:
        decoration["alpha"] = alpha
    if scale is not None:
        decoration["scale"] = scale
    if min_size is not None:
        decoration["minSize"] = min_size

    return decoration


class SceneElement(Plot.LayoutItem):
    """Base class for all 3D scene elements."""

    def __init__(self, type_name: str, data: Dict[str, Any], **kwargs):
        super().__init__()
        self.type = type_name
        self.data = data
        self.decorations = kwargs.get("decorations")
        self.on_hover = kwargs.get("onHover")
        self.on_click = kwargs.get("onClick")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the element to a dictionary representation."""
        element = {"type": self.type, "data": self.data}
        if self.decorations:
            element["decorations"] = self.decorations
        if self.on_hover:
            element["onHover"] = self.on_hover
        if self.on_click:
            element["onClick"] = self.on_click
        return element

    def for_json(self) -> Dict[str, Any]:
        """Convert the element to a JSON-compatible dictionary."""
        return Scene(self).for_json()

    def __add__(self, other: Union["SceneElement", "Scene", Dict[str, Any]]) -> "Scene":
        """Allow combining elements with + operator."""
        if isinstance(other, Scene):
            return other + self
        elif isinstance(other, SceneElement):
            return Scene(self, other)
        elif isinstance(other, dict):
            return Scene(self, other)
        else:
            raise TypeError(f"Cannot add SceneElement with {type(other)}")

    def __radd__(self, other: Dict[str, Any]) -> "Scene":
        """Allow combining elements with + operator when dict is on the left."""
        return Scene(self, other)


class Scene(Plot.LayoutItem):
    """A 3D scene visualization component using WebGPU.

    This class creates an interactive 3D scene that can contain multiple types of elements:
    - Point clouds
    - Ellipsoids
    - Ellipsoid bounds (wireframe)
    - Cuboids

    The visualization supports:
    - Orbit camera control (left mouse drag)
    - Pan camera control (shift + left mouse drag or middle mouse drag)
    - Zoom control (mouse wheel)
    - Element hover highlighting
    - Element click selection
    """

    def __init__(
        self,
        *elements_and_props: Union[SceneElement, Dict[str, Any]],
    ):
        """Initialize the scene.

        Args:
            *elements_and_props: Scene elements and optional properties.
        """
        elements = []
        scene_props = {}
        for item in elements_and_props:
            if isinstance(item, SceneElement):
                elements.append(item)
            elif isinstance(item, dict):
                scene_props.update(item)
            else:
                raise TypeError(f"Invalid type in elements_and_props: {type(item)}")

        self.elements = elements
        self.scene_props = scene_props
        super().__init__()

    def __add__(self, other: Union[SceneElement, "Scene", Dict[str, Any]]) -> "Scene":
        """Allow combining scenes with + operator."""
        if isinstance(other, Scene):
            return Scene(*self.elements, *other.elements, self.scene_props)
        elif isinstance(other, SceneElement):
            return Scene(*self.elements, other, self.scene_props)
        elif isinstance(other, dict):
            return Scene(*self.elements, {**self.scene_props, **other})
        else:
            raise TypeError(f"Cannot add Scene with {type(other)}")

    def __radd__(self, other: Dict[str, Any]) -> "Scene":
        """Allow combining scenes with + operator when dict is on the left."""
        return Scene(*self.elements, {**other, **self.scene_props})

    def for_json(self) -> Any:
        """Convert to JSON representation for JavaScript."""
        elements = [
            e.to_dict() if isinstance(e, SceneElement) else e for e in self.elements
        ]

        props = {"elements": elements, **self.scene_props}

        return [Plot.JSRef("scene3d.Scene"), props]


def point_cloud(
    positions: Any,
    colors: Any = None,
    scales: Any = None,
    **kwargs,
) -> SceneElement:
    """Create a point cloud element.

    Args:
        positions: Nx3 array of point positions or flattened array
        colors: Nx3 array of RGB colors or flattened array (optional)
        scales: N array of point scales or flattened array (optional)
        **kwargs: Additional arguments like decorations, onHover, onClick
    """
    # Ensure arrays are flattened float32/uint8
    if isinstance(positions, (np.ndarray, list)):
        positions = np.asarray(positions, dtype=np.float32)
        if positions.ndim == 2:
            positions = positions.flatten()

    data: Dict[str, Any] = {"positions": positions}

    if isinstance(colors, (np.ndarray, list)):
        colors = np.asarray(colors, dtype=np.float32)
        if colors.ndim == 2:
            colors = colors.flatten()
        data["colors"] = colors

    if isinstance(scales, (np.ndarray, list)):
        scales = np.asarray(scales, dtype=np.float32)
        if scales.ndim > 1:
            scales = scales.flatten()
        data["scales"] = scales

    return SceneElement("PointCloud", data, **kwargs)


def ellipsoid(
    centers: Any,
    radii: Any,
    colors: Any = None,
    **kwargs,
) -> SceneElement:
    """Create an ellipsoid element.

    Args:
        centers: Nx3 array of ellipsoid centers or flattened array
        radii: Nx3 array of radii (x,y,z) or flattened array
        colors: Nx3 array of RGB colors or flattened array (optional)
        **kwargs: Additional arguments like decorations, onHover, onClick
    """
    # Ensure arrays are flattened float32
    centers = np.asarray(centers, dtype=np.float32)
    if centers.ndim == 2:
        centers = centers.flatten()

    radii = np.asarray(radii, dtype=np.float32)
    if radii.ndim == 2:
        radii = radii.flatten()

    data = {"centers": centers, "radii": radii}

    if colors is not None:
        colors = np.asarray(colors, dtype=np.float32)
        if colors.ndim == 2:
            colors = colors.flatten()
        data["colors"] = colors

    return SceneElement("Ellipsoid", data, **kwargs)


def ellipsoid_bounds(
    centers: Any,
    radii: Any,
    colors: Any = None,
    **kwargs,
) -> SceneElement:
    """Create an ellipsoid bounds (wireframe) element.

    Args:
        centers: Nx3 array of ellipsoid centers or flattened array
        radii: Nx3 array of radii (x,y,z) or flattened array
        colors: Nx3 array of RGB colors or flattened array (optional)
        **kwargs: Additional arguments like decorations, onHover, onClick
    """
    # Ensure arrays are flattened float32
    centers = np.asarray(centers, dtype=np.float32)
    if centers.ndim == 2:
        centers = centers.flatten()

    radii = np.asarray(radii, dtype=np.float32)
    if radii.ndim == 2:
        radii = radii.flatten()

    data = {"centers": centers, "radii": radii}

    if colors is not None:
        colors = np.asarray(colors, dtype=np.float32)
        if colors.ndim == 2:
            colors = colors.flatten()
        data["colors"] = colors

    return SceneElement("EllipsoidBounds", data, **kwargs)


def cuboid(
    centers: Any,
    sizes: Any,
    colors: Any = None,
    **kwargs,
) -> SceneElement:
    """Create a cuboid element.

    Args:
        centers: Nx3 array of cuboid centers or flattened array
        sizes: Nx3 array of sizes (width,height,depth) or flattened array
        colors: Nx3 array of RGB colors or flattened array (optional)
        **kwargs: Additional arguments like decorations, onHover, onClick
    """
    # Ensure arrays are flattened float32
    centers = np.asarray(centers, dtype=np.float32)
    if centers.ndim == 2:
        centers = centers.flatten()

    sizes = np.asarray(sizes, dtype=np.float32)
    if sizes.ndim == 2:
        sizes = sizes.flatten()

    data = {"centers": centers, "sizes": sizes}

    if colors is not None:
        colors = np.asarray(colors, dtype=np.float32)
        if colors.ndim == 2:
            colors = colors.flatten()
        data["colors"] = colors

    return SceneElement("Cuboid", data, **kwargs)


def create_demo_scene():
    """Create a demo scene with examples of all element types."""
    # 1. Create a point cloud in a spiral pattern
    n_points = 1000
    t = np.linspace(0, 10 * np.pi, n_points)
    r = t / 30
    x = r * np.cos(t)
    y = r * np.sin(t)
    z = t / 10

    # Create positions array
    positions = np.column_stack([x, y, z])

    # Create rainbow colors
    hue = t / t.max()
    colors = np.zeros((n_points, 3))
    # Red component
    colors[:, 0] = np.clip(1.5 - abs(3.0 * hue - 1.5), 0, 1)
    # Green component
    colors[:, 1] = np.clip(1.5 - abs(3.0 * hue - 3.0), 0, 1)
    # Blue component
    colors[:, 2] = np.clip(1.5 - abs(3.0 * hue - 4.5), 0, 1)

    # Create varying scales for points
    scales = 0.01 + 0.02 * np.sin(t)

    # Create the base scene with shared elements
    base_scene = (
        point_cloud(
            positions,
            colors,
            scales,
            onHover=Plot.js(
                "(i) => console.log('hover') || $state.update({hover_point: i})"
            ),
            decorations=[
                {
                    "indexes": Plot.js(
                        "$state.hover_point ? [$state.hover_point] : []"
                    ),
                    "color": [1, 1, 0],
                    "scale": 1.5,
                }
            ],
        )
        +
        # Ellipsoids with one highlighted
        ellipsoid(
            centers=np.array([[0.5, 0.5, 0.5], [-0.5, -0.5, 0.5], [0.0, 0.0, 0.0]]),
            radii=np.array([[0.1, 0.2, 0.1], [0.2, 0.1, 0.1], [0.15, 0.15, 0.15]]),
            colors=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            decorations=[deco([1], color=[1, 1, 0], alpha=0.8)],
        )
        +
        # Ellipsoid bounds with transparency
        ellipsoid_bounds(
            centers=np.array([[0.8, 0.0, 0.0], [-0.8, 0.0, 0.0]]),
            radii=np.array([[0.2, 0.1, 0.1], [0.1, 0.2, 0.1]]),
            colors=np.array([[1.0, 0.5, 0.0], [0.0, 0.5, 1.0]]),
            decorations=[deco([0, 1], alpha=0.5)],
        )
        +
        # Cuboids with one enlarged
        cuboid(
            centers=np.array([[0.0, -0.8, 0.0], [0.0, -0.8, 0.3]]),
            sizes=np.array([[0.3, 0.1, 0.2], [0.2, 0.1, 0.2]]),
            colors=np.array([[0.8, 0.2, 0.8], [0.2, 0.8, 0.8]]),
            decorations=[deco([0], scale=1.2)],
        )
    ) + {
        "width": 400,
        "height": 400,
        "camera": Plot.js("$state.camera"),
        "onCameraChange": Plot.js("(camera) => $state.update({camera})"),
    }

    # Create a layout with two scenes side by side
    scene = (base_scene & base_scene) | Plot.initialState(
        {
            "camera": {
                "orbitRadius": 1.5,
                "orbitTheta": 0.2,
                "orbitPhi": 1.0,
                "panX": 0,
                "panY": 0,
                "fov": 1.0472,  # Math.PI/3
                "near": 0.01,
                "far": 100.0,
            }
        }
    )

    return scene


create_demo_scene()
