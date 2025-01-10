import genstudio.plot as Plot
from typing import Any
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


#
#
class Points(Plot.LayoutItem):
    """A 3D point cloud visualization component.

    WARNING: This is an alpha package that will not be maintained as-is. The API and implementation
    are subject to change without notice.

    This class creates an interactive 3D point cloud visualization using WebGL.
    Points can be colored, decorated, and support mouse interactions like clicking,
    hovering, orbiting and zooming.

    Args:
        points: A dict containing point cloud data with keys:
            - position: Float32Array of flattened XYZ coordinates [x,y,z,x,y,z,...]
            - color: Optional Uint8Array of flattened RGB values [r,g,b,r,g,b,...]
            - scale: Optional Float32Array of per-point scale factors
        props: A dict of visualization properties including:
            - backgroundColor: RGB background color (default [0.1, 0.1, 0.1])
            - pointSize: Base size of points in pixels (default 0.1)
            - camera: Camera settings with keys:
                - position: [x,y,z] camera position
                - target: [x,y,z] look-at point
                - up: [x,y,z] up vector
                - fov: Field of view in degrees
                - near: Near clipping plane
                - far: Far clipping plane
            - decorations: Dict of decoration settings, each with keys:
                - indexes: List of point indices to decorate
                - scale: Scale factor for point size (default 1.0)
                - color: Optional [r,g,b] color override
                - alpha: Opacity value 0-1 (default 1.0)
                - minSize: Minimum point size in pixels (default 0.0)
            - onPointClick: Callback(index, event) when points are clicked
            - onPointHover: Callback(index) when points are hovered
            - onCameraChange: Callback(camera) when view changes
            - width: Optional canvas width
            - height: Optional canvas height
            - aspectRatio: Optional aspect ratio constraint

    The visualization supports:
    - Orbit camera control (left mouse drag)
    - Pan camera control (shift + left mouse drag or middle mouse drag)
    - Zoom control (mouse wheel)
    - Point hover highlighting
    - Point click selection
    - Point decorations for highlighting subsets
    - Picking system for point interaction

    Example:
        ```python
        # Create a torus knot point cloud
        xyz = np.random.rand(1000, 3).astype(np.float32).flatten()
        rgb = np.random.randint(0, 255, (1000, 3), dtype=np.uint8).flatten()
        scale = np.random.uniform(0.1, 10, 1000).astype(np.float32)

        points = Points(
            {"position": xyz, "color": rgb, "scale": scale},
            {
                "pointSize": 5.0,
                "camera": {
                    "position": [7, 4, 4],
                    "target": [0, 0, 0],
                    "up": [0, 0, 1],
                    "fov": 40,
                },
                "decorations": {
                    "highlight": {
                        "indexes": [0, 1, 2],
                        "scale": 2.0,
                        "color": [1, 1, 0],
                        "minSize": 10
                    }
                }
            }
        )
        ```
    """

    def __init__(self, points, props):
        self.props = {**props, "points": points}
        super().__init__()

    def for_json(self) -> Any:
        return [Plot.JSRef("scene3dNew.Scene"), self.props]


xyz, rgb = make_torus_knot(10000)
#
Plot.html(
    [
        Plot.JSRef("scene3dNew.Torus"),
        {
            "elements": [
                {"type": "PointCloud", "data": {"positions": xyz, "colors": rgb}}
            ]
        },
    ]
)
