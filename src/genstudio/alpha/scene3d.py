import genstudio.plot as Plot
from typing import Any


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
        props: A dict of visualization properties including:
            - backgroundColor: RGB background color (default [0.1, 0.1, 0.1])
            - pointSize: Size of points in pixels (default 4.0)
            - camera: Camera position and orientation settings
            - decorations: Point decoration settings for highlighting
            - onPointClick: Callback when points are clicked
            - onPointHover: Callback when points are hovered
            - onCameraChange: Callback when camera view changes

    Example:
        ```python
        xyz = np.array([0,0,0, 1,1,1], dtype=np.float32)  # Two points
        rgb = np.array([255,0,0, 0,255,0], dtype=np.uint8)  # Red and green
        points = Points(
            {"position": xyz, "color": rgb},
            {"pointSize": 5.0}
        )
        ```
    """

    def __init__(self, points, props):
        self.props = {**props, "points": points}

    def for_json(self) -> Any:
        return [Plot.JSRef("scene3d.Scene"), self.props]
