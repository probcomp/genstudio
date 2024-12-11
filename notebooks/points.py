# %%
import genstudio.plot as Plot
import numpy as np
from genstudio.plot import js
from typing import Any

# Create a beautiful 3D shape with 50k points
n_points = 50000

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

# Create vibrant colors based on position and add some randomness
from colorsys import hsv_to_rgb

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


class Points(Plot.LayoutItem):
    def __init__(self, props):
        self.props = props

    def for_json(self) -> Any:
        return [Plot.JSRef("points.PointCloudViewer"), self.props]


# Camera parameters - positioned to see the spiral structure
camera = {
    "position": [5, 5, 3],
    "target": [0, 0, 0],
    "up": [0, 0, 1],
    "fov": 45,
    "near": 0.1,
    "far": 1000,
}


def scene(controlled, point_size=4):
    cameraProps = (
        {"defaultCamera": camera}
        if not controlled
        else {
            "onCameraChange": js("(camera) => $state.update({'camera': camera})"),
            "camera": js("$state.camera"),
        }
    )
    return Points(
        {
            "points": {"xyz": xyz, "rgb": rgb},
            "backgroundColor": [0.1, 0.1, 0.1, 1],  # Dark background to make colors pop
            "className": "h-[400px] w-[400px]",
            "pointSize": point_size,
            "onPointClick": js("(e) => console.log('clicked', e)"),
            "highlightColor": [1.0, 1.0, 0.0],
            **cameraProps,
        }
    )


(
    Plot.initialState({"camera": camera})
    | scene(True, 1) & scene(True, 10)
    | scene(False, 4) & scene(False, 8)
)
