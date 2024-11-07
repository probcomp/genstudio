# %%
import genstudio.plot as Plot
import numpy as np
from genstudio.plot import js


def generate_pixels(width=100, height=100, num_frames=60):
    # generate a series of images to be animated

    # Create coordinate grids
    x, y = np.meshgrid(np.linspace(-4, 4, width), np.linspace(-4, 4, height))

    # Generate time values
    t = np.linspace(0, 2 * np.pi, num_frames)[:, None, None]

    # Calculate distance from center
    r = np.sqrt(x**2 + y**2)

    # Create a pulsing circular pattern
    intensity = np.sin(r - t) * 255

    # Create RGB channels with different phase shifts
    red = np.clip(intensity * np.sin(t), 0, 255)
    green = np.clip(intensity * np.sin(t + 2 * np.pi / 3), 0, 255)
    blue = np.clip(intensity * np.sin(t + 4 * np.pi / 3), 0, 255)

    # Stack RGB channels
    rgb = np.stack([red, green, blue], axis=-1)

    # Create alpha that fades based on distance from center
    alpha = np.clip(255 * (1 - r / 8), 0, 255)
    alpha = np.broadcast_to(alpha[None, :, :, None], (num_frames, height, width, 1))

    # Combine RGB and alpha
    rgba = np.concatenate([rgb, alpha], axis=-1)

    # Reshape to [frames, pixels*4] and convert to uint8
    return list(rgba.reshape(num_frames, -1).astype(np.uint8))


def render(width=100, height=100, num_frames=30, fps=30):
    """
    Renders an animated visualization of pixel data with interactive controls.

    Creates an interactive plot showing animated pixel data with a purple background,
    frame slider control, and data size indicator. The pixel data is generated using
    the generate_pixels() function.

    Args:
        width (int, optional): Width of the pixel grid in pixels. Defaults to 100.
        height (int, optional): Height of the pixel grid in pixels. Defaults to 100.
        num_frames (int, optional): Number of animation frames to generate. Defaults to 30.
        fps (int, optional): Frames per second for animation playback. Defaults to 30.

    Returns:
        PlotSpec: A composable plot specification containing:
            - Black background rectangle
            - Animated pixel data display
            - Frame slider control
            - Data size indicator in MB
    """
    data = generate_pixels(width=width, height=height, num_frames=num_frames)
    initial_state = Plot.initial_state(
        {"pixels": data, "width": width, "height": height, "frame": 0, "fps": fps}
    )
    return (
        Plot.rect(
            [[0, 0, js("$state.width"), js("$state.height")]],
            x1="0",
            y1="1",
            x2="2",
            y2="3",
            fill="purple",
        )
        + Plot.pixels(
            js("$state.pixels[$state.frame]"),
            imageWidth=js("$state.width"),
            imageHeight=js("$state.height"),
        )
        | initial_state
        | Plot.Slider("frame", rangeFrom=js("$state.pixels"), fps=js("$state.fps"))
        | Plot.html(
            [
                "div.text-gray-500",
                js(
                    "`${($state.width * $state.height * 4 * $state.pixels.length / (1024 * 1024)).toFixed(1)}MB`"
                ),
            ]
        )
    )


plot = render(width=50, height=50, num_frames=10, fps=30)
plot

# %%


W = H = 400  # pick 1000 for the ~max possible message size of 100mb
N = 26
plot.state.update(
    {"pixels": generate_pixels(W, H, N), "width": W, "height": H, "fps": 60}
)
