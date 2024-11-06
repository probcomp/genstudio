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
    return rgba.reshape(num_frames, -1).astype(np.uint8)


def render(width=100, height=100, num_frames=30, fps=30):
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
            fill="black",
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

# these numbers produce the ~maximum possible message size,
# just under 100mb.
W = 1000
H = 1000
N = 27
plot.state.update(
    {"pixels": generate_pixels(W, H, N), "width": W, "height": H, "fps": 60}
)
