# %% [markdown]
# This guide demonstrates:
# 1. Using sliders for dynamic parameter adjustment
# 2. Creating animated sliders
# 3. Using Plot.Frames for frame-by-frame animations

# %%
import genstudio.plot as Plot

# %% [markdown]
# ## Sliders
#
# Sliders allow users to dynamically adjust parameters. Each slider is bound to a reactive variable in `$state`, accessible in Plot.js functions as `$state.{key}`.
#
# Here's an example of a sine wave with an adjustable frequency:

# %%
slider = Plot.Slider(
    key="frequency", label="Frequency", range=[0.5, 5], step=0.1, init=1
)

line = (
    Plot.line(
        {"x": range(100)},
        {
            "y": Plot.js("""(d, i) => {
                return Math.sin(i * 2 * Math.PI / 100 * $state.frequency)
            }""")
        },
    )
    + Plot.domain([0, 99], [-1, 1])
    + {"height": 300, "width": 500}
)

line | slider

# %% [markdown]
# ### Animated Sliders
#
# Sliders can also be used to create animations. When a slider is given an [fps](bylight?match=fps=30) (frames per second) parameter, it automatically animates by updating [its value](bylight?match=$state.frame,key="frame") over time. This approach is useful when all frame differences can be expressed using JavaScript functions that read from $state variables.

# %%
(
    Plot.line(
        {"x": range(100)},
        {
            "y": Plot.js(
                """(d, i) => {
                    return Math.sin(
                        i * 2 * Math.PI / 100 + 2 * Math.PI * $state.frame / 60
                    )
                }"""
            )
        },
    )
    + Plot.domain([0, 99], [-1, 1])
) | Plot.Slider(key="frame", fps=30, range=[0, 59])

# %% [markdown]
# ## Plot.Frames
#
# `Plot.Frames` provides a convenient way to scrub or animate over a sequence of arbitrary plots. Each frame is rendered individually. It implicitly creates a slider and cycles through the provided frames. Here's a basic example:

# %%
import random

shapes = [
    [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)],  # Square
    [(0, 0), (1, 0), (0.5, 1), (0, 0)],  # Triangle
    [(0, 0.5), (0.5, 0), (1, 0.5), (0.5, 1), (0, 0.5)],  # Diamond
    [
        (0, 0.5),
        (0.33, 0),
        (0.66, 0),
        (1, 0.5),
        (0.66, 1),
        (0.33, 1),
        (0, 0.5),
    ],  # Hexagon
]


def show_shapes(color):
    return Plot.Frames(
        [
            Plot.line(shape, fill=color)
            + Plot.domain([-0.1, 1.1], [-0.1, 1.1])
            + {"height": 300, "width": 300, "aspectRatio": 1}
            for shape in shapes
        ],
        fps=2,  # Change shape every 0.5 seconds
    )


show_shapes("blue")

# %% [markdown]
# ## Reactive Variable Lifecycle
#
# > **Note:** this section will only work in a live jupyter session because it depends on python/JavaScript communication.
#
# When using the widget display mode, we can reset the contents of a plot in-place. Reactive variables _maintain their current values_ even when a plot is reset, unless the number or names of the reactive values change.
#
# This allows us to update an in-progress animation without restarting the animation:

# %%
# Create an empty plot:
shapes_plot = Plot.new()
shapes_plot

# %%
import time

# Change the color every 200ms for 5 seconds
start_time = time.time()
duration = 5  # seconds
while time.time() - start_time < duration:
    shapes_plot.reset(
        show_shapes(
            random.choice(["orange", "blue", "green", "yellow", "purple", "pink"])
        )
    )
    time.sleep(0.2)
