# %% [markdown]

# This guide demonstrates how to create interactive and animated plots using GenStudio. We'll cover:
# 1. Using sliders for dynamic parameter adjustment
# 2. Creating animated sliders
# 3. Using Plot.Frames for frame-by-frame animations

# %%
import genstudio.plot as Plot

# %% [markdown]
# ## Sliders

# Sliders allow users to dynamically adjust parameters. Each slider is bound to a reactive variable in `$state`, accessible in Plot.js functions as `$state.{key}`.

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

# Sliders can also be used to create animations. When a slider is given an [fps](uplight?dir=down&match=fps=30) (frames per second) parameter, it automatically animates by updating [its value](uplight?dir=down&match=$state.frame,key="frame") over time. This approach is useful when all frame differences can be expressed using JavaScript functions that read from $state variables.

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

# `Plot.Frames` provides a convenient way to scrub or animate over a sequence of arbitrary plots. Each frame is rendered individually. It implicitly creates a slider and cycles through the provided frames. Here's a basic example:

# %%

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

Plot.Frames(
    [
        Plot.line(shape, fill="orange")
        + Plot.domain([-0.1, 1.1], [-0.1, 1.1])
        + {"height": 300, "width": 300, "aspectRatio": 1}
        for shape in shapes
    ],
    fps=2,  # Change shape every 0.5 seconds
)
