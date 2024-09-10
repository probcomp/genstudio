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
            "y": Plot.js(
                """(d, i) => {
                    console.log($state, Math.sin(i * 2 * Math.PI / 100 * $state.frequency))
                return Math.sin(i * 2 * Math.PI / 100 * $state.frequency)
            }"""
            )
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
                """(d, i) => Math.sin(
                        i * 2 * Math.PI / 100 + 2 * Math.PI * $state.frame / 60
                    )"""
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
import genstudio.plot as Plot
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

# %% tags=["hide_source"]
Plot.html(
    "div.bg-black.text-white.p-3",
    """NOTE: The following examples depend on communication with a python backend, and will not be interactive on the docs website.""",
)

# %% [markdown]
# ## Reactive Variable Lifecycle
#
# When using the widget display mode, we can reset the contents of a plot in-place. Reactive variables _maintain their current values_ even when a plot is reset.
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

# %% [markdown]
# ## Append data to a running animation
#
# We can refer to the same value more than once in a plot by wrapping it in `Plot.cache`. The value will only be serialized once.
# Plot marks are automatically cached in this way.
#
# Later, we can modify cached values and affected views will re-render. To do so, we call the `update_cache` method of the layout, and pass it any number of `[cached_object, operation, value]` lists.
# %%
numbers = Plot.cache([1, 2, 3])
view = Plot.html("div", numbers).display_as("widget")
view

# %%

view.update_cache([numbers, "append", 4])

# %% [markdown]

# There are three supported operations:
# - `"reset"` for replacing the entire value,
# - `"append"` for adding a single value to a list,
# - `"concat"` for adding multiple values to a list.

# If multiple updates are provided, they are applied synchronously.

# One can update `$state` values by specifying the full name of the variable in the first position, eg. `view.update_cache(["$state.foo", "reset", "bar"])`.

# %% [markdown]
# ## Use `tail` with `rangeFrom` to continuously animate data while it is added
# When animating using `Plot.Frames` or `Plot.Slider`, the range of the slider can be set dynamically by passing a reference to a list as a `rangeFrom` parameter. If the `tail` option is `True`, the animation will pause at the end of the range, then continue when more data is added to the list.

# %%
import genstudio.plot as Plot

numbers = Plot.cache([1, 2, 3])
tailedFrames = Plot.Frames(numbers, fps=1, tail=True, slider=False) | Plot.html(numbers)
tailedFrames

# %%
tailedFrames.update_cache([numbers, "concat", [7, 8, 9]])

# %%
tailedSlider = (
    Plot.Slider("n", fps=1, rangeFrom=numbers, tail=True, visible=False)
    | Plot.js(f"$state.cached('{numbers.id}').toString()")
    | Plot.js(f"$state.cached('{numbers.id}')[$state.n]")
).display_as("widget")
tailedSlider

# %%
tailedSlider.update_cache([numbers, "concat", [4, 5, 6]])
