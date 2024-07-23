# %load_ext autoreload
# %autoreload 2
import genstudio.plot as Plot
from walkthrough import bean_data_dims
import random


# Generate random walk data
def generate_random_walk(steps, start=0):
    walk = [start]
    for _ in range(steps - 1):
        walk.append(walk[-1] + random.choice([-1, 1]))
    return walk


# Generate three random walks
walk1 = generate_random_walk(100)
walk2 = generate_random_walk(100)
walk3 = generate_random_walk(100)

# Create sliders for interactivity
time_slider = Plot.Slider("time", label="Time", range=[0, 99], fps=30)
effect_slider = Plot.Slider("effect", label="Effect", range=[0, 2], step=0.1, init=1)

# Create three plots with slider interactions
plot1 = (
    Plot.line(
        list(enumerate(walk1)),
        {
            "x": Plot.js("d => d[0]"),
            "y": Plot.js("d => d[1] * Math.sin($state.effect * d[0] * 0.1)"),
            "stroke": "red",
            "filter": Plot.js("d => d[0] <= $state.time"),
        },
    )
    + Plot.frame()
)
#
plot2 = (
    Plot.dot(
        list(enumerate(walk2)),
        {
            "x": Plot.js("d => d[0]"),
            "y": Plot.js("d => d[1] * Math.cos($state.effect * d[0] * 0.1)"),
            "fill": "blue",
            "r": Plot.js("d => 3 + 2 * Math.abs(Math.sin($state.effect * d[0] * 0.1))"),
            "filter": Plot.js("d => d[0] <= $state.time"),
        },
    )
    + Plot.frame()
)
#
plot3 = (
    Plot.areaY(
        list(enumerate(walk3)),
        {
            "y": Plot.js(
                "d => d[1] * (1 + 0.5 * Math.sin($state.effect * d[0] * 0.1))"
            ),
            "y1": 0,
            "fill": "green",
            "fillOpacity": Plot.js("() => 0.3 + 0.2 * Math.sin($state.effect)"),
            "filter": Plot.js("d => d[0] <= $state.time"),
        },
    )
    + Plot.frame()
)

complex_layout = plot1 | (plot2 & plot3) | time_slider | effect_slider

complex_layout

# Plot.Frames creates an animated plot that cycles through a list of frames.
# Each frame can be a plot specification or any renderable object.

# Example of a simple animation with default settings:
Plot.Frames(["a", "b", "c", "d"], fps=2)


# By default, Plot.Frames automatically renders a slider.
# The slider's range is inferred from the number of passed-in frames.

# For more control, you can synchronize multiple Plot.Frames instances
# with an explicitly created slider. To do this:
# 1. Give each Plot.Frames instance the same 'key' argument
# 2. Create a Plot.Slider with that same key

# Example of synchronized animations with custom slider:
(
    Plot.Frames([1, 2, 3, 4, 5], key="custom_animation")
    & Plot.Frames(["A", "B", "C", "D", "E"], key="custom_animation")
    | Plot.Slider("custom_animation", range=[0, 4], fps=5)
)

# In this example, both animations will be controlled by the same slider,
# cycling through their respective frames in sync.

(
    Plot.dot(
        bean_data_dims,
        {
            "x": "day",
            "y": "stem_length",
            "fill": "black",
            "facetGrid": "bean",
            "filter": Plot.js("(d) => d.day <= $state.currentDay"),
        },
    )
    + Plot.frame()
    # animate a $state variable
    | Plot.Slider("currentDay", fps=8, range=bean_data_dims.size("day"))
)
