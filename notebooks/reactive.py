# %%
# %load_ext autoreload
# %autoreload 2
import genstudio.plot as Plot
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
        enumerate(walk1),
        {
            "x": Plot.js("d => d[0]"),
            "y": Plot.js("d => d[1] * Math.sin($state.effect * d[0] * 0.1)"),
            "stroke": "red",
            "filter": Plot.js("d => d[0] <= $state.time"),
        },
    )
    + Plot.frame()
)

plot2 = (
    Plot.dot(
        enumerate(walk2),
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

plot3 = (
    Plot.areaY(
        enumerate(walk3),
        {
            "x": Plot.js("d => d[0]"),
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

# Create the layout
complex_layout = (plot1 | plot2 | plot3) & (time_slider | effect_slider)

# Display the layout
complex_layout
