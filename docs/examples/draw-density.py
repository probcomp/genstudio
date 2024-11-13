# %% [markdown]
# # Interactive Density Plot
#
# This example demonstrates how to create an interactive density plot that generates random points following a normal distribution around clicked locations. It combines several key features:
#
# - State management with `Plot.initialState`
# - Event handling with `Plot.events`
# - Multiple layered marks (density plot and scatter plot)
# - JavaScript-based point generation using Box-Muller transform
#
# Click anywhere on the plot to generate a cluster of normally distributed points:

# %%

import genstudio.plot as Plot
from genstudio.plot import js

# %% [markdown]

# Draw density on a plot:

BUTTON = "div.border.rounded-md.p-5.text-center.font-bold.hover:bg-gray-200"

(
    Plot.initialState(
        {
            "points": [],
            "handleMouse": js("(e) => $state.update(['points', 'append', [e.x, e.y]])"),
        },
        sync={"points"},
    )
    | (
        Plot.density(js("$state.points"))
        + Plot.domain([0, 1])
        + Plot.events(
            {"onDraw": js("$state.handleMouse"), "onClick": js("$state.handleMouse")}
        )
        + Plot.clip()
    )
) | Plot.Row(
    [BUTTON, {"onClick": lambda widget, e: print(widget.state.points)}, "Print"],
    [BUTTON, {"onClick": Plot.js("(e) => $state.points = []")}, "Clear"],
)

# %% [markdown]
# Create a scatter plot with interactive point generation
# %%
(
    Plot.initialState({"points": []}, sync=True)
    | (
        Plot.density(js("$state.points"), fill="density")
        + Plot.colorScheme("Viridis")
        + Plot.colorLegend()
        + Plot.events(
            {
                "onClick": js(
                    """(e) => {
                const std = 0.05;
                const points = Array.from({length: 20}, () => {
                    const r = Math.sqrt(-2 * Math.log(Math.random()));
                    const theta = 2 * Math.PI * Math.random();
                    return [
                        e.x + std * r * Math.cos(theta),
                        e.y + std * r * Math.sin(theta)
                    ];
                });
                $state.update(['points', 'concat', points]);
            }"""
                )
            }
        )
        + Plot.dot(
            js("$state.points"),
            r=6,
            stroke="rgba(255,255,255,0.3)",
            render=Plot.renderChildEvents(
                {
                    "onDrag": js(
                        "(e) => $state.update(['points', 'setAt', [e.index, [e.x, e.y]]])"
                    )
                }
            ),
        )
        + Plot.domain([0, 1])
    )
    | Plot.Row(
        [BUTTON, {"onClick": lambda w, e: print(w.state.points)}, "Print"],
        [BUTTON, {"onClick": Plot.js("(e) => $state.points = []")}, "Clear"],
    )
)

# %% [markdown]
# Prior art: [drawdata](https://calmcode.io/labs/drawdata)
