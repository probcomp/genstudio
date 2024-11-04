import genstudio.plot as Plot

# ## Color Schemes
#
# ### Using a built-in color scheme
#
# You can use a [built-in color scheme](https://observablehq.com/@d3/color-schemes) from D3.js by specifying the [scheme name](bylight?match="Viridis") in the `color` option:

# %%
(
    Plot.cell(range(20), {"x": Plot.identity, "fill": Plot.identity, "inset": -0.5})
    + {"color": {"type": "linear", "scheme": "Viridis"}}
    + Plot.height(50)
)

# %% [markdown]
# Here, the `x` and `fill` channels both use [Plot.identity](bylight?in=-1) to use the corresponding value in the provided [range](bylight?in=-1&match=range(...\)).
#
# ### Custom color interpolation
#
# You can also create custom color scales by specifying a range and an [interpolation function](bylight?match=Plot.js(...\)):

# %%
(
    Plot.cell(range(20), {"x": Plot.identity, "fill": Plot.identity, "inset": -0.5})
    + {
        "color": {
            "range": ["blue", "red"],
            "interpolate": Plot.js("(start, end) => d3.interpolateHsl(start, end)"),
        }
    }
    + Plot.height(50)
)

# %% [markdown]
# In this example:
# - `"range"` specifies the start and end colors for the scale (blue to red).
# - `"interpolate"` defines how to transition between these colors, using D3's HSL interpolation.
# This results in a smooth color gradient from blue to red across the cells.
#
# ### Using D3 color scales directly
#
# GenStudio allows you to use [D3 color scales](https://github.com/d3/d3-scale-chromatic) directly in your plots:

# %%
(
    Plot.cell(
        range(10),
        {"x": Plot.identity, "fill": Plot.js("(d) => d3.interpolateSpectral(d/10)")},
    )
    + Plot.height(50)
)

# %% [markdown]
# ### Using colorMap and colorLegend
#
# [Plot.colorMap(...)](bylight) assigns specific colors to categories, while [Plot.colorLegend()](bylight) adds a color legend to your plot. In the following example, we create a dot plot with categorical data. The [fill channel](bylight?match="fill":+"category") determines the color of each dot based on its category.

# %%
categorical_data = [
    {"category": "A", "value": 10},
    {"category": "B", "value": 20},
    {"category": "C", "value": 15},
    {"category": "D", "value": 25},
]

(
    Plot.dot(
        categorical_data, {"x": "value", "y": "category", "fill": "category"}, r=10
    )
    + Plot.colorMap({"A": "red", "B": "blue", "C": "green", "D": "orange"})
    + Plot.colorLegend()
)

# %% [markdown]
# ### Applying a constant color to an entire mark
#
# When specifying colors for marks, there's an important distinction to be aware of:
#
# 1. Direct color specification:
#    Use a string to specify a color directly. This will set the color for all points in the mark, but it's not possible to label the color in a legend.
#    Example: `{"fill": "red"}` or `{"fill": "#FF0000"}`
#
# 2. Categorical color assignment:
#    For automatic color assignment based on categories, use a function (or string, for property access) that returns a value for each data point. For a constant category across all data points, use [Plot.constantly(...)](bylight).
#    Example: `{"fill": Plot.constantly("Category A")}`.
#
# `Plot.constantly` returns a function that always returns the same value, regardless of its input. When used as a channel specifier (like for `fill` or `stroke`), it assigns a single categorical value to the entire mark.
#
# Categorical color assignment has the advantage that we can use it with [Plot.colorMap(...)](bylight) to assign specific colors to categories, and [Plot.colorLegend()](bylight) to display the color mappings.

# %%
import random

(
    Plot.line(
        [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]],
        {"stroke": Plot.constantly("Walls")},
    )
    + Plot.ellipse([[5, 5, 1]], {"fill": Plot.constantly("Target")})
    + Plot.ellipse(
        [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(20)],
        {"fill": Plot.constantly("Guesses"), "r": 0.5, "opacity": 0.2},
    )
    + Plot.colorMap({"Walls": "black", "Target": "blue", "Guesses": "purple"})
    + Plot.colorLegend()
    + {"width": 400, "height": 400, "aspectRatio": 1}
)


# %% [markdown]
# ### Using RGB(A) Colors
#
# You can specify colors using RGB or RGBA values by specifying color channel value as CSS rgb/rgba syntax: `"rgb(255, 0, 0)"` or `"rgba(255, 0, 0, 0.5)"`. For example:

# %%
x = [0, 1, 2, 3, 4]
y = [0, 1, 2, 3, 4]
colors = [
    (255, 0, 0),  # Red
    (0, 255, 0),  # Green
    (0, 0, 255),  # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
]

Plot.dot({"x": x, "y": y, "fill": [f"rgb({r}, {g}, {b})" for (r, g, b) in colors]})
