# %% [markdown]

# Let's begin by importing GenStudio and creating our first plot.

# %% [markdown]

# To use GenStudio, first import it:

# %%
import genstudio.plot as Plot

# %% [markdown]

# Let's start with a simple line plot. Given a dataset of six `[x, y]` coordinates,

# %%

six_points = [[1, 1], [2, 4], [1.5, 7], [3, 10], [2, 13], [4, 15]]

# %% [markdown]

# Here is a line plot:

# %%

Plot.line(six_points)

# %% [markdown]
# ## Understanding Marks

# In GenStudio (and Observable Plot), [marks](https://observablehq.com/plot/features/marks) are the basic visual elements used to represent data. The `line` we just used is one type of mark. Other common marks include `dot` for scatter plots, `bar` for bar charts, and `text` for adding labels.

# Each mark type has its own set of properties that control its appearance and behavior. For example, with `line`, we can control the stroke, stroke width, and curve:

# %%

Plot.line(
    six_points,
    {
        "stroke": "steelblue",  # Set the line color
        "strokeWidth": 3,  # Set the line thickness
        "curve": "natural",  # Set the curve type
    },
)

# %% [markdown]

# To learn more, refer to the [marks documentation](https://observablehq.com/plot/features/marks).

# ## Layering Marks and Options

# We can layer multiple marks and add options to plots using the `+` operator. For example, here we compose a [line mark](bylight?match=Plot.line(...\)) with a [dot mark](bylight?match=Plot.dot(...\)), then add a [frame](bylight?match=Plot.frame(\)):

# %%
(
    Plot.line(six_points, {"stroke": "pink", "strokeWidth": 10})
    + Plot.dot(six_points, {"fill": "purple"})
    + Plot.frame()
)

# %% [markdown]

# ## Specifying Data and Channels

# Channels are how we map our data to visual properties of the mark. For many marks, `x` and `y` are the primary channels, but others like `color`, `size`, or `opacity` are also common. We typically specify our _data_ and _channels_ separately.

# Say we have a list of objects:
# %%

object_data = [
    {"X": 1, "Y": 2, "CATEGORY": "A"},
    {"X": 2, "Y": 4, "CATEGORY": "B"},
    {"X": 1.5, "Y": 7, "CATEGORY": "C"},
    {"X": 3, "Y": 10, "CATEGORY": "D"},
    {"X": 2, "Y": 13, "CATEGORY": "E"},
    {"X": 4, "Y": 15, "CATEGORY": "F"},
]

# %% [markdown]
# A mark takes [data](bylight?match=object_data) followed by an options dictionary, which specifies how [channel names](bylight?match="x","y","stroke","strokeWidth","r","fill") get their values.
#
# There are several ways to specify channel values in Observable Plot:

# 1. A [string](bylight?match="X","Y","CATEGORY") is used to specify a property name in the data object. If it matches, that property's value is used. Otherwise, it's treated as a literal value.
# 2. A [function](bylight?match=Plot.js(...\)) will receive two arguments, `(data, index)`, and should return the desired value for the channel. We use `Plot.js` to insert a JavaScript source string - this function is evaluated within the rendering environment, and not in python.
# 3. An [array](bylight?match=[...]) provides explicit values for each data point. It should have the same length as the list passed in the first (data) position.
# 4. [Other values](bylight?match=8,None) will be used as a constant for all data points.

# %%
Plot.dot(
    object_data,
    {
        "x": "X",
        "y": "Y",
        "stroke": Plot.js("(data, index) => data.CATEGORY"),
        "strokeWidth": [1, 2, 3, 4, 5, 6],
        "r": 8,
        "fill": None,
    },
)

# %% [markdown]
# ## Color Schemes

# %% [markdown]
# ### Using a built-in color scheme

# You can use a [built-in color scheme](https://observablehq.com/@d3/color-schemes) from D3.js by specifying the scheme name in the `color` option:

# %%
(
    Plot.cell(range(20), {"x": Plot.identity, "fill": Plot.identity, "inset": -0.5})
    + {"color": {"scheme": "Viridis"}}
    + Plot.height(50)
)

# %% [markdown]
# In this example, `"x"` and `"fill"` are both set to `Plot.identity`.
# - `"x": Plot.identity` means that the x-position of each cell corresponds directly to its index in the range (0 to 19).
# - `"fill": Plot.identity` also uses the index (0 to 19) to determine the fill color.
# Observable Plot automatically maps this domain (0 to 19) to the `"Viridis"` color scheme.
# The result is a visualization where each cell's position and color represent its index,
# with colors progressing through the Viridis scheme from left to right.

# %% [markdown]
# ### Custom color interpolation

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

# %% [markdown]
# ### Using D3 color scales directly

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

# [Plot.constantly(...)](bylight) returns a function that always returns the same value, regardless of its input. When used as a channel specifier (like for `fill` or `stroke`), it assigns a single categorical value to the entire mark. This is useful for creating consistent visual elements in your plot. Separately, one can use [Plot.colorMap(...)](bylight) to assign specific colors to these categorical values, usually in combination with [Plot.colorLegend()](bylight).

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
# ## Rendering Modes

# GenStudio offers two rendering modes:

# 1. **HTML mode**: Renders visualizations as standalone HTML, ideal for embedding in web pages or exporting. Plots persist across kernel restarts.

# 2. **Widget mode**: Renders visualizations as interactive Jupyter widgets. Enables bidirectional communication between Python and JavaScript.


# You can choose the rendering mode in two ways:

# 1. Globally, using `Plot.configure()`:

# %%
Plot.configure(display_as="widget")  # Set global rendering mode to widget

# %% [markdown]
# 2. Using a plot's [.display_as(...)](bylight) method:

# %%
(
    Plot.dot(categorical_data, {"x": "value", "y": "category", "fill": "category"})
    + Plot.colorLegend()
).display_as("html")  # This specific plot will render as HTML

# %% [markdown]
# The global setting affects all subsequent plots unless overridden by `.display_as()`.
# You can switch between modes as needed for different use cases.
