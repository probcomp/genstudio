# %% [markdown]

# To use GenStudio, first import it:

# %%
import genstudio.plot as Plot

# %% [markdown]

# Let's start with a simple line plot. First, we'll define our dataset:

# %%

six_points = [[1, 1], [2, 4], [1.5, 7], [3, 10], [2, 13], [4, 15]]

# %% [markdown]

# This `six_points` is a list of `[x, y]` coordinates. Now, let's create a line plot using this data:

# %%

Plot.line(six_points)

# %% [markdown]
# This creates a basic line plot with six points.

# %% [markdown]
# ## Understanding Marks

# In GenStudio (and Observable Plot), [marks](https://observablehq.com/plot/marks) are the basic visual elements used to represent data. The `line` we just used is one type of mark. Other common marks include `dot` for scatter plots, `bar` for bar charts, and `text` for adding labels.

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

# To learn more, refer to the [marks documentation](https://observablehq.com/plot/marks).

# ## Layering Marks and Options

# We can layer multiple marks and add options to plots using the [+](uplight?dir=down&match=%2B) operator. For example, below we'll layer a dot plot with a line plot, then add a frame.

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
# A mark takes [data](uplight?dir=down&match=object_data) followed by an options dictionary, which specifies how [channel names](uplight?dir=down&match="x","y","stroke","strokeWidth","r","fill") get their values.
#
# There are several ways to specify channel values in Observable Plot:

# 1. A [string](uplight?dir=down&match="X","Y","CATEGORY") is used to specify a property name in the data object. If it matches, that property's value is used. Otherwise, it's treated as a literal value.
# 2. A [function](uplight?dir=down&match=Plot.js(...\)) will receive two arguments, `(data, index)`, and should return the desired value for the channel. We use `Plot.js` to insert a JavaScript source string - this function is evaluated within the rendering environment, and not in python.
# 3. An [array](uplight?dir=down&match=[...]) provides explicit values for each data point.
# 4. [Other values](uplight?dir=down&match=8,None) will be used as a constant for all data points.

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
# ## Additional Marks

# GenStudio extends Observable Plot with some additional marks that aren't directly available in the original library.

# %% [markdown]
# ### Histogram

# The `histogram` mark is a convenient extension that combines a [rectY mark](https://observablehq.com/plot/marks/rect) with a [binX transform](https://observablehq.com/plot/transforms/bin).
# It accepts a list or array-like object of values and supports the various [bin options](https://observablehq.com/plot/transforms/bin#bin-options) such as `thresholds`, `interval`, `domain`, and `cumulative` as keyword arguments.

# Here's a basic example:

# %%
histogram_data = [1, 2, 2, 3, 3, 3, 4, 4, 5]
Plot.histogram(histogram_data)

# %% [markdown]
# You can customize the [number of bins](uplight?dir=down&match=thresholds=5):

# %%
Plot.histogram(histogram_data, thresholds=5)

# %% [markdown]
# ### Ellipse

# The `Plot.ellipse` mark allows you to create ellipses or circles on your plot, with sizes that correspond to the x and y domains. This differs from `Plot.dot`, which specifies its radius in pixels.

# An ellipse is defined by its center coordinates (x, y) and its radii. You can specify:
# - A single radius `r` for a circle (which may appear as an ellipse if the aspect ratio is not 1)
# - Separate `rx` and `ry` values for an ellipse with different horizontal and vertical radii
# - An optional rotation angle in degrees

# In addition to the usual channel notation, data for ellipses can be provided as an array of arrays, each inner array in one of these formats:
#   [x, y, r]
#   [x, y, rx, ry]
#   [x, y, rx, ry, rotation]

# Here's an example demonstrating different ways to specify ellipses:

# %%
ellipse_data = [
    [1, 1, 1],  # Circle: x, y, r
    [2, 2, 1, 0.8, 15],  # Ellipse: x, y, rx, ry, rotate
    [3, 3, 1, 0.6, 30],
    [4, 4, 1, 0.4, 45],
    [5, 5, 1, 0.2, 60],
]

(
    Plot.ellipse(
        ellipse_data,
        {"fill": "red", "opacity": 0.5},
    )
    + Plot.domain([0, 7])
    + Plot.aspectRatio(1)
)

# %% [markdown]
# You can customize the appearance of ellipses using various options such as fill color, stroke, opacity, and more.

# Here's another example using a list of objects to specify ellipses:

# %%
import random

ellipse_object_data = [
    {
        "X": random.uniform(0, 4),
        "Y": random.uniform(0, 4),
        "SIZE": random.uniform(0.4, 0.9) ** 2,
    }
    for _ in range(10)
]

(
    Plot.ellipse(
        ellipse_object_data,
        {"x": "X", "y": "Y", "r": "SIZE", "opacity": 0.7, "fill": "purple"},
    )
    + Plot.domain([0, 4])
    + Plot.aspectRatio(1)
)

# %% [markdown]
# In this case, we specify the channel mappings in the options object, telling the plot how to interpret our data.

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

# You can also create custom color scales by specifying a range and an [interpolation function](uplight?dir=down&match=Plot.js(...)):

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

# You can use `colorMap` to manually specify colors for categorical data, and `colorLegend` to display a legend for these colors:

# %%
categorical_data = [
    {"category": "A", "value": 10},
    {"category": "B", "value": 20},
    {"category": "C", "value": 15},
    {"category": "D", "value": 25},
]

(
    Plot.dot(categorical_data, {"x": "value", "y": "category", "fill": "category"})
    + Plot.colorMap({"A": "red", "B": "blue", "C": "green", "D": "orange"})
    + Plot.colorLegend()
)

# %% [markdown]
# In this example:
# - We use `colorMap` to assign specific colors to each category.
# - `colorLegend()` adds a color legend to the plot.
# - The `fill` channel is set to "category", which tells the plot to use the category for coloring.
#

# %% [markdown]
# ### Applying a constant color to an entire mark

# [Plot.constantly()](uplight?dir=down&match=Plot.constantly(...)) returns a function that always returns the same value, regardless of its input. When used as a channel specifier (like for `fill` or `stroke`), it assigns a single categorical value to the entire mark. This is useful for creating consistent visual elements in your plot. Separately, one can use [Plot.colorMap(...)](uplight?dir=down) to assign specific colors to these categorical values, usually in combination with [Plot.colorLegend()](uplight?dir=down).

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
# ## Sliders

# Sliders allow users to adjust parameters dynamically. Each slider has a key (its name) which is bound to a reactive variable contained in `$state`. These variables can be accessed in Plot.js functions using `$state.{key}`. Here's an example of a sine wave with an adjustable frequency:

# %%
# Create a slider for frequency
frequency_slider = Plot.Slider(
    "frequency", label="Frequency", range=[0.5, 5], step=0.1, init=1
)

# Create the plot with slider interaction
(
    Plot.line(
        {"x": range(100)},
        {"y": Plot.js("(d, i) => Math.sin(i * 2 * Math.PI / 100 * $state.frequency)")},
    )
    + Plot.domain([0, 99], [-1, 1])
    + {"height": 300, "width": 500}
) | frequency_slider

# %% [markdown]
# In this example:
# - A slider with the key "frequency" is created using `Plot.Slider("frequency", ...)`.
# - This key is bound to a reactive variable in `$state`.
# - The Plot.js function accesses this variable using `$state.frequency`.
# - The `|` operator combines the plot and slider.

# %% [markdown]
# ### Animated Sliders

# Sliders can also be used to create animations. When a slider is given an `fps` (frames per second) parameter, it automatically animates by updating its value over time. This approach is useful when all frame differences can be expressed using JavaScript functions that read from $state variables.

# %%

(
    Plot.line(
        {"x": range(100)},
        {
            "y": Plot.js(
                "(d, i) => Math.sin(i * 2 * Math.PI / 100 + 2 * Math.PI * $state.frame / 60)"
            )
        },
    )
    + Plot.domain([0, 99], [-1, 1])
) | Plot.Slider("frame", fps=30, range=[0, 59])

# %% [markdown]
# This approach uses a single `Plot.line` with a JavaScript function for the `y` value. The slider controls the `frame` state variable, creating the animation. The `fps=30` parameter causes the slider to automatically update 30 times per second, creating a smooth animation.

# %% [markdown]
# ## Plot.Frames

# Building on the concept of animated sliders, `Plot.Frames` provides a convenient way to create a sequence of plots that animate over time. It implicitly creates a slider and cycles through the provided frames. Here's a basic example:

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

# %% [markdown]
# This example cycles through different shapes, changing every half second. The `fps` parameter controls the speed of the animation, similar to how it works with animated sliders.

# Both animated sliders and Plot.Frames have their uses, and you can choose based on your specific needs. Animated sliders offer more fine-grained control and are useful when you need to manipulate the animation state directly, while Plot.Frames provides a more convenient way to cycle through a predefined set of plots.


# %% [markdown]

# ### barX example

# For a richer example, we'll use some sample data from ancient civilizations.

# %%

civilizations = [
    {
        "name": "Mesopotamia",
        "start": -3500,
        "end": -539,
        "continent": "Asia",
        "peak_population": 10000000,
    },
    {
        "name": "Indus Valley Civilization",
        "start": -3300,
        "end": -1300,
        "continent": "Asia",
        "peak_population": 5000000,
    },
    {
        "name": "Ancient Egypt",
        "start": -3100,
        "end": -30,
        "continent": "Africa",
        "peak_population": 5000000,
    },
    {
        "name": "Ancient China",
        "start": -2070,
        "end": 1912,
        "continent": "Asia",
        "peak_population": 60000000,
    },
    {
        "name": "Maya Civilization",
        "start": -2000,
        "end": 1500,
        "continent": "North America",
        "peak_population": 2000000,
    },
    {
        "name": "Ancient Greece",
        "start": -800,
        "end": -146,
        "continent": "Europe",
        "peak_population": 8000000,
    },
    {
        "name": "Persian Empire",
        "start": -550,
        "end": 651,
        "continent": "Asia",
        "peak_population": 50000000,
    },
    {
        "name": "Roman Empire",
        "start": -27,
        "end": 476,
        "continent": "Europe",
        "peak_population": 70000000,
    },
    {
        "name": "Byzantine Empire",
        "start": 330,
        "end": 1453,
        "continent": "Europe",
        "peak_population": 30000000,
    },
    {
        "name": "Inca Empire",
        "start": 1438,
        "end": 1533,
        "continent": "South America",
        "peak_population": 12000000,
    },
    {
        "name": "Aztec Empire",
        "start": 1428,
        "end": 1521,
        "continent": "North America",
        "peak_population": 5000000,
    },
]

# %% [markdown]

# Below: a [barX](uplight?dir=down&match=Plot.barX) mark specifies [x1 and x2](uplight?dir=down&match="x1","x2") channels to show civilization timespans, with a [text mark](uplight?dir=down&match=Plot.text) providing labels that align with the bars. Both marks use the civilization name for the [y channel](uplight?dir=down&match="y":+"name"). [Color is used](uplight?dir=down&match=Plot.colorLegend(\),"fill":+"continent") to indicate the civilization name, and a legend is provided.

# %%

(
    Plot.barX(
        civilizations,
        {
            "x1": "start",
            "x2": "end",
            "y": "name",
            "fill": "continent",
            "sort": {"y": "x1"},
        },
    )
    + Plot.text(
        civilizations,
        {"text": "name", "x": "start", "y": "name", "textAnchor": "end", "dx": -3},
    )
    + {"axis": None, "marginLeft": 100}
    + Plot.colorLegend()
)
