# %% [markdown]

# To use GenStudio, first import it:

# %%
import genstudio.plot as Plot
import random

# %% [markdown]
# ## Your first plot

# Let's start with a simple line plot:

# %%

simple_line_data = [[1, 1], [2, 4], [1.5, 7], [3, 10], [2, 13], [4, 15]]
Plot.line(simple_line_data) + {"width": 300, "height": 200} + Plot.frame()

# %% [markdown]
# This creates a basic line plot with three points. We've added a width of 300 pixels and a frame around the plot for better visibility.

# %% [markdown]
# ## Layering Marks and Options

# GenStudio allows you to layer multiple marks and add options to your plots using the `+` operator. For example:

# %%
(
    Plot.line(simple_line_data, {"stroke": "pink", "strokeWidth": 10})
    + Plot.dot(simple_line_data, {"fill": "purple"})
    + {"width": 300, "height": 200}
    + Plot.frame()
)

# %% [markdown]
# In this example, we've layered a dot plot and a line plot, then added plot options for width and height, and finally added a frame.

# %% [markdown]
# ## Understanding Marks

# In GenStudio (and Observable Plot), [marks](https://observablehq.com/plot/marks) are the basic visual elements used to represent data. The `line` we just used is one type of mark. Other common marks include `dot` for scatter plots, `bar` for bar charts, and `text` for adding labels.

# Each mark type has its own set of properties that control its appearance and behavior. For example, with `line`, we can control the stroke color, line width, and curve type.

# To learn more, refer to the [marks documentation](https://observablehq.com/plot/marks).

# ## Specifying Data and Channels

# Channels are how we map our data to visual properties of the mark. For many marks, `x` and `y` are the primary channels, but others like `color`, `size`, or `opacity` are also common.

# Let's look at a few ways to specify data and channels:

# Using a list of objects
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
# In addition to our list of objects, we need to pass a dictionary of options which indicate how to read the channel from a given data point.

# %%
Plot.dot(object_data, {"x": "X", "y": "Y", "fill": "CATEGORY"}) + {
    "width": 300,
    "height": 200,
}

# %% [markdown]
# In this example, we're using a list of objects for our data. In the second argument (the options object), we map each channel to a key in our data objects. For instance, `"x": "X"` tells the plot to use the "X" property of each data object for the x-coordinate of each point.

# We can also use functions to derive channel values:

# %%
Plot.dot(
    object_data,
    {
        "x": "X",
        "y": "Y",
        "r": Plot.js("(d, i) => (i + 1) * 3"),  # radius increases with index
        "fill": "CATEGORY",
    },
) + {"width": 300, "height": 200}

# %% [markdown]
# Here, we're using a JavaScript function for the "r" (radius) channel. This function receives two arguments:
# - `d`: the current data point (one of the objects in our `data` list)
# - `i`: the index of the current data point
#
# The function should return the value for that channel for the given data point. In this case, we're making the radius increase with the index. We use `Plot.js` to write JavaScript code inline.

# %% [markdown]
# We can also use separate lists for each channel (columnar data):

# %%
x = [1, 2, 1.5, 3, 2, 4]
y = [1, 4, 7, 10, 13, 15]

Plot.line({"x": x, "y": y, "strokeWidth": 2, "stroke": "blue"}) + {
    "width": 300,
    "height": 200,
}


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
# You can customize the number of bins:

# %%
Plot.histogram(histogram_data, thresholds=5)

# %% [markdown]
# ### Ellipse

# The `Plot.ellipse` mark allows you to create ellipses or circles on your plot, with sizes that correspond to the x and y domains. This differs from `Plot.dot`, which specifies its radius in pixels.

# An ellipse is defined by its center coordinates (x, y) and its radii. You can specify:
# - A single radius `r` for a circle (which may appear as an ellipse if the aspect ratio is not 1)
# - Separate `rx` and `ry` values for an ellipse with different horizontal and vertical radii
# - An optional rotation angle in degrees

# The data for ellipses can be provided in several formats:
# - An object with columnar data for channels (x, y, rx, ry, rotate)
# - An array of arrays, each inner array in one of these formats:
#   [x, y, r]
#   [x, y, rx, ry]
#   [x, y, rx, ry, rotation]
# - A list of objects containing properties for each ellipse

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
    + {"height": 200}
    + Plot.aspectRatio(1)
)

# %% [markdown]
# You can customize the appearance of ellipses using various options such as fill color, stroke, opacity, and more.

# Here's another example using a list of objects to specify ellipses:

# %%

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
    + {"width": 300}
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

# You can also create custom color scales by specifying a range and an interpolation function:

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
    + Plot.width(400)
    + Plot.height(200)
)

# %% [markdown]
# In this example:
# - We use `colorMap` to assign specific colors to each category.
# - `colorLegend()` adds a color legend to the plot.
# - The `fill` channel is set to "category", which tells the plot to use the category for coloring.
#

# %% [markdown]
# ### Applying a constant color to an entire mark

# `Plot.constantly()` returns a function that always returns the same value, regardless of its input. When used as a channel specifier (like for `fill` or `stroke`), it assigns a single categorical value to the entire mark. This is useful for creating consistent visual elements in your plot. Let's illustrate this with a simple scenario:

# %%

(
    # Walls: a black square outline
    Plot.line(
        [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]],
        {"stroke": Plot.constantly("Walls")},
    )
    # Target: a blue circle at the center
    + Plot.ellipse([[5, 5, 1]], {"fill": Plot.constantly("Target")})
    # Guesses: multiple semi-transparent purple dots
    + Plot.ellipse(
        [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(20)],
        {"fill": Plot.constantly("Guesses"), "r": 0.5, "opacity": 0.2},
    )
    # Define colors and add legend
    + Plot.colorMap({"Walls": "black", "Target": "blue", "Guesses": "purple"})
    + Plot.colorLegend()
    # Set plot dimensions
    + {"width": 400, "height": 400, "aspectRatio": 1}
)

# %% [markdown]
# Key points:
# 1. `Plot.constantly("Walls")` creates a function that always returns "Walls". This is used as the `stroke` value for the entire wall outline.
# 2. Similarly, `Plot.constantly("Target")` and `Plot.constantly("Guesses")` are used for the target and guesses, respectively.
# 3. The `colorMap` defines the actual colors corresponding to these categorical values.
# 4. `colorLegend()` adds a legend showing these color assignments.

# This approach ensures consistent coloring for each element and provides a clear legend for interpretation.

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
    + {"height": 300, "width": 500}
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
