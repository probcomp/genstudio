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
#
# In GenStudio (and Observable Plot), [marks](https://observablehq.com/plot/features/marks) are the basic visual elements used to represent data. The `line` we just used is one type of mark. Other common marks include `dot` for scatter plots, `bar` for bar charts, and `text` for adding labels.
#
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
#
# ## Layering Marks and Options
#
# We can layer multiple marks and add options to plots using the `+` operator. For example, here we compose a [line mark](bylight?match=Plot.line(...\)) with a [dot mark](bylight?match=Plot.dot(...\)), then add a [frame](bylight?match=Plot.frame(\)):

# %%
(
    Plot.line(six_points, {"stroke": "pink", "strokeWidth": 10})
    + Plot.dot(six_points, {"fill": "purple"})
    + Plot.frame()
)

# %% [markdown]
# Plots are immutable; we often reuse layers repeatedly throughout a notebook as we gradually built up a more complex visualization, or show different phenomena on top of the same background layer.

# %%
import random

walls = (
    Plot.line(
        [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0], [0, 5], [5, 5], [5, 10]],
        {"stroke": "black", "strokeWidth": 2},
    )
    + Plot.hideAxis()
)
walls

# %%
# Generate a cluster of random points within the larger room
center_x, center_y = random.uniform(0, 10), random.uniform(5, 10)
positions = Plot.dot(
    [
        [center_x + random.uniform(-0.5, 0.5), center_y + random.uniform(-0.5, 0.5)]
        for _ in range(10)
    ],
    {"r": 5, "fill": "red"},
)

# plot our positions cluster on top of the walls plot
walls + positions

# %% [markdown]
# ## Specifying Data and Channels
#
# Channels are how we map our data to visual properties of the mark. For many marks, `x` and `y` are the primary channels, but others like `color`, `size`, or `opacity` are also common. We typically specify our _data_ and _channels_ separately.
#
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
#
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
# ## Data Serialization
#
# Data is passed to the JavaScript runtime as JSON, using the [orjson](https://github.com/ijl/orjson) library with additional fallback behaviour:
#
# | Condition | Conversion |
# |-----------|------------|
# | Object has `for_json` method | `$object.for_json()` |
# | Object has `tolist` method | `$object.tolist()` |
# | Object is iterable | `list($object)` |
# | Datetime objects | Converted to a JavaScript `Date` |
# | Callable objects | Converted to JavaScript functions that return values to Python <br/> (only for "widget" display mode)|
#
# Alternate modes of serialization (eg. for better performance with larger datasets) are possible but not yet implemented.
#


# %% [markdown]
# ## Rendering Modes
#
# GenStudio offers two rendering modes:
#
# 1. **HTML mode**: Renders visualizations as standalone HTML, ideal for embedding in web pages or exporting. Plots persist across kernel restarts.
#
# 2. **Widget mode**: Renders visualizations as interactive Jupyter widgets. Enables bidirectional communication between Python and JavaScript.
#
# You can choose the rendering mode in two ways:
#
# 1. Globally, using `Plot.configure()`:

# %%
Plot.configure(display_as="widget")  # Set global rendering mode to widget

# %% [markdown]
# 2. Using a plot's [.display_as(...)](bylight) method:

# %%
categorical_data = [
    {"category": "A", "value": 10},
    {"category": "B", "value": 20},
    {"category": "C", "value": 15},
    {"category": "D", "value": 25},
]
(
    Plot.dot(categorical_data, {"x": "value", "y": "category", "fill": "category"})
    + Plot.colorLegend()
).display_as("html")

# %% [markdown]
# The global setting affects all subsequent plots unless overridden by `.display_as()`.
# You can switch between modes as needed for different use cases.
#
# ## Exporting and Saving
#
# GenStudio provides methods to save your visualizations as standalone HTML files or images.
#
# To save a plot as an HTML file, use [.save_html(...)](bylight:):

# ```
# Plot.dot([[1, 1]]).save_html("basic_plot.html")
# ````
#
# This will create a file named "basic_plot.html" in the current directory containing the interactive visualization.
#
# To save a plot as an image, use [.save_image(...)](bylight:):
#
# ```
# Plot.dot([[1, 1]]).save_image("basic_plot.png", width=800, height=600)
# ```
#
# This will create an image file named "basic_plot.png" with the specified `width` and `height` in pixels. The image will be automatically cropped to remove any transparent regions.
