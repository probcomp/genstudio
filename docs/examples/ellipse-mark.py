import genstudio.plot as Plot

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
