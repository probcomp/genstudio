# %%
import genstudio.plot as Plot

# %% [markdown]
# ## Layout Items

# A `LayoutItem` is the base class for all visual elements in GenStudio that can be composed into layouts.
# It provides core functionality for:
# - Arranging items using `&` (row) and `|` (column) operators
# - Serializing to JSON for rendering via `for_json()`
#
# The most common layout items you'll see are Plot marks (eg. `Plot.line`) and `Plot.html`.
#
# ## Using & and |
#
# Here's a simple example combining two plots into a row:

# %%
plot1 = Plot.html(
    ["div.bg-blue-200.flex.items-center.justify-center.p-5", "Hello, world."]
)
plot2 = Plot.dot([[1, 2], [2, 1]], {"fill": "red"}) + {"height": 200}

plot1 & plot2  # Displays plots side-by-side in a row

# %% [markdown]
# We can also combine them in a column:
# %%
plot1 | plot2  # Displays plots stacked in a column

# %% [markdown]
# Or, use both together:
# %%
(plot1 & plot2) | Plot.html(
    ["div.bg-green-300.p-5", "Welcome to layouts in GenStudio!"]
)

# %% [markdown]
# ## Plot.Grid
# There is also `Plot.Grid`, which accepts accepts any number of children as well as `minWidth` (default: 165px) and `gap` keyword params, and automatically lays out the child elements in a grid while automatically computing the number of columns.

# %%
Plot.Grid(
    Plot.html(["div.bg-red-200.p-5", "A"]),
    Plot.html(["div.bg-orange-200.p-5", "B"]),
    Plot.html(["div.bg-yellow-200.p-5", "C"]),
    Plot.html(["div.bg-green-200.p-5", "D"]),
    Plot.html(["div.bg-blue-200.p-5", "E"]),
    Plot.html(["div.bg-indigo-200.p-5", "F"]),
    Plot.html(["div.bg-purple-200.p-5", "G"]),
    Plot.html(["div.bg-pink-200.p-5", "H"]),
)


# %% [markdown]
# ## Plot.Row and Plot.Column
# `&` and `|` are implemented on top of `Plot.Row` and `Plot.Column`, which can also be used directly:

# %%
Plot.Column(
    Plot.html(["div.bg-red-200.p-5", "A"]),
    Plot.html(["div.bg-orange-200.p-5", "B"]),
    Plot.Row(
        Plot.html(["div.bg-yellow-200.p-5", "C"]),
        Plot.html(["div.bg-green-200.p-5", "D"]),
    ),
)
