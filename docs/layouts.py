# %%
import genstudio.plot as Plot

# %% [markdown]
# ## Layout Items

# A `LayoutItem` is the base class for all visual elements in GenStudio that can be composed into layouts.
# It provides core functionality for:
# - Arranging items using `&` (row) and `|` (column) operators
# - Serializing to JSON for rendering via `for_json()`
#
# The most common layout items you'll see are Plot marks (eg. `Plot.line`), `Plot.html`, and `Plot.md`.
#
# ## Using & and |
#
# Here's a simple example combining two plots into a row:

# %%
item1 = Plot.html(
    ["div.bg-blue-200.flex.items-center.justify-center.p-5", "Hello, world."]
)
item2 = Plot.dot([[1, 2], [2, 1]], {"fill": "red"}) + {"height": 200}

item1 & item2  # Displays plots side-by-side in a row

# %% [markdown]
# We can also combine them in a column:
# %%
item1 | item2  # Displays plots stacked in a column

# %% [markdown]
# Or, use both together:
# %%
(item1 & item2) | Plot.html(
    [
        "div.bg-green-300.p-5",
        Plot.md(r"""
$$\oint_{\partial \Omega} \mathbf{E} \cdot d\mathbf{S} = \frac{1}{\epsilon_0} \int_\Omega \rho \, dV$$
"""),
    ]
)

# %% [markdown]
# ## Plot.Grid
# `Plot.Grid` accepts any number of children and automatically lays them out in a grid based on a minimum width (default: 165px), overridable using the `minWidth` keyword argument.

# %%
Plot.Grid(
    Plot.html(["div.bg-red-200.p-5", ["pre", "A\n\n\n\nA"]]),
    Plot.html(["div.bg-orange-200.p-5", "B"]),
    Plot.html(["div.bg-yellow-200.p-5", "C"]),
    Plot.html(["div.bg-green-200.p-5", "D"]),
    Plot.html(["div.bg-blue-200.p-5", "E"]),
    Plot.html(["div.bg-indigo-200.p-5", "F"]),
    Plot.html(["div.bg-purple-200.p-5", "G"]),
    Plot.html(["div.bg-pink-200.p-5", "H"]),
)

# %% [markdown]
# `Plot.Grid` accepts several options to control the layout:
# - `gap`: Sets both horizontal and vertical spacing between items (default: 1)
# - `rowGap`: Vertical spacing between rows, overrides `gap`
# - `colGap`: Horizontal spacing between columns, overrides `gap`
# - `cols`: Fixed number of columns (by default, columns are calculated based on container width)
# - `minCols`: Minimum number of columns (default: 1)
# - `maxCols`: Maximum number of columns

# %%
Plot.Grid(
    Plot.html(["div.bg-red-200.p-5", "A"]),
    Plot.html(["div.bg-orange-200.p-5", "B"]),
    Plot.html(["div.bg-yellow-200.p-5", "C"]),
    Plot.html(["div.bg-green-200.p-5", "D"]),
    rowGap=2,
    colGap=8,
    cols=2,
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

# %% [markdown]
# The `widths` option in `Plot.Row` allows you to specify the width of each child element:
# - Numbers like `1` or `3` are treated as flex grow values, determining how remaining space is distributed
# - String values like `"30px"` set fixed pixel widths
# - String values like `"1/2"` set fractional widths
#
# For example, in this row:
# - The first element gets 1/2 of the space
# - The second element has a fixed width of 30 pixels
# - The third element gets 1 parts of the remaining space
# - The fourth element gets 2 part of the remaining space

# %%
Plot.Row(
    Plot.html(["div.bg-red-200.p-5", "half width"]),
    Plot.html(["div.bg-orange-200.p-5", "70px"]),
    Plot.html(["div.bg-yellow-200.p-5", "flex-1"]),
    Plot.html(["div.bg-green-200.p-5", "flex-2"]),
    widths=["1/2", "70px", 1, 2],
    height="200px",
)

# %% [markdown]
# The `heights` option of `Plot.Column` works the same way:

# %%
import genstudio.plot as Plot

Plot.Column(
    Plot.html(["div.bg-red-200.p-2", "1/2 height"]),
    Plot.html(["div.bg-orange-200.p-2", "70px"]),
    Plot.html(["div.bg-yellow-200.p-2", "1 flex"]),
    Plot.html(["div.bg-green-200.p-2", "2 flex"]),
    heights=["1/2", "70px", 1, 2],
    height="400px",
    width="400px",
) & ["div.flex.items-center.justify-center.h-full", "Hello, world."]
