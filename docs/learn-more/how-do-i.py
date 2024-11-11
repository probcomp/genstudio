# %% tags=["hide_source"]
import genstudio.plot as Plot

my_data = [[1, 1], [2, 1.5], [3, 1.25], [4, 2]]
(my_plot := Plot.line(my_data, r=10))

# %% [markdown]
### Show a grid

# %%
my_plot + Plot.grid()

# %% [markdown]
### Label axes

# Add an options object to your plot, putting a "label" in the options for the `"x"` or `"y"` axis:

# %%
my_plot + {"x": {"label": "My x axis"}, "y": {"label": False}}

# %% [markdown]
### Hide axes

# Pass `False` for an axis label (see above), or use `Plot.hideAxis`:

# %%
my_plot + Plot.hideAxis() & my_plot + Plot.hideAxis(y=True)

# %% [markdown]
### Assign colors to particular marks

# - In a mark use `Plot.constantly(<label>)` for a color value like `fill` or `stroke`. The `<label>`, a string, is how you will refer to this color and it will show up in the legend.
# - Then use `Plot.colorMap({<label>: <color>})` to specify a color for that label.
# - Include `Plot.colorLegend()` to show the legend.
# See the [color](./color.html) page for more details.

# %%
(
    Plot.line(my_data, stroke=Plot.constantly("Progress"))
    + Plot.colorMap({"Progress": "orange"})
    + Plot.colorLegend()
)
