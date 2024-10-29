import genstudio.plot as Plot

# %% [markdown]
# ### Histogram

# The `histogram` mark is a convenient extension that combines a [rectY mark](https://observablehq.com/plot/marks/rect) with a [binX transform](https://observablehq.com/plot/transforms/bin).
# It accepts a list or array-like object of values and supports the various [bin options](https://observablehq.com/plot/transforms/bin#bin-options) such as `thresholds`, `interval`, `domain`, and `cumulative` as keyword arguments.

# Here's a basic example:

# %%
histogram_data = [1, 2, 2, 3, 3, 3, 4, 4, 5]
Plot.histogram(histogram_data)

# %% [markdown]
# You can customize the [number of bins](bylight?match=thresholds=5):

# %%
Plot.histogram(histogram_data, thresholds=5)
