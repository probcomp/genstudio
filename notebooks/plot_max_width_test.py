# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Testing Plot maxWidth Feature
#
# This notebook demonstrates the new `maxWidth` constraint for plots.

# %%
import genstudio.plot as Plot
import numpy as np

# %% [markdown]
# ## Test 1: Plot with no width constraints
# This plot will expand to fill the container width

# %%
# Create some sample data
x = np.linspace(0, 10, 100)
y = np.sin(x)
data = {"x": x, "y": y}

basic_plot = (
    Plot.line(data, {"x": "x", "y": "y"})
    + Plot.grid()
    + Plot.title("Plot with no width constraint")
)
basic_plot

# %% [markdown]
# ## Test 2: Plot with maxWidth
# This plot will be constrained to 500px even in a wider container

# %%
constrained_plot = (
    Plot.line(data, {"x": "x", "y": "y"})
    + Plot.grid()
    + Plot.title("Plot with 500px maxWidth")
    + {"maxWidth": 500}
)
constrained_plot

# %% [markdown]
# ## Test 3: Comparison
# Let's put both plots side by side to see the difference

# %%
basic_plot & constrained_plot

# %% [markdown]
# ## Test 4: Different maxWidth values
# Testing various maxWidth constraints

# %%
(
    Plot.line(data, {"x": "x", "y": "y"})
    + Plot.grid()
    + Plot.title("300px maxWidth")
    + {"maxWidth": 300}
) & (
    Plot.line(data, {"x": "x", "y": "y"})
    + Plot.grid()
    + Plot.title("700px maxWidth")
    + {"maxWidth": 700}
)
