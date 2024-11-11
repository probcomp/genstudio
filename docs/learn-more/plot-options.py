# %% [markdown]
# # Plot Options
#
# `genstudio.plot` has many helper functions to specify Observable Plot options.

# %% [markdown]
# ## Size and spacing

# %%
import genstudio.plot as Plot

data = [[1, 1], [2, 2], [3, 3]]

# %%
(Plot.line(data) + Plot.width(200))  # Set width only

# %%
(Plot.line(data) + Plot.height(100))  # Set height only

# %%
(Plot.line(data) + Plot.size(250))  # Set both width and height to same value

# %%
(Plot.line(data) + Plot.size(600, 200))  # Set different width and height

# %% [markdown]
# Control margins (follows CSS margin syntax)

# %%
small_plot = Plot.line(data) + {"className": "bg-blue-200"}
(small_plot + Plot.margin(70))  # All sides

# %%
(small_plot + Plot.margin(30, 100))  # Vertical, horizontal

# %%
(small_plot + Plot.margin(0, 0, 50, 100))  # Top, right, bottom, left

# %% [markdown]
# ## Axes and Grid Options
#
# Customize the appearance of axes and grid:

# %%
# Add or remove grid lines
(Plot.line(data) + Plot.grid())  # Both axes

# %%
(Plot.line(data) + Plot.grid(y=True))  # Y-axis only

# %%
# Hide axes
(Plot.line(data) + Plot.hideAxis())  # Hide both axes

# %%
(Plot.line(data) + Plot.hideAxis(x=True))  # Hide x-axis only

# %%
(Plot.line(data) + Plot.hideAxis(y=True))  # Hide y-axis only

# %%
# Set axis domains
(Plot.line(data) + Plot.domainX([0, 4]))  # Set x-axis range

# %%
(Plot.line(data) + Plot.domainY([0, 4]))  # Set y-axis range

# %%
(Plot.line(data) + Plot.domain([0, 4]))  # Set both axes to same range

# %%
(Plot.line(data) + Plot.domain([0, 4], [0, 3]))  # Set different ranges

# %% [markdown]
# ## Title and Labels
#
# Add descriptive text to your plot:

# %%
# Add titles and captions
(
    Plot.line(data)
    + Plot.title("My Plot")
    + Plot.subtitle("A simple line plot")
    + Plot.caption("Source: Example data")
)

# %% [markdown]
# ## Aspect Ratio
#
# Control the plot's proportions of the x and y scales.

# %%
# Set aspect ratio (width/height)
(Plot.line(data) + Plot.aspectRatio(1.5))  # Make plot 1.5 times wider than tall

# %% [markdown]
# The aspect ratio in Observable Plot controls the relationship between the physical size
# of one unit on the x-axis versus one unit on the y-axis. This is not the same as the
# width/height ratio of the plot container itself. An aspect ratio of 1 can still result
# in a plot with a 2:1 width:height ratio, as seen below, where the domain of x is `[0, 2]`
# while the domain of y is `[0, 1]`.

(Plot.line([[0, 0], [1, 0.5], [2, 1]])) + Plot.aspectRatio(1)

# %% [markdown]
# ## Clipping
#
# Control whether marks extend beyond the plot area:

# %%
# Enable clipping
(Plot.line(data) + Plot.clip())
