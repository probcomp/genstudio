# %% [markdown]
# ## Exporting and Saving

# %%
import genstudio.plot as Plot

# %% [markdown]
# To save a plot as an HTML file, use `.save_html(path)`:

# %%
Plot.dot([[1, 1]]).save_html("scratch/foo/basic_plot.html")

# %% [markdown]
# This will create a file named "basic_plot.html" in the current directory containing the interactive visualization.
#
# To save a plot as an image, use `.save_image(path)`:

# %%
Plot.dot([[1, 1]]).save_image("scratch/bar/basic_plot.png", width=800, height=600)

# %% [markdown]
# This will create an image file named "basic_plot.png" with the specified `width` and `height` in pixels. The image will be automatically cropped to remove any transparent regions.
