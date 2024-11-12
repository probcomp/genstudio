# %% [markdown]
# The `img` mark renders images on a plot. Unlike Observable Plot's built-in image mark,
# this mark accepts width and height in x/y scale units rather than pixels.
#
# Required data channels:
# - `src`: The image source path (as string or function)
# - `width`: Width in x-scale units
# - `height`: Height in y-scale units
#
# Optional parameters:
# - `x`: X coordinate of top-left corner (default: 0)
# - `y`: Y coordinate of top-left corner (default: 0)

# %%
import genstudio.plot as Plot

# allows filepath to work in both VS Code and mkdocs
prefix = "docs" if "__file__" in globals() else ""
(
    Plot.img([prefix + "/genjax-logo.png"], src=Plot.identity, width=1530, height=330)
    + Plot.aspectRatio(1)
)
