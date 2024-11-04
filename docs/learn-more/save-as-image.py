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
