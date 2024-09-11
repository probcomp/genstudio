import genstudio.plot as Plot

# This example demonstrates how to create an interactive scatter plot with draggable points.

# Create a cached dataset with initial point coordinates
data = Plot.cache([[1, 1], [2, 2], [0, 2], [2, 0]])


# Define a callback function to update the point positions when dragged
def callback(event):
    # Update the cache with the new position of the dragged point
    event["widget"].update_cache(
        [data, "setAt", [event["index"], [event["x"], event["y"]]]]
    )


# Create the plot
(
    # Draw cyan, semi-transparent dots
    Plot.dot(data, fill="cyan", fillOpacity=0.5, r=20)
    # Add draggable dots with the callback function
    + Plot.dot(data, render=Plot.render.draggableChildren({"onMouseUp": callback}))
    # Set the domain of the plot to [0, 2] for both x and y axes
    + Plot.domain([0, 2])
)
