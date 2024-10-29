import genstudio.plot as Plot
from genstudio.plot import js

# %% tags=["hide_source"]
interactivity_warning = Plot.html(
    "div.bg-black.text-white.p-3",
    """This example depends on communication with a python backend, and will not be interactive on the docs website.""",
)

# %% [markdown]
# ## Mouse Events
#
# The `Plot.events` mark supports mouse interactions via the following callbacks:
# - `onDrawStart`
# - `onDraw`
# - `onDrawEnd`
# - `onClick`
# - `onMouseMove
# Each callback receives an event object containing `type` (the event name), `point` (an `[x, y]` array for the current mouse position), and (for draw events only) `startTime`, which can distinguish one draw event from another.

# %%

(
    Plot.initial_state("points", [])
    # Use `Plot.events`, setting $state.points in the `onDraw` callback,
    # which is passed an event containing a `point`, an `[x, y]` array.
    + Plot.events(
        onDrawStart=js("(event) => $state.points = [[event.x, event.y]]"),
        onDraw=js("(event) => $state.points = [...$state.points, [event.x, event.y]]"),
    )
    # Draw a line through all points
    + Plot.line(js("$state.points"), stroke="blue", strokeWidth=4)
    + Plot.ellipse([[1, 1]], r=1, opacity=0.5, fill="red")
    + Plot.domain([0, 2])
)

# %% [markdown]
# ## Drawing


# %%
interactivity_warning


# %% [markdown]
# Say we wanted to pass a drawn path back to Python. We can initialize a ref, with an initial value of an empty list, to hold drawn points. Then, we pass in a python `onDraw` callback to update the points using the widget's `update_state` method. This time, let's add some additional dot marks to make our line more interesting.

# %%
all_points = Plot.ref([])
drawn_points = Plot.ref([])
clicked_points = Plot.ref([])
(
    # Create drawing area and update points on draw
    Plot.events(
        onDraw=lambda event: event["widget"].update_state(
            [drawn_points, "append", [event["x"], event["y"], event["startTime"]]]
        ),
        onMouseMove=lambda event: event["widget"].update_state(
            [all_points, "append", [event["x"], event["y"]]]
        ),
        onClick=lambda event: event["widget"].update_state(
            [clicked_points, "append", [event["x"], event["y"]]]
        ),
    )
    # Draw a continuous line through drawn points
    + Plot.line(drawn_points, z="2")
    # Add small dots for drawn points
    + Plot.dot(drawn_points)
    # Highlight every 6th drawn point in red
    + Plot.dot(
        drawn_points,
        Plot.select(
            js("(indexes) => indexes.filter(i => i % 6 === 0)"),
            {"fill": "red", "r": 10},
        ),
    )
    # Add symbol for clicked points
    + Plot.dot(clicked_points, r=10, symbol="star")
    # Add light gray line for all points
    + Plot.line(all_points, stroke="rgba(0, 0, 0, 0.2)")
    + Plot.domain([0, 2])
)

# %% [markdown]
# The `onDraw` callback function updates the `points` cache with the newly drawn path.
# This triggers a re-render of the plot, immediately reflecting the user's drawing.


# %% [markdown]
# ## Child Events

# %%
interactivity_warning

# %% [markdown]
# This example demonstrates how to create an interactive scatter plot with draggable points. We will use `Plot.renderChildEvents`, a [render transform](https://github.com/observablehq/plot/pull/1811/files#diff-1ca87be5c06a54d3c21471e15cd0d320338916c0f9588fd681a708b7dd2b028b). It handles click and drag events for any mark which produces an ordered list of svg elements, such as `Plot.dot`.

# %% [markdown]
# We first define a [reference](bylight:?match=Plot.ref) with initial point coordinates to represent the points that we want to interact with.

# %%
import genstudio.plot as Plot

data = Plot.ref([[1, 1], [2, 2], [0, 2], [2, 0]])

# %% [markdown]
# Next we define a callback function, which will receive mouse events from our plot. Each event will contain
# information about the child that triggered the event, as well as a reference to the current widget, which has
# a `.update_state` method. This is what allows us to modify the plot in response to user actions.


# %%
def update_position(event):
    widget = event["widget"]
    x = event["x"]
    y = event["y"]
    index = event["index"]
    widget.update_state([data, "setAt", [index, [x, y]]])


# %% [markdown]
# When creating the plot, we pass `Plot.render.childEvents` as a `render` option to the `Plot.dot` mark.
# For demonstration purposes we also include a `Plot.ellipse` mark behind the interactive dots.
# The `Plot.dot` mark updates immediately in JavaScript, while the `Plot.Ellipse` mark updates only in
# response to our callback.

# %%
(
    Plot.ellipse(data, {"fill": "cyan", "fillOpacity": 0.5, "r": 0.2})
    + Plot.dot(
        data,
        render=Plot.renderChildEvents(
            {"onDrag": update_position, "onDragEnd": print, "onClick": print}
        ),
    )
    + Plot.domain([0, 2])
    + Plot.aspectRatio(1)
).display_as("widget")
