# %%
import genstudio.plot as Plot

# %% tags=["hide_source"]
interactivity_warning = Plot.html(
    "div.bg-black.text-white.p-3",
    """This example depends on communication with a python backend, and will not be interactive on the docs website.""",
)

# %% [markdown]
# ## Sliders
#
# Sliders allow users to dynamically adjust parameters. Each slider is bound to a reactive variable in `$state`, accessible in Plot.js functions as `$state.{key}`.
#
# Here's an example of a sine wave with an adjustable frequency:

# %%
slider = Plot.Slider(
    key="frequency", label="Frequency", range=[0.5, 5], step=0.1, init=1
)

line = (
    Plot.line(
        {"x": range(100)},
        {
            "y": Plot.js(
                """(d, i) => {
                    console.log($state, Math.sin(i * 2 * Math.PI / 100 * $state.frequency))
                return Math.sin(i * 2 * Math.PI / 100 * $state.frequency)
            }"""
            )
        },
    )
    + Plot.domain([0, 99], [-1, 1])
    + {"height": 300, "width": 500}
)

line | slider

# %% [markdown]
# ### Animated Sliders
#
# Sliders can also be used to create animations. When a slider is given an [fps](bylight?match=fps=30) (frames per second) parameter, it automatically animates by updating [its value](bylight?match=$state.frame,key="frame") over time. This approach is useful when all frame differences can be expressed using JavaScript functions that read from $state variables.

# %%
(
    Plot.line(
        {"x": range(100)},
        {
            "y": Plot.js(
                """(d, i) => Math.sin(
                        i * 2 * Math.PI / 100 + 2 * Math.PI * $state.frame / 60
                    )"""
            )
        },
    )
    + Plot.domain([0, 99], [-1, 1])
) | Plot.Slider(key="frame", fps=30, range=[0, 59])

# %% [markdown]
# ## Plot.Frames
#
# `Plot.Frames` provides a convenient way to scrub or animate over a sequence of arbitrary plots. Each frame is rendered individually. It implicitly creates a slider and cycles through the provided frames. Here's a basic example:

# %%
import genstudio.plot as Plot
import random

shapes = [
    [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)],  # Square
    [(0, 0), (1, 0), (0.5, 1), (0, 0)],  # Triangle
    [(0, 0.5), (0.5, 0), (1, 0.5), (0.5, 1), (0, 0.5)],  # Diamond
    [
        (0, 0.5),
        (0.33, 0),
        (0.66, 0),
        (1, 0.5),
        (0.66, 1),
        (0.33, 1),
        (0, 0.5),
    ],  # Hexagon
]


def show_shapes(color):
    return Plot.Frames(
        [
            Plot.line(shape, fill=color)
            + Plot.domain([-0.1, 1.1], [-0.1, 1.1])
            + {"height": 300, "width": 300, "aspectRatio": 1}
            for shape in shapes
        ],
        fps=2,  # Change shape every 0.5 seconds
    )


show_shapes("blue")

# %% [markdown]
# ## Reset a Plot

# %% tags=["hide_source"]
interactivity_warning

# %% [markdown]
# When using the widget display mode, we can reset the contents of a plot in-place. Reactive variables _maintain their current values_ even when a plot is reset.
#
# This allows us to update an in-progress animation without restarting the animation:

# %%
# Create an empty plot:
shapes_plot = Plot.new()
shapes_plot

# %%
import time

# Change the color every 200ms for 5 seconds
start_time = time.time()
duration = 5  # seconds
while time.time() - start_time < duration:
    shapes_plot.reset(
        show_shapes(
            random.choice(["orange", "blue", "green", "yellow", "purple", "pink"])
        )
    )
    time.sleep(0.2)

# %% [markdown]
# ## Plot.ref

# %% tags=["hide_source"]
interactivity_warning


# %% [markdown]
# We can refer to the same value more than once in a plot by wrapping it in `Plot.ref`. The value will only be serialized once. Plot marks are automatically referenced in this way.
#
# `Plot.ref` gives us a "target" inside the widget that we can send updates to, using the `update_state` method. We can send any number of operations, in the form `[ref_object, operation, value]`. The referenced value will change, and affected nodes will re-render.


# %%
numbers = Plot.ref([1, 2, 3])
view = Plot.html("div", numbers).display_as("widget")
view

# %%

view.update_state([numbers, "append", 4])

# %% [markdown]

# There are three supported operations:
# - `"reset"` for replacing the entire value,
# - `"append"` for adding a single value to a list,
# - `"concat"` for adding multiple values to a list.

# If multiple updates are provided, they are applied synchronously.

# One can update `$state` values by specifying the full name of the variable in the first position, eg. `view.update_state(["$state.foo", "reset", "bar"])`.

# %% [markdown]
# ## Tail Data

# %% tags=["hide_source"]
interactivity_warning


# %% [markdown]
# When animating using `Plot.Frames` or `Plot.Slider`, the range of the slider can be set dynamically by passing a reference to a list as a `rangeFrom` parameter. If the `tail` option is `True`, the animation will pause at the end of the range, then continue when more data is added to the list.

# %%
import genstudio.plot as Plot

numbers = Plot.ref([1, 2, 3])
tailedFrames = Plot.Frames(numbers, fps=1, tail=True, slider=False) | Plot.html(numbers)
tailedFrames

# %%
tailedFrames.update_state([numbers, "concat", [7, 8, 9]])

# %%
tailedSlider = (
    Plot.Slider("n", fps=1, rangeFrom=numbers, tail=True, visible=False)
    | Plot.js(f"$state['{numbers.id}'].toString()")
    | Plot.js(f"$state['{numbers.id}'][$state.n]")
).display_as("widget")
tailedSlider

# %%
tailedSlider.update_state([numbers, "concat", [4, 5, 6]])

# %% [markdown]
# ## Mouse Events

# %% tags=["hide_source"]
interactivity_warning

# %% [markdown]
# This example demonstrates how to create an interactive scatter plot with draggable points. We will use `Plot.render.childEvents`, a [render transform](https://github.com/observablehq/plot/pull/1811/files#diff-1ca87be5c06a54d3c21471e15cd0d320338916c0f9588fd681a708b7dd2b028b). It handles click and drag events for any mark which produces an ordered list of svg elements, such as `Plot.dot`.

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
        render=Plot.render.childEvents(
            {"onDrag": update_position, "onDragEnd": print, "onClick": print}
        ),
    )
    + Plot.domain([0, 2])
    + Plot.aspectRatio(1)
).display_as("widget")


# %% [markdown]
# ## Interactive Drawing
#
# The `Plot.draw` mark allows users to draw lines on a plot. By default no line is "drawn"; it's up to you to do something with the `path` included in the event data.
# Supported callbacks: `onDrawStart`, `onDraw`, and `onDrawEnd`.


# %%
import genstudio.plot as Plot

(
    Plot.initial_state("points", [])
    # Use `Plot.draw`, setting $state.points in the `onDraw` callback,
    # which is passed an event containing `path`, an array of [x, y] points.
    + Plot.draw(onDraw=Plot.js("(event) => $state.points = event.path"))
    # Draw a line through all points
    + Plot.line(Plot.js("$state.points"), stroke="blue", strokeWidth=4)
    + Plot.ellipse([[1, 1]], r=1, opacity=0.5, fill="red")
    + Plot.domain([0, 2])
)

# %% [markdown]
# ### With Python in-the-loop


# %%
interactivity_warning


# %% [markdown]
# Say we wanted to pass a drawn path back to Python. We can initialize a ref, with an initial value of an empty list, to hold drawn points. Then, we pass in a python `onDraw` callback to update the points using the widget's `update_state` method. This time, let's add some additional dot marks to make our line more interesting.

# %%
points = Plot.ref([])
(
    # Create drawing area and update points on draw
    Plot.draw(
        onDraw=lambda event: event["widget"].update_state(
            [points, "reset", event["path"]]
        )
    )
    # Draw a continuous line through all points
    + Plot.line(points)
    # Add dots for all points
    + Plot.dot(points)
    # Highlight every 6th point in red
    + Plot.dot(
        points,
        Plot.select(
            Plot.js("(indexes) => indexes.filter(i => i % 6 === 0)"),
            {"fill": "red", "r": 10},
        ),
    )
    + Plot.domain([0, 2])
)

# %% [markdown]
# The `onDraw` callback function updates the `points` cache with the newly drawn path.
# This triggers a re-render of the plot, immediately reflecting the user's drawing.
