# %% [markdown]
# This example demonstrates how binary data (NumPy arrays) can be efficiently transferred
# between Python and JavaScript in both directions. The points are stored as a NumPy array
# in Python, sent to JavaScript for rendering, and any updates from dragging are sent back
# to Python as binary data.

# %%
import genstudio.plot as Plot
import numpy
from genstudio.plot import js

# Generate points as a NumPy array (binary data)
points = numpy.random.rand(8)

# %% [markdown]
# Create an interactive plot that demonstrates bidirectional binary data transfer:
# - Points are sent to JS as binary data via NumPy array
# - Dragging updates are sent back to Python as binary data
# - Print button shows the data is still a NumPy array in Python

# %%
(
    # Plot dots using binary data from state
    Plot.dot(
        {"length": 4},
        x=js("(_, i) => $state.points[i * 2]"),
        y=js("(_, i) => $state.points[(i * 2) + 1]"),
        r=20,
        render=Plot.renderChildEvents(
            onDrag=js("""(e) => {
                 $state.update(["points", "setAt", [e.index * 2, e.x]],
                               ["points", "setAt", [e.index * 2 + 1, e.y]])
                 }""")
        ),
    )
    # Set the plot domain to [0,1]
    + Plot.domain([0, 1])
    # Initialize state with NumPy array and enable binary sync
    | Plot.initialState({"points": points}, sync=True)
    | ["div", js("`javascript type: ${Object.prototype.toString.call($state.points)}`")]
    # Print button shows we still have NumPy array in Python
    | [
        "div.bg-blue-500.text-white.rounded-md.p-5.text-xl.text-center.font-mono",
        {"onClick": lambda w, e: print(w.state.points, type(w.state.points))},
        "🖨️ Print NumPy Array",
    ]
)
