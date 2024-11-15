# %% [markdown]
# # GenStudio LLM Interaction Guide
#
# ## Library Context
# GenStudio is a visualization library built for interactive data visualization in python using a declarative API. It features a flexible layout system and transparently exposes Observable Plot. It offers state management between python and the javascript view environment.
#
# Layout components (Row, Column, Grid) allow flexible arrangement of elements and can be composed using & (Row) and | (Column) operators. HTML (via React) can be created using "hiccup in python", and tailwind css classes are supported.
#
# There is a "state" api across both Python and JavaScript. In Python one sets initial state by including `Plot.initialState({key: value})` as a layout item (can be anywhere), and `Plot.onChange({key: callback})` to invoke functions when state changes. These onChange Callbacks receive (widget, event) arguments where event contains `"value"`. In Python one reads state via `widget.key`, resets via `widget.key = foo`, set multiple values via `widget.update({key: value})` or pass `widget.update` any number of `[key, operation, payload]` lists where operation can be "append", "concat", "setAt", or "reset". In JavaScript one reads state via `$state.key`, write via `$state.key = foo`, and updates via `$state.update({key: value})` or pass $state.update any number of operations as in python, eg. `$state.update(["foo", "append", 1], ["bar", "concat", [1, 2]], ["baz", "setAt", [index, value]])`.
#
# The `Plot.Slider` component not only shows a slider but can also cycle a state variable through a range of values at a specified fps, eg. `Plot.Slider(key, init, [0, 10], fps=3)` will cycle `key` from 0-10 inclusive starting at `init`. `fps="ref"` is also valid and uses requestAnimationFrame instead of a numeric interval. Range can be list of two values, `[from, until]` (inclusive), to be traversed by `step`. Or a single value `n` which becomes `[from, n-1]`, aligned with python's range(n). Thus if one wants to show a series of 'frames' one can do `Plot.Slider("frame", range=len(frames))` to cycle through all the valid indexes of `frames`.
# <examples>
# %%
# <example>
# User: Make a clickable yellow box that shows how many times it has been clicked.
# Assistant:
import genstudio.plot as Plot
from genstudio.plot import js

(
    Plot.initialState({"clicks": 0})
    | [
        "div",
        {
            "class": "bg-yellow-200 p-4",
            "onClick": Plot.js("(e) => $state.clicks = ($state.clicks || 0) + 1"),
        },
        Plot.js("`Clicked ${$state.clicks} times`"),
    ]
)
# </example>
# %%
# <example>
# User: Show an empty scatter plot. Let the user add points by clicking.
# Assistant:
import genstudio.plot as Plot

(
    Plot.initialState({"points": []})
    | Plot.dot((js("$state.points")))
    + Plot.events(
        {"onClick": js("(e) => $state.update(['points', 'append', [e.x, e.y]])")}
    )
    # explicitly set a domain to make 'space' for clicking
    + Plot.domain([0, 1])
)
# User: Great. Can you let me drag them around?
# Assistant:
import genstudio.plot as Plot

(
    Plot.initialState({"points": []})
    | Plot.dot(
        (Plot.js("$state.points")),
        r=10,  # larger, easier to drag
        render=Plot.renderChildEvents(
            {
                "onDrag": Plot.js(""" (e) => {
                $state.update(["points", "setAt", [e.index, [e.x, e.y]]])
               }""")
            }
        ),
    )
    + Plot.events(
        {"onClick": js("(e) => $state.update(['points', 'append', [e.x, e.y]])")}
    )
    + Plot.domain([0, 1])
)
# </example>
# %%
# <example>
# User: Show a scatter plot of points in a stair-like pattern next to Gauss's law in LaTeX. Add pastel background colors.
# Assistant:
import genstudio.plot as Plot
import numpy as np

x = np.arange(10) / 10
points = np.c_[
    x, x + (np.arange(10) % 3) / 10
]  # Creates stair-like pattern by adding 0, 0.1, 0.2 cyclically

plot1 = Plot.dot(points) + {"className": "bg-green-100"}
plot2 = Plot.md(
    r"""
$$\oint_{\partial \Omega} \mathbf{E} \cdot d\mathbf{S} = \frac{1}{\epsilon_0} \int_\Omega \rho \, dV$$
""",
    className="bg-yellow-100 p-3 flex items-center",
)

plot1 & plot2
# </example>
# %%
# <example>
# User: Show a green square connected to a blue circle using a thick red line. Show a legend. Build up the plot from small/composable parts.
# Assistant:
import genstudio.plot as Plot

# we'll specify colors using Plot.constantly and Plot.colorMap
# (if we use literal color values they won't show up in the legend)
square_plot = Plot.rect(
    [[0, 0.5, 0.5, 1]],
    x1="0",
    y1="1",
    x2="2",
    y2="3",
    # "Square" is the name of the color that will show up in the legend,
    # and is what we must specify in Plot.colorMap
    fill=Plot.constantly("Square"),
)

ellipse_plot = Plot.ellipse([[1, 0.5, 0.25]], fill=Plot.constantly("Circle"))

line_plot = Plot.line(
    [[0.25, 0.75], [1, 0.5]], strokeWidth=10, stroke=Plot.constantly("Line")
)

(
    square_plot
    + ellipse_plot
    + line_plot
    + Plot.colorMap({"Square": "green", "Circle": "blue", "Line": "red"})
    + Plot.colorLegend()
    + Plot.domain(x=[0, 1.5], y=[0, 1])
    + Plot.aspectRatio(1)  # necessary for square + circle to render proportionally
)
# </example>
# %%
# <example>
# User: Create an animated pixel display showing circular waves
# Assistant:
import genstudio.plot as Plot
import numpy as np
from genstudio.plot import js


def generate_pixels(width=100, height=100, num_frames=60):
    x, y = np.meshgrid(np.linspace(-4, 4, width), np.linspace(-4, 4, height))
    t = np.linspace(0, 2 * np.pi, num_frames)[:, None, None]
    r = np.sqrt(x**2 + y**2)

    intensity = np.sin(r - t) * 255  # Radial wave pattern

    rgb = np.stack(
        [
            np.clip(intensity * np.sin(t + phase), 0, 255)
            for phase in [0, 2 * np.pi / 3, 4 * np.pi / 3]
        ],
        axis=-1,
    )
    # return a list of frames; each frame is
    # a flat array of rgb or rgba values for Plot.pixel
    return list(rgb.reshape(num_frames, -1).astype(np.uint8))


width = 50
height = 50
data = generate_pixels(width=width, height=height, num_frames=60)

(
    Plot.initialState(
        {"pixels": data, "width": width, "height": height, "frame": 0, "fps": 30}
    )
    | Plot.pixels(
        # reads the current frame from our generated pixels
        js("$state.pixels[$state.frame]"),
        imageWidth=js("$state.width"),
        imageHeight=js("$state.height"),
    )
    | Plot.Slider(
        "frame",
        rangeFrom=js("$state.pixels"),
        # could also show "slider" if desired
        controls=["play", "fps"],
        fps="raf",  # can also use a numeric value
    )
)

# </example>
# %%
# <example>
# User: Create an alphabet slideshow (controlled by click)
# Assistant:
import genstudio.plot as Plot
from genstudio.plot import js

(
    Plot.initialState({"frame": 0, "letters": "ABCDEFGHIJKLMNOPQRSTUVWXYZ"})
    | [
        "div.flex.items-center.justify-center.font-bold.select-none",
        {
            "onClick": js("""(e) => {
          e.preventDefault();
          const midPoint = e.currentTarget.offsetWidth / 2;
          if (e.clientX < midPoint) {
            // Click on left side - go back
            $state.frame = (prevValue) => prevValue === 0 ? $state.letters.length - 1 : prevValue - 1;
          } else {
            // Click on right side - go forward
            $state.frame = (prevValue) => prevValue === $state.letters.length - 1 ? 0 : prevValue + 1;
          }
          }""")
        },
        ["div.text-[120px]", js("$state.letters[$state.frame]")],
        [
            "div.text-xl.absolute.left-0.top-0.bottom-0.right-[50%].flex.justify-start.items-center.p-2.opacity-0.hover:opacity-50",
            "←",
        ],
        [
            "div.text-xl.absolute.left-[50%].top-0.bottom-0.right-0.flex.justify-end.items-center.p-2.opacity-0.hover:opacity-50",
            "→",
        ],
    ]
)
# </example>
# %%
# <example>
# User: Create a simple live code editor that evaluates Python expressions
# Assistant:
import genstudio.plot as Plot
from genstudio.plot import js
import io
import sys

initial_source = """x = 42
y = [1, 2, 3]
print(f"x={x}, y={y}")"""


def with_out_str(fn):
    """evaluates fn and returns retval, stdout"""
    stdout = io.StringIO()
    sys.stdout = stdout
    try:
        value = fn()
    finally:
        out = stdout.getvalue()
        sys.stdout = sys.__stdout__
    return value, out


def evaluate(widget, _e):
    try:
        # Execute code and capture stdout
        _, out = with_out_str(lambda: exec(widget.state.toEval))

        # Update state with output if any, otherwise show "No output"
        widget.state.update({"result": out if out else "No output"})
    except Exception as e:
        widget.state.update({"result": f"Error: {str(e)}"})


(
    Plot.initialState({"result": "", "toEval": "", "source": initial_source})
    | Plot.onChange({"toEval": evaluate})
    | Plot.html(
        [
            "form.!flex.flex-col.gap-3",
            {
                "onSubmit": js(
                    "e => { e.preventDefault(); $state.toEval = $state.source}"
                )
            },
            [
                "div.bg-gray-100.p-4.rounded",
                js("$state.result || 'Result will appear here'"),
            ],
            [
                "textarea.whitespace-pre-wrap.text-[13px].p-3.rounded-md.bg-gray-100.h-[100px].font-mono",
                {
                    "onChange": js("(e) => $state.source = e.target.value"),
                    "value": js("$state.source"),
                    "onKeyDown": js(
                        "(e) => { if (e.ctrlKey && e.key === 'Enter') { e.stopPropagation(); $state.toEval = $state.source } }"
                    ),
                },
            ],
            [
                "div.flex.items-stretch.gap-2",
                [
                    "button.flex-auto.!bg-blue-500.!hover:bg-blue-600.text-white.text-center.px-4.py-2.rounded-md.cursor-pointer",
                    {"type": "submit"},
                    "Evaluate (Ctrl+Enter)",
                ],
                [
                    "button.px-4.py-2.border.rounded-md",
                    {
                        "onClick": js(
                            "() => $state.source = `"
                            + initial_source.replace("`", "\\`")
                            + "`"
                        )
                    },
                    "Reset",
                ],
            ],
        ]
    )
)
# </example>
# %%
# <example>
# <given-user-file-context>
civilizations = [
    {
        "name": "Mesopotamia",
        "start": -3500,
        "end": -539,
        "continent": "Asia",
        "peak_population": 10000000,
    },
    {
        "name": "Indus Valley Civilization",
        "start": -3300,
        "end": -1300,
        "continent": "Asia",
        "peak_population": 5000000,
    },
    {
        "name": "Ancient Egypt",
        "start": -3100,
        "end": -30,
        "continent": "Africa",
        "peak_population": 5000000,
    },
    {
        "name": "Ancient China",
        "start": -2070,
        "end": 1912,
        "continent": "Asia",
        "peak_population": 60000000,
    },
    {
        "name": "Maya Civilization",
        "start": -2000,
        "end": 1500,
        "continent": "North America",
        "peak_population": 2000000,
    },
    {
        "name": "Ancient Greece",
        "start": -800,
        "end": -146,
        "continent": "Europe",
        "peak_population": 8000000,
    },
]
# </given-user-file-context>
# User: Create a timeline visualization of the ancient civilizations I've defined above.
# Assistant:
import genstudio.plot as Plot

(
    Plot.barX(
        civilizations,
        {
            "x1": "start",
            "x2": "end",
            "y": "name",
            "fill": "continent",
            "sort": {"y": "x1"},
        },
    )
    + Plot.text(
        civilizations,
        {"text": "name", "x": "start", "y": "name", "textAnchor": "end", "dx": -3},
    )
    + {"axis": None, "marginLeft": 100}
    + Plot.colorLegend()
)
# </example>
# %%
# <example>
# User: What kinds of plots can I create with GenStudio? Please show me some examples.
# Assistant:
import genstudio.plot as Plot
import numpy as np

# Generate some sample data
x = np.linspace(0, 10, 100)
y = np.sin(x)
points = np.column_stack([x, y])
categories = ["A", "B", "C", "D"]
values = [23, 45, 12, 78]
cat_data = [{"category": c, "value": v} for c, v in zip(categories, values)]

# Create a grid of common plot types
(
    Plot.initialState({"title": "Common Plot Types"})
    | Plot.Grid(
        # Line plot
        Plot.line(points) + Plot.title("Line"),
        # Scatter plot
        Plot.dot(points) + Plot.title("Scatter"),
        # Area plot
        Plot.area(points) + Plot.title("Area"),
        # Bar chart
        Plot.barY(cat_data, x="category", y="value") + Plot.title("Bar"),
        # Histogram
        Plot.rectY(np.random.random(1000), Plot.binX({"y": "count"}))
        + Plot.title("Histogram"),
        # Box plot
        Plot.boxY(
            [{"group": "A", "value": v} for v in np.random.normal(0, 1, 100)]
            + [{"group": "B", "value": v} for v in np.random.normal(2, 0.5, 100)],
            x="group",
            y="value",
        )
        + Plot.title("Box"),
        # Heatmap
        Plot.rect(
            [
                [i, j, np.sin(i / 5) * np.cos(j / 5)]
                for i in range(20)
                for j in range(20)
            ],
            x="0",
            y="1",
            fill="2",
        )
        + Plot.title("Heatmap"),
        # Bubble plot
        Plot.dot(
            [
                [np.random.random(), np.random.random(), np.random.random() * 30]
                for _ in range(50)
            ],
            x="0",
            y="1",
            r="2",
        )
        + Plot.title("Bubble"),
        cols=3,
    )
    | Plot.md("""
GenStudio provides access to the full Observable Plot API, which includes many plot types:

- Basic: line, dot (scatter), area, rect
- Statistical: histogram, box, bin, density
- Categorical: bar, group, stack
- Specialized: hexbin, contour, arrow, link, vector
- And more: text, rule, frame, image, delaunay, voronoi

Each plot can be customized with various options for color, size, labels, etc.
See the [Observable Plot API documentation](https://observablehq.com/@observablehq/plot-api-reference)
for the complete reference.

Plots can be composed using:
- `+` operator to layer marks
- `&` operator to arrange horizontally
- `|` operator to arrange vertically
- `Plot.Grid()` for grid layouts

Interactive features include:
- Tooltips and hover effects
- Pan and zoom
- Click and drag selection
- Animation and transitions
- State management between Python and JavaScript
""")
)
# </example>
# %%
# <example>
# <given-user-file-context>
# scientific discoveries with years until widespread acceptance
discoveries = [
    {
        "year": 1687,
        "acceptance_delay": 20,
        "scientist": "Newton",
        "discovery": "Laws of Motion",
        "description": "Published Principia Mathematica, establishing the laws of motion and universal gravitation. Took ~20 years to be widely accepted due to complexity and need for mathematical verification.",
    },
    {
        "year": 1859,
        "acceptance_delay": 70,
        "scientist": "Darwin",
        "discovery": "Evolution",
        "description": "Published On the Origin of Species. Faced significant resistance. took ~70 years before evolutionary theory was broadly accepted by the scientific community.",
    },
    {
        "year": 1905,
        "acceptance_delay": 15,
        "scientist": "Einstein",
        "discovery": "Special Relativity",
        "description": "Introduced special relativity. Took ~15 years for broad acceptance, requiring experimental confirmation like the 1919 eclipse observations.",
    },
    {
        "year": 1925,
        "acceptance_delay": 30,
        "scientist": "Heisenberg",
        "discovery": "Quantum Mechanics",
        "description": "Formulated uncertainty principle. Took ~30 years for quantum mechanics to be fully accepted due to its counterintuitive nature.",
    },
    {
        "year": 1953,
        "acceptance_delay": 5,
        "scientist": "Watson & Crick",
        "discovery": "DNA Structure",
        "description": "Discovered DNA's double helix structure. Quick acceptance (~5 years) due to clear experimental evidence and immediate practical applications.",
    },
]
# </given-user-file-context>
# User: Could you help me plot this dataset, showing a timeline of when each discovery was published and the period until it gained widespread acceptance? Include tooltips with the detailed descriptions.
# Assistant:
import genstudio.plot as Plot

(
    Plot.initialState({"discoveries": discoveries})
    |
    # Background grid
    Plot.gridY(tickSpacing=20)
    +
    # Timeline bars showing acceptance period
    Plot.rect(
        Plot.js("$state.discoveries"),
        {
            "x1": "year",
            "x2": Plot.js("d => d.year + d.acceptance_delay"),
            "y": "discovery",
            "height": 15,
            "fill": "steelblue",
            "fillOpacity": 0.3,
            "title": Plot.js(
                "d => `${d.discovery}\nPublished: ${d.year}\nFully accepted: ${d.year + d.acceptance_delay}`"
            ),
        },
    )
    +
    # Publication points
    Plot.dot(
        Plot.js("$state.discoveries"),
        {"x": "year", "y": "discovery", "r": 6, "fill": "steelblue"},
    )
    +
    # Tooltips with detailed information
    Plot.tip(
        Plot.js("$state.discoveries"),
        Plot.pointer(
            {
                "x": "year",
                "y": "discovery",
                "title": Plot.js(
                    "d => `${d.discovery} (${d.year})\n\n${d.description}`"
                ),
            }
        ),
    )
    +
    # Customize plot appearance
    {
        "y": {
            "label": None
        },  # Remove y-axis label since discovery names are self-explanatory
        "x": {"label": "Year", "grid": True},
        "marginLeft": 120,  # Make room for discovery names
        "height": 300,
    }
    | Plot.md("""
This timeline shows when major scientific discoveries were published (dots) and how long they took to achieve widespread acceptance (bars).
Hover over elements to see details about each discovery.
    """)
)
# </example>
# %%
# </examples>
