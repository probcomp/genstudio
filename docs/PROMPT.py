# %% [markdown]
# # GenStudio LLM Interaction Guide
#
# ## Library Context
# GenStudio is a visualization library built for interactive data visualization in python using a declarative API. It features a flexible layout system and transparently exposes Observable Plot. It offers state management between python and the javascript view environment.

# Layout components (Row, Column, Grid) allow flexible arrangement of elements and can be composed using & (Row) and | (Column) operators. HTML (via React) can be created using "hiccup in python", and tailwind css classes are supported.

# <examples>
# %%
# <example>
# User: Make a yellow box showing a click count
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
# </examples>
