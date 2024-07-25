import genstudio.plot as Plot
import numpy as np

# Generate 20 points along a sine wave with a random value
x = np.linspace(0, 2 * np.pi, 20)
y = (np.sin(x) + 1) / 2  # Shift and scale the sine wave to have bottom at 0
v = np.random.random(20)
data = list(zip(x, y, v))

# a built-in color scheme from https://observablehq.com/plot/features/scales#color-scales

(
    Plot.dot(data, {"fill": "1", "r": 8})
    + {"color": {"type": "linear", "scheme": "YlOrBr"}}
)

# interpolate between any two colors

(
    Plot.dot(data, {"fill": "1", "r": 8})
    + {
        "color": {
            "range": ["blue", "red"],
            "interpolate": Plot.js("(start, end) => d3.interpolateHsl(start, end)"),
        }
    }
)

# Use d3 color scales directly - pick from here https://observablehq.com/@d3/color-schemes
ys = np.linspace(0, 1, 10)
(
    Plot.dot(
        [[1, y] for y in ys],
        r=6,
        # for any linear d3 scale
        fill=Plot.js("(d) => d3.interpolateSpectral(d[1])"),
    )
    + Plot.dot(
        [[2, y] for y in ys],
        r=6,
        # for any categorical d3 scale (NOTE: this can be tricky, the indexes don't always start at 0)
        fill=Plot.js("(d, i) => d3.schemeObservable10[i % 10]"),
    )
)
