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
