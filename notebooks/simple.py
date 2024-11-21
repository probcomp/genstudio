import genstudio.plot as Plot
from math import cos, sin

#
# Generate points for 6 different shapes. [x, y, z] where z is the identity of the shape.
points = [
    # Triangle
    [0, 0, 0],
    [1, 0, 0],
    [0.5, 1, 0],
    [0, 0, 0],
    # Square
    [2, 0, 1],
    [3, 0, 1],
    [3, 1, 1],
    [2, 1, 1],
    [2, 0, 1],
    # Pentagon
    [4, 0.2, 2],
    [4.8, 0.2, 2],
    [5, 0.8, 2],
    [4.4, 1.2, 2],
    [3.8, 0.8, 2],
    [4, 0.2, 2],
    # Diamond
    [0.5, 2, 3],
    [1, 2.5, 3],
    [0.5, 3, 3],
    [0, 2.5, 3],
    [0.5, 2, 3],
    # Hexagon
    [2, 2, 4],
    [2.5, 2, 4],
    [3, 2.5, 4],
    [2.5, 3, 4],
    [2, 3, 4],
    [1.5, 2.5, 4],
    [2, 2, 4],
    # Circle
    *[
        [4 + 0.5 * cos(t * 2 * 3.14159 / 36), 2.5 + 0.5 * sin(t * 2 * 3.14159 / 36), 5]
        for t in range(37)
    ],
]
#
Plot.line(
    points,
    {
        "x": "0",
        "y": "1",  # these can be omitted
        "z": "2",  # Use the z channel to draw separate lines - points sharing the same z are joined
        # also map colors to the shape
        "stroke": Plot.js("""d => {
            const colors = ["blue", "red", "green", "fuschia", "purple", "cyan"];
            return colors[d[2]];  // Use z coordinate (index 2) to pick color
        }"""),
        "fillOpacity": 0.8,
    },
)
