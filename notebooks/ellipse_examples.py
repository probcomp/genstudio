import genstudio.plot as Plot

Plot.configure({"display_as": "widget"})

# Plot ellipses + dots + rects + rules to see how measurements align
data = [[0, 0], [1, 1], [2, 2]]
#
(
    Plot.ruleX([0, 0.5, 1, 1.5, 2, 2.5], stroke="lightgray")
    + Plot.ruleY([0, 0.5, 1, 1.5, 2, 2.5], stroke="lightgray")
    + Plot.dot(data, r=10, fill="black")
    + Plot.rect(
        data,
        {
            "x1": Plot.js("(d) => d[0] - 0.5"),
            "x2": Plot.js("(d) => d[0] + 0.5"),
            "y1": Plot.js("(d) => d[1] - 0.5"),
            "y2": Plot.js("(d) => d[1] + 0.5"),
            "fill": "none",
            "strokeWidth": 4,
            "stroke": "lightgreen",
        },
    )
    + Plot.ellipse(data, {"fill": "blue"}, r=0.5, opacity=0.5, tip=True)
)

# Note that the computed domain/extents do not include the radii.
# AspectRatio=1 ensures a circle.
data = [
    {"x": 0, "y": 0, "rx": 0.5, "ry": 0.5, "fill": "blue"},
    {"x": 2, "y": 2, "rx": 0.5, "ry": 0.5, "fill": "pink"},
]
#
(
    Plot.ellipse(
        data,
        {"x": "x", "y": "y", "rx": "rx", "ry": "ry", "fill": "fill"},
        opacity=0.5,
        tip=True,
    )
)

# Keep scaled_circle but implement on top of ellipse
(
    Plot.scaled_circle(1, 1, 1, fill="red", opacity=0.5, tip=True)
    + Plot.scaled_circle(1.5, 1.5, 1, fill="blue", opacity=0.5, tip=True)
    + Plot.domain([0, 2.5])
    + Plot.aspect_ratio(1)
)

# Example of ellipses with different rx and ry values
data_ellipses = [
    {"x": 1, "y": 1, "rx": 0.8, "ry": 0.3, "fill": "red", "rotate": 0},
    {"x": 2, "y": 2, "rx": 0.3, "ry": 0.8, "fill": "blue", "rotate": 45},
    {"x": 3, "y": 1, "rx": 0.5, "ry": 0.5, "fill": "green", "rotate": 90},
    {"x": 1, "y": 3, "rx": 0.2, "ry": 0.7, "fill": "purple", "rotate": 135},
]
#
(
    Plot.ellipse(
        data_ellipses,
        {
            "x": "x",
            "y": "y",
            "rx": "rx",
            "ry": "ry",
            "fill": "fill",
            "rotate": "rotate",
        },
        opacity=0.7,
        stroke="black",
        strokeWidth=1,
        tip=True,
    )
    + Plot.ruleX([0, 1, 2, 3, 4], stroke="lightgray")
    + Plot.ruleY([0, 1, 2, 3, 4], stroke="lightgray")
    + Plot.dot(data_ellipses, r=3, fill="black")  # Add center points
    + Plot.domain([0, 4])
    + Plot.aspect_ratio(1)
    + Plot.title("Ellipses with Different rx, ry, and rotate values")
).widget()
