import genstudio.plot as Plot

#
six_points = [[1, 1], [2, 4], [1.5, 7], [3, 10], [2, 13], [4, 15]]
#
Plot.line(
    six_points,
    {
        "stroke": "steelblue",  # Set the line color
        "strokeWidth": 3,  # Set the line thickness
        "curve": "natural",  # Set the curve type
    },
)
