import genstudio.plot as Plot

# say we have data in format [x1, y1, x2, y2, category]
# each entry represents a single line
data = [
    [0.2, 3.5, 0.7, 3.5, "Apple1"],
    [1.5, 2.0, 2.0, 2.0, "Banana1"],
    [0.8, 4.2, 1.3, 4.2, "Apple2"],
    [2.5, 1.8, 3.0, 1.8, "Candy1"],
    [3.2, 3.7, 3.7, 3.7, "Banana1"],
    [1.0, 0.5, 1.5, 0.5, "Candy1"],
    [2.8, 2.3, 3.3, 2.3, "Apple2"],
    [0.5, 1.2, 1.0, 1.2, "Banana42"],
]

# the least code & easiest to read - one Plot.line mark for each entry.
Plot.new(
    [
        Plot.line(
            [[x1, y1], [x2, y2]],
            # specify the stroke color by passing in a color directly, or
            # (if you want a legend) return an arbitrary identifier wrapped
            # in Plot.constantly. Identifiers are assigned colors automatically
            # or via Plot.color_map as seen below.
            {"stroke": Plot.constantly(category.rstrip("0123456789")), "tip": True},
        )
        for (x1, y1, x2, y2, category) in data
    ],
    # specify colors manually. This is optional.
    Plot.color_map({"Banana": "#FFD700", "Apple": "green", "Candy": "purple"}),
    Plot.color_legend(),
)

# Observable.Plot "prefers" to be given a single (larger) data structure for
# each mark (presumably this improves performance). In the case of a line,
# we can pass in one long list of points, and draw multiple lines in one pass by
# specifying a "z" channel to indicate which points belong to which line.

# approach 1, pass a list of lists. In this case Plot.line assumes the first two
# items in each list/array are x, y.
Plot.line(
    [
        entry
        # enumerate over the data and include the index with each point
        # to use as the z channel
        for index, (x1, y1, x2, y2, category) in enumerate(data)
        for entry in [[x1, y1, category, index], [x2, y2, category, index]]
    ],
    {
        # read the z index from the given entry, could also pass in "3"
        "z": Plot.js("(data) => data[3]"),
        # for "stroke" we pass in a javascript function which
        # - reads the category from the current entry,
        # - strips trailing numbers from it
        # (we can't use Plot.constantly here because we don't have a single
        #  stroke for all the points in the line)
        "stroke": Plot.js("(data) => data[2].replace(/\\d+$/, '')"),
        "tip": True,
    },
) + Plot.color_map({"Banana": "#FFD700", "Apple": "green", "Candy": "purple"})


# approach 2, a list of dicts.
Plot.line(
    [
        entry
        for index, (x1, y1, x2, y2, category) in enumerate(data)
        # Create 2 points for each entry. Include the index - it'll
        # serve as the "z" channel, indicating that these two points belong
        # to their own line.
        for entry in [
            {"x": x1, "y": y1, "category": category, "index": index},
            {"x": x2, "y": y2, "category": category, "index": index},
        ]
    ],
    {  # when passing in dicts rather than lists, x & y channels must be specified explicitly.
        # it appears redundant here because they have the same names in our dicts.
        "x": "x",
        "y": "y",
        "z": "index",
        "stroke": Plot.js("({category}) => category.replace(/\\d+$/, '')"),
        "tip": True,
    },
)

# approach 3, columnar data.
data_columnar = {
    "x": [
        0.2,
        0.7,
        1.5,
        2.0,
        0.8,
        1.3,
        2.5,
        3.0,
        3.2,
        3.7,
        1.0,
        1.5,
        2.8,
        3.3,
        0.5,
        1.0,
    ],
    "y": [
        3.5,
        3.5,
        2.0,
        2.0,
        4.2,
        4.2,
        1.8,
        1.8,
        3.7,
        3.7,
        0.5,
        0.5,
        2.3,
        2.3,
        1.2,
        1.2,
    ],
    "category": [
        "Apple1",
        "Apple1",
        "Banana1",
        "Banana1",
        "Apple2",
        "Apple2",
        "Candy1",
        "Candy1",
        "Banana1",
        "Banana1",
        "Candy1",
        "Candy1",
        "Apple2",
        "Apple2",
        "Banana42",
        "Banana42",
    ],
}
#
Plot.line(
    {
        "x": data_columnar["x"],
        "y": data_columnar["y"],
        "stroke": [
            category.rstrip("0123456789") for category in data_columnar["category"]
        ],
    },
    {
        "tip": True,
        # Specify a z index to group pairs of points into lines.
        # This function uses the point's index (i) to determine if
        # it's the first or second point of a pair. For odd indices
        # (second points), it returns the previous even index to match with its pair
        "z": Plot.js("(_data, i) => Math.floor(i / 2)"),
    },
) + Plot.color_map({"Banana": "#FFD700", "Apple": "green", "Candy": "purple"})
