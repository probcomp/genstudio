import genstudio.plot as Plot

Plot.configure({"display_as": "html"})

Plot.dot([[1, 1], [2, 3]]).widget()

Plot.dot([[1, 1], [2, 3]]).html()
