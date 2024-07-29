import genstudio.plot as Plot

Plot.configure({"display_as": "html", "dev": True})

Plot.dot([[1, 1], [2, 3]]).widget()

Plot.dot([[1, 1], [2, 3]]).html()
