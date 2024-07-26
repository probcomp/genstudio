import genstudio.plot as Plot

d = Plot.cache([[1, 2], [3, 4], [5, 6]])

Plot.dot(d, r=40, opacity=0.1) + Plot.dot(d, r=8, opacity=0.5)
