import genstudio.plot as Plot

data = [[1, 2], [3, 4], [5, Plot.js("console.log('evaluating cached data') || 6")]]

#
d = Plot.dot(data)
#
Plot.Frames([d for _ in range(6)], key="frame") | Plot.Slider("frame", 5)


dots = Plot.dot(data, r=40, opacity=0.2) + Plot.dot(data, r=8, opacity=0.5)
dots

bg = Plot.line([[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]])

bg

Plot.dot(data, r=40, opacity=0.2) + bg

data = [[1, 2], [3, 4], [5, Plot.js("console.log('evaluating cached data') || 6")]]
bg = Plot.line([[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]])
Plot.new(Plot.cache(Plot.dot(data, r=40, opacity=0.2) + bg))
