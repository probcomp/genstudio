import genstudio.plot as Plot

#
d = Plot.cache(
    [[1, 2], [3, 4], [5, Plot.js("console.log('evaluating cached data') || 6")]],
    static=False,
)
#
Plot.dot(d, r=40, opacity=0.1) + Plot.dot(d, r=8, opacity=0.5)
