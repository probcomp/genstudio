import genstudio.plot as Plot

#
render_2 = Plot.cache(Plot.js("console.log('rendering 2') || $state.foo"))
#
(
    Plot.js("'top'")
    | Plot.Slider("foo", init=1)
    | Plot.js("console.log('rendering 1') || $state.foo")
    | render_2
)
