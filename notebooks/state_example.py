import genstudio.plot as Plot

#
render_2 = Plot.cache(Plot.js("console.log('foo') || 'foo: '+$state.foo")) | Plot.cache(
    Plot.js("console.log('bar') || 'bar: '+$state.bar")
)
#
(
    Plot.js("'top'")
    | Plot.Slider("foo", label="foo", init=1)
    | Plot.Slider("bar", label="bar", init=1)
    | Plot.js("console.log('foo') || 'foo: '+$state.foo")
    | Plot.js("console.log('bar') || 'bar: '+$state.bar")
    | render_2
)
