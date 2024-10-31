import genstudio.plot as Plot
from genstudio.plot import js

w = (
    Plot.initialState({"x": 0, "y": 0, "clicked": []}, sync=True)
    | (
        Plot.dot(js("$state.clicked"))
        + Plot.events(
            {
                "onMouseMove": js("(e) => $state.update({x: e.x, y: e.y})"),
                "onClick": js(
                    "(e) => $state.update(['clicked', 'append', [e.x, e.y] ])"
                ),
            }
        )
        + Plot.domain([0, 1])
    )
).onChange({"clicked": print})
w
