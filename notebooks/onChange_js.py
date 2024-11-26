import genstudio.plot as Plot
from genstudio.plot import js

(
    Plot.initialState({"position": None, "position_rounded": None})
    | Plot.onChange(
        {
            "position": Plot.js("""
            (e) => {
                $state.position_rounded = e.value?.map(x => Math.round(x));
            }
    """)
        }
    )
    | Plot.dot(
        {"x": [1, 2, 3, 4, 5], "y": [2, 4, 1, 3, 5]},
        {"stroke": "steelblue", "fill": "white"},
    )
    + Plot.events(
        {
            "onMouseMove": js("""(e) => {
            $state.position = [e.x, e.y];
        }""")
        }
    )
    + Plot.grid(True)
    + {"width": 400, "height": 300, "style": {"border": "1px solid #ccc"}}
    | [
        "div.space-y-4",
        [
            "div",
            "Mouse position: ",
            js("$state.position?.join(', ') || 'Move mouse over plot'"),
        ],
        [
            "div",
            "Rounded position: ",
            js("$state.position_rounded?.join(', ') || 'Move mouse over plot'"),
        ],
    ]
)
