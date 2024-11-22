# %%

# in python
import genstudio.plot as Plot
from genstudio.plot import js


def double_clicks(widget, event):
    widget.state.doubled = event["value"] * 2


def square_doubled(widget, event):
    widget.state.squared = event["value"] ** 2


(
    Plot.initialState({"clicks": 0, "doubled": 0, "squared": 0})
    | Plot.onChange({"clicks": double_clicks, "doubled": square_doubled})
    | [
        "div.flex.flex-col.gap-4.p-8",
        [
            "button.px-4.py-2.bg-blue-500.text-white.rounded-md.hover:bg-blue-600",
            {"onClick": js("() => $state.clicks += 1")},
            "Click me!",
        ],
        [
            "div.space-y-2",
            ["div", "Clicks: ", js("$state.clicks")],
            ["div", "Doubled: ", js("$state.doubled")],
            ["div", "Squared: ", js("$state.squared")],
        ],
    ]
)

# %%
# in js

import genstudio.plot as Plot
from genstudio.plot import js

(
    Plot.initialState({"clicks": 0, "doubled": 0, "squared": 0})
    | Plot.onChange(
        {
            "clicks": js("(e) => $state.doubled = e.value * 2"),
            "doubled": js("(e) => $state.squared = e.value ** 2"),
        }
    )
    | [
        "div.flex.flex-col.gap-4.p-8",
        [
            "button.px-4.py-2.bg-blue-500.text-white.rounded-md.hover:bg-blue-600",
            {"onClick": js("() => $state.clicks += 1")},
            "Click me!",
        ],
        [
            "div.space-y-2",
            ["div", "Clicks: ", js("$state.clicks")],
            ["div", "Doubled: ", js("$state.doubled")],
            ["div", "Squared: ", js("$state.squared")],
        ],
    ]
)
