# %%

# in python, the closest we have (now) to computed state is chaining onChange listeners.
import genstudio.plot as Plot
from genstudio.plot import js


(
    Plot.initialState({"clicks": 0, "doubled": 0, "squared": 0})
    | Plot.onChange(
        {
            "clicks": lambda widget, event: setattr(
                widget.state, "doubled", event["value"] * 2
            ),
            "doubled": lambda widget, event: setattr(
                widget.state, "squared", event["value"] ** 2
            ),
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

# %%
# in js, we _could_ chain onChange listeners the same way...

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

# %%
# BUT, js already supports computed state:

import genstudio.plot as Plot
from genstudio.plot import js

(
    Plot.initialState(
        {
            "clicks": 0,
            "doubled": js("$state.clicks * 2"),
            "squared": js("$state.doubled ** 2"),
        }
    )
    | Plot.onChange({"squared": lambda w, e: print(e["value"])})
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
