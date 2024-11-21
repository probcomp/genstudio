# %%
import genstudio.plot as Plot
from genstudio.plot import js

# Example using Plot.cond for if/else rendering.
# Shows how to use cond for more complex conditional logic.

(
    Plot.initialState({"count": 0})
    | Plot.Column(
        [
            "button.p-3.bg-blue-100",
            {"onClick": js("(e) => $state.count += 1")},
            "Click me!",
        ],
        Plot.cond(
            js("$state.count === 0"),
            ["div.p-4.bg-gray-100", "You haven't clicked yet!"],
            js("$state.count < 5"),
            [
                "div.p-4.bg-green-100",
                js("`You've clicked ${$state.count} times - keep going!`"),
            ],
            js("$state.count < 10"),
            ["div.p-4.bg-yellow-100", js("`${$state.count} clicks - getting warmer!`")],
            js("$state.count < 15"),
            ["div.p-4.bg-orange-100", js("`${$state.count} clicks - almost there!`")],
            ["div.p-4.bg-red-100", js("`${$state.count} clicks - you did it!`")],
        ),
    )
)

# %%
# Example using Plot.case

(
    Plot.initialState({"selected": None})
    | Plot.Column(
        [
            "div.p-3.bg-blue-100",
            {"onClick": lambda widget, event: widget.state.update({"selected": "a"})},
            "A",
        ],
        [
            "div.p-3.bg-pink-100",
            {"onClick": lambda widget, event: widget.state.update({"selected": "b"})},
            "B",
        ],
        [
            "div.p-3.bg-green-100",
            {"onClick": lambda widget, event: widget.state.update({"selected": "c"})},
            "C",
        ],
    )
    & Plot.case(
        js("$state.selected"),
        "a",
        [
            "div.flex.items-center.justify-center.bg-blue-50.p-4",
            Plot.js("console.log('you clicked A')"),
            "You selected A!",
        ],
        "b",
        ["div.flex.items-center.justify-center.bg-pink-50.p-4", "You selected B!"],
        "c",
        ["div.flex.items-center.justify-center.bg-green-50.p-4", "You selected C!"],
        [
            "div.flex.items-center.justify-center.bg-gray-100.p-4",
            "← Click a letter to see what happens!",
        ],
    )
)

# %%

# We can also store a more complex value in $state, even from a
# python callback.


def detail_view(content):
    view = (
        Plot.text([content], x=1, y=1, text=Plot.identity, fontSize=40)
        + Plot.dot([[0, 0], [2, 3]])
        + Plot.size(300)
    )
    return lambda widget, event: widget.state.update({"detail": view})


(
    Plot.initialState({"detail": None})
    | Plot.Column(
        ["div.p-3.bg-blue-100", {"onClick": detail_view("a")}, "A"],
        ["div.p-3.bg-pink-100", {"onClick": detail_view("b")}, "B"],
    )
    & js("$state.detail")  # Only shows when detail is truthy
    & {"widths": [1, "auto"]}  # Configure column widths
)

# %%
