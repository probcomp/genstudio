# %% [markdown]
# # Conditional Rendering

# GenStudio provides two main ways to conditionally render content:
# - `Plot.cond` for if/else logic
# - `Plot.case` for switch/case logic

# ## Plot.cond

# `Plot.cond` takes pairs of conditions and content to render. Each condition is evaluated in order until one returns true.

# Here's a simple example that changes text based on click count:

# %%
import genstudio.plot as Plot

(
    Plot.initialState({"count": 0})
    | Plot.Column(
        [
            "button.p-3.bg-blue-100",
            {"onClick": Plot.js("(e) => $state.count += 1")},
            "Click me!",
        ],
        Plot.cond(
            Plot.js("$state.count === 0"),
            ["div.p-4.bg-gray-100", "You haven't clicked yet!"],
            Plot.js("$state.count < 5"),
            [
                "div.p-4.bg-green-100",
                Plot.js("`You've clicked ${$state.count} times - keep going!`"),
            ],
            ["div.p-4.bg-red-100", Plot.js("`${$state.count} clicks - you did it!`")],
        ),
    )
)

# %% [markdown]
# ## Plot.case

# `Plot.case` provides switch/case-like functionality, matching a value against possible options.
# It's especially useful when working with enumerated states or categorical data.

# %%
import genstudio.plot as Plot

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
    )
    & Plot.case(
        Plot.js("$state.selected"),
        "a",
        ["div.p-4.bg-blue-50", "You selected A!"],
        "b",
        ["div.p-4.bg-pink-50", "You selected B!"],
        ["div.p-4.bg-gray-100", "← Click a letter to see what happens!"],
    )
)

# %% [markdown]
# ## Advanced Usage

# You can store more complex values in state and use them in conditions:


# %%
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
    & Plot.js("$state.detail")
    & {"widths": [1, "auto"]}
)

# %% [markdown]
# ## Conditional marks

# %%
(
    Plot.initialState({"showEllipse": True})
    | Plot.dot([[1, 1], [2, 2]])
    + Plot.cond(Plot.js("$state.showEllipse"), Plot.ellipse([[1.5, 1.5, 0.5]]))
    | [
        "div.p-5.bg-purple-100.text-center.text-lg.font-bold",
        {"onClick": Plot.js("(e) => $state.showEllipse = !$state.showEllipse")},
        "Toggle Ellipse",
    ]
)
