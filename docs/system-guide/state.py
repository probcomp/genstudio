# %% tags=["hide_source"]

import genstudio.plot as Plot
from genstudio.plot import md

# %% [markdown]
# # State
#
# State in GenStudio is used when:
# - Data is used in more than one place, or
# - Data changes over time (via user interaction or from Python)
#
# It provides:
# - Deduplication (during serialization)
# - Efficient change propagation between Python and JavaScript
# - Reactive UI updates within JavaScript

# ## State API
# %% tags=["hide_source"]
Plot.Grid(
    ["div.col-span-2.bg-gray-100.font-bold.p-3", "python: plot definition"],
    md("`Plot.initialState(...)`"),
    "Set initial state",
    md("""`Plot.initialState({"foo": "bar"}, sync={"bar"})`"""),
    md("""...and sync `"bar"` """),
    md("""`Plot.onChange({"x": lambda widget, event: _})`"""),
    md("""Run callback when "x" changes"""),
    md("""`["div", {"onClick": lambda w,e: ...}, "inc"]`"""),
    "Run callback on user events",
    [
        "div.col-span-2.bg-gray-100.font-bold.p-3",
        "python: interacting with live widgets",
    ],
    md("`widget.state.foo`"),
    "Read",
    md("""`widget.state.foo = "baz"`"""),
    "Reset",
    md("""`widget.state.update({"bar": "baz"})`"""),
    "Reset multiple",
    md("""`widget.state.update(["points", "append", [x, y]])`"""),
    "Append, concat, setAt, reset (any number)",
    ["div.col-span-2.bg-gray-100.font-bold.p-3", "javascript"],
    md("""`$state.foo`"""),
    "Read",
    md("""`$state.foo = "baz"`"""),
    "Write",
    md("""`$state.update({"bar": "baz"})`"""),
    "Reset multiple",
    md("""`$state.update(["points", "append", [x, y]])`"""),
    "Append, concat, setAt, reset (any number)",
    widths=["auto", 1],
    gap=4,
    className="text-xs",
)

# %% [markdown]
# ## Minimal example

# %%

import genstudio.plot as Plot

(
    Plot.initialState({"clicks": 0})
    | [
        "div.bg-yellow-200.p-4",
        {"onClick": Plot.js("(e) => $state.clicks = ($state.clicks || 0) + 1")},
        Plot.js("`Clicked ${$state.clicks} times`"),
    ]
    | [
        "div.p-3.border.text-center",
        {"onClick": Plot.js("(e) => $state.update({clicks: 0})")},
        "Reset",
    ]
)

# %% [markdown]
# ## State Updates
#
# State updates in GenStudio work bidirectionally between Python and JavaScript:
#
# ### Python → JavaScript Updates
# When you update state from Python using `widget.state.update()` or by setting attributes directly:
#
# 1. The update is normalized into a list of `[key, operation, payload]` tuples
# 2. For synced state, the update is applied locally to `widget.state`
# 3. The update is serialized to JSON and sent to JavaScript via widget messaging
# 4. Any registered Python listeners are notified
#
# ### JavaScript → Python Updates
# When state is updated from JavaScript using `$state.update()`:
#
# 1. The update is normalized into `[key, operation, payload]` format
# 2. The update is applied locally to the JavaScript state store
# 3. For synced state keys, the update is sent back to Python
# 4. Python applies the update and notifies any listeners
#
# ### Update Operations
# Updates support different operations beyond just setting values:
#
# - `append`: Add item to end of list
# - `concat`: Join two lists together
# - `setAt`: Set value at specific index
# - `reset`: Reset value to initial state
