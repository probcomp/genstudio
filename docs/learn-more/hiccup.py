import genstudio.plot as Plot
from genstudio.plot import html, js

Plot.configure(display_as="html")

# %% [markdown]
# `Plot.html` is a Python implementation of Clojure's [Hiccup](https://github.com/weavejester/hiccup), which uses built-in data structures to represent HTML. The basic format of an html element is a list containing the [element name](bylight:?match="p"), [optional props](bylight:?match=%7B%22style%22%3A%20%7B...%7D%7D), followed by any number of [children](bylight:?match="Hello...!"):

# %%
html(["p", {"style": {"border": "1px solid black"}}, "Hello, world!"])

# %% [markdown]
# ## CSS Classes & Tailwind
#
# Add classes to an element either using the `class` prop, or using the [shorthand syntax](bylight:?match=.bg-black.text-white):

# %%
html(["p.bg-black.text-white", "Hello, world!"])

# %% [markdown]
# [Tailwind](https://tailwindcss.com) css classes are supported (via [twind](https://twind.style)).

# %% [markdown]
# Add attributes and nest elements:

# %%
html(
    [
        "div",
        {"style": {"fontSize": "20px"}},
        [
            "button.bg-blue-500.hover:bg-green-500.text-white.font-bold.py-2.px-4.rounded",
            "Hover me",
        ],
    ]
)

# %% [markdown]
# ## Interactive Elements
#
# Create an interactive slider using reactive `$state`:

# %%
html(
    [
        "div",
        [
            "input",
            {
                "type": "range",
                "min": 0,
                "max": 100,
                "value": js("$state.sliderValue || 0"),
                "onInput": js("(e) => $state.sliderValue = e.target.value"),
            },
        ],
        ["p", js("`Current value: ${$state.sliderValue || 0}`")],
    ]
)

# %% [markdown]
# ## Combining with Observable Plot
#
# Combine Plot.html with Observable Plot:

# %%
(
    Plot.line(
        {"x": range(100)},
        {
            "y": js(
                """(d, i) => {
                return Math.sin(i * 2 * Math.PI / 100 * $state.frequency)
            }"""
            )
        },
    )
    + Plot.domain([0, 99], [-1, 1])
    + {"height": 300, "width": 500}
) | Plot.Slider(key="frequency", label="Frequency", range=[0.5, 5], step=0.1, init=1)
