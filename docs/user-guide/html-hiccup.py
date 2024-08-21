import genstudio.plot as Plot
from genstudio.plot import html, js

# %% [markdown]
# # HTML Generation with Plot.html

# Plot.html is a Python implementation of Clojure's Hiccup, enabling HTML structure creation and interactive visualizations. This guide covers:
# 1. Basic HTML generation
# 2. Creating interactive elements
# 3. Combining Plot.html with Observable Plot

# %%
html(["p", "Hello, world!"])

# %% [markdown]
# Add attributes and nest elements:

# %%
html(
    [
        "div",
        {"style": {"color": "blue", "font-size": "20px"}},
        ["p", "This is a blue paragraph"],
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
            "y": js("""(d, i) => {
                return Math.sin(i * 2 * Math.PI / 100 * $state.frequency)
            }""")
        },
    )
    + Plot.domain([0, 99], [-1, 1])
    + {"height": 300, "width": 500}
) | Plot.Slider(key="frequency", label="Frequency", range=[0.5, 5], step=0.1, init=1)
