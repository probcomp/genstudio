import genstudio.plot as Plot

# %% [markdown]
# ## Sliders
#
# Sliders allow users to dynamically adjust parameters. Each slider is bound to a reactive variable in `$state`, accessible in Plot.js functions as `$state.{key}`.
#
# Here's an example of a sine wave with an adjustable frequency:

# %%
slider = Plot.Slider(
    key="frequency", label="Frequency", range=[0.5, 5], step=0.1, init=1
)

line = (
    Plot.line(
        {"x": range(100)},
        {
            "y": Plot.js(
                """(d, i) => {
                    console.log($state, Math.sin(i * 2 * Math.PI / 100 * $state.frequency))
                return Math.sin(i * 2 * Math.PI / 100 * $state.frequency)
            }"""
            )
        },
    )
    + Plot.domain([0, 99], [-1, 1])
    + {"height": 300, "width": 500}
)

line | slider

# %% [markdown]
# ### Animated Sliders
#
# Sliders can also be used to create animations. When a slider is given an [fps](bylight?match=fps=30) (frames per second) parameter, it automatically animates by updating [its value](bylight?match=$state.frame,key="frame") over time. This approach is useful when all frame differences can be expressed using JavaScript functions that read from $state variables.

# %%
(
    Plot.line(
        {"x": range(100)},
        {
            "y": Plot.js(
                """(d, i) => Math.sin(
                        i * 2 * Math.PI / 100 + 2 * Math.PI * $state.frame / 60
                    )"""
            )
        },
    )
    + Plot.domain([0, 99], [-1, 1])
) | Plot.Slider(key="frame", fps=30, range=[0, 59])
