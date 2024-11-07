import genstudio.plot as Plot

# %% tags=["hide_source"]
Plot.html(
    [
        "div.bg-black.text-white.p-3",
        """This example depends on communication with a python backend, and will not be interactive on the docs website.""",
    ],
)
# %% [markdown]
#
# When using the widget display mode, we can reset the contents of a plot in-place. Reactive variables _maintain their current values_ even when a plot is reset. This allows us to update an in-progress animation without restarting the animation.

# Below, we use a Python lambda to call our `render` function on click,
# which resets the contents of `counting_plot`.

# %%
import random

colors = ["orange", "blue", "green", "yellow", "purple", "pink"]

counting_plot = Plot.new()


def render():
    width = 500
    height = 100
    left = Plot.js(
        f"(Math.sin($state.frame * 2 * Math.PI / 60) * 0.5 + 0.5) * ({width} - 50) + 'px'"
    )
    counting_plot.reset(
        (
            Plot.html(
                [
                    f"div.relative.h-[{height}px]",
                    {"onClick": lambda _: render(), "style": {"width": width}},
                    "Click anywhere to change color",
                    [
                        "div.absolute.w-[50px].h-[100%]",
                        {
                            "style": {
                                "background-color": random.choice(colors),
                                "left": left,
                            }
                        },
                    ],
                ]
            )
            | Plot.Slider(key="frame", range=60, fps=24, visible=False)
        )
    )


render()
counting_plot
