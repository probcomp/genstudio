import genstudio.plot as Plot

# %% [markdown]

# %% tags=["hide_source"]
Plot.html(
    [
        "div.bg-black.text-white.p-3",
        """This example depends on communication with a python backend, and will not be interactive on the docs website.""",
    ],
)

# %% [markdown]
# When animating using `Plot.Frames` or `Plot.Slider`, the range of the slider can be set dynamically by passing a reference to a list as a `rangeFrom` parameter. If the `tail` option is `True`, the animation will pause at the end of the range, then continue when more data is added to the list.

# %%
from genstudio.plot import js

letters = Plot.ref(["A", "B", "C"], state_key="letters")

tailedSlider = (
    Plot.Slider("n", fps=2, rangeFrom=letters, tail=True, visible=False)
    | ["span", "All letters: ", ["span.text-gray-400", js("$state.letters.toString()")]]
    | ["span", "Current letter: ", js("$state.letters[$state.n]")]
    | ["span", "Current index: ", js("$state.n")]
).display_as("widget")
tailedSlider

# %%
Plot.html(
    [
        "button.bg-blue-500.hover:bg-blue-700.text-white.font-bold.py-2.px-4.rounded",
        {
            "onClick": lambda e: tailedSlider.state.update(
                ["letters", "concat", ["D", "E", "F", "G", "H", "I"]]
            )
        },
        "Add more letters to the widget above",
    ]
)
