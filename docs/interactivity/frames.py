# %% [markdown]
#
# `Plot.Frames` provides a convenient way to scrub or animate over a sequence of arbitrary plots. Each frame is rendered individually. It implicitly creates a slider and cycles through the provided frames. Here's a basic example:

# %%
import genstudio.plot as Plot

Plot.Frames(
    [
        Plot.html(["div.p-4.bg-gray-200", number])
        for number in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ],
    fps=2,
)

# %% [markdown]
# A slider is implicitly created to control animation. Pass `slider=False` to hide it.

# %%

Plot.Frames(
    [
        Plot.html(["div.p-4.bg-gray-200", number])
        for number in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ],
    fps=2,
    slider=False,
)

# %% [markdown]
# Or, pass a `key` param to specify a `$state` variable that should be used to control the current frame. The current frame can then be controlled from elsewhere. In the following example we increment the `$state.frame` variable using a button, and pass `key="frame"` to `Plot.Frames`.

# %%

(
    Plot.initialState({"frame": 0})
    | Plot.Frames(
        [
            Plot.html(["div.p-4.bg-gray-200", number])
            for number in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        ],
        key="frame",
    )
    | [
        "div.text-white.bg-blue-500.hover:bg-blue-600.p-3.cursor-default",
        {
            "onClick": Plot.js("""(e) => {
            $state.frame = $state.frame < 9 ? $state.frame + 1 : 0
            }""")
        },
        "Next Frame",
    ]
)
