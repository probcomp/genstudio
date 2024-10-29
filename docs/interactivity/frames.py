# %% [markdown]
# ## Plot.Frames
#
# `Plot.Frames` provides a convenient way to scrub or animate over a sequence of arbitrary plots. Each frame is rendered individually. It implicitly creates a slider and cycles through the provided frames. Here's a basic example:

# %%
import genstudio.plot as Plot

shapes = [
    [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)],  # Square
    [(0, 0), (1, 0), (0.5, 1), (0, 0)],  # Triangle
    [(0, 0.5), (0.5, 0), (1, 0.5), (0.5, 1), (0, 0.5)],  # Diamond
    [
        (0, 0.5),
        (0.33, 0),
        (0.66, 0),
        (1, 0.5),
        (0.66, 1),
        (0.33, 1),
        (0, 0.5),
    ],  # Hexagon
]


def show_shapes(color):
    return Plot.Frames(
        [
            Plot.line(shape, fill=color)
            + Plot.domain([-0.1, 1.1], [-0.1, 1.1])
            + {"height": 300, "width": 300, "aspectRatio": 1}
            for shape in shapes
        ],
        fps=2,  # Change shape every 0.5 seconds
    )


show_shapes("blue")
