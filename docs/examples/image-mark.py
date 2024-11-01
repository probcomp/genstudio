import genstudio.plot as Plot

#
(
    # Path from project root (for Jupyter environment)
    Plot.img(["docs/genjax-logo.png"], src=Plot.identity, width=1530, height=330)
    + Plot.aspectRatio(1)
)
