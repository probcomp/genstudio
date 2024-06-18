# Gen Studio
_Visualization tools for GenJAX._

-----

`genstudio.plot` provides a composable way to create interactive plots using [Observable Plot](https://observablehq.com/plot/)
and [AnyWidget](https://github.com/manzt/anywidget), leveraging the foundational work of [pyobsplot](https://github.com/juba/pyobsplot).

Key features:

- Functional, composable plot creation
- Support for multidimensional data (see `Plot.dimensions`)
- Grid, slider, and animation support

For runnable examples, refer to `notebooks/plot_examples.py`. Detailed examples of all mark types are available at [Observable Plot](https://observablehq.com/plot/).

## Installation

GenStudio is available in the same artifact registry as GenJAX. Follow [these instructions](https://github.com/probcomp/genjax?tab=readme-ov-file#quickstart) and replace `genjax` with `genstudio` as the package name.

## Dev

Run `yarn watch` to compile the javascript bundle.