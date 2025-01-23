[![PyPI version](https://badge.fury.io/py/genstudio.svg)](https://badge.fury.io/py/genstudio)

# Gen Studio
_Visualization tools for GenJAX._

-----

`genstudio.plot` provides a composable way to create interactive plots using [Observable Plot](https://observablehq.com/plot/).

Key features:

- Functional, composable plot creation built on Observable Plot (with near 1:1 API correspondence between Python and JavaScript)
- Support for sliders & animations
- Works in Jupyter / Google Colab
- HTML mode which persists plots across kernel restart/shutdown, and a Widget mode which supports Python<>JavaScript interactivity
- Terse layout syntax for organizing plots into rows and columns
- Hiccup implementation for interspersing arbitrary HTML

For detailed usage instructions and examples, refer to the [Gen Studio User Guide](https://studio.gen.dev).

## Development

Run `yarn watch` to compile the JavaScript bundle.

## Credits

- [AnyWidget](https://github.com/manzt/anywidget) provides a nice Python<>JavaScript widget API
- [pyobsplot](https://github.com/juba/pyobsplot) was the inspiration for our Python->JavaScript approach
