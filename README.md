# Gen Studio 
_Visualization tools for GenJAX._

-----

`gen.studio.plot` provides a composable way to create interactive plots using [Observable Plot](https://observablehq.com/plot/)
and [AnyWidget](https://github.com/manzt/anywidget), built on the work of [pyobsplot](https://github.com/juba/pyobsplot).

Key features:

- Create plot specifications declaratively by combining marks, options and transformations.
- Render specs to interactive plot widgets, with lazy evaluation and caching.
- Inline specification of extra dimensions (eg. time, sample) to be controlled by a slider or viewed in a grid.
- Includes shortcuts for common options like grid lines, color legends, margins.

Runnable examples are in `notebooks/plot_examples.py`. See [Observable Plot](https://observablehq.com/plot/) for detailed examples of all mark types.

## Installation 

gen-studio is published to the same artifact registry as genjax, so you can follow [these instructions](https://github.com/probcomp/genjax?tab=readme-ov-file#quickstart) but use `gen-studio` for the package name.

```
gen-studio = {version = "v2024.05.22.195933", source = "gcp"}
```


## Usage


Given the following setup:

```python 
import gen.studio.plot as Plot
import numpy as np

def normal_100():
  return np.random.normal(loc=0, scale=1, size=1000)
```

1. Histogram

```python 
Plot.rectY(normal_100(), Plot.binX({"y": "count"})) + Plot.ruleY()
```

2. Scatter plot 

```python
Plot.dot(normal_100(), normal_100()) + Plot.frame()
```

3. Compose plots and options

```python 
circle = Plot.dot([[0, 0]], r=100)
circle + Plot.frame() + {'inset': 50}
```

4. Display plot documentation:

```python 
Plot.doc(Plot.line)
```

## With GenJAX

Use `Plot.get_address` to read a path into a genjax trace. Use `Plot.Dimension` to match a segment of the path which has many values, eg:

```python
Plot.get_address(traces, ["ys", Plot.Dimension("samples"), "y", "value"])

// this would correspond to data in a shape like ["ys", i, "y", "value"]
```

`Plot.Dimension` adds a slider to the plot, allowing the user to scrub over data. The key will be used as the slider's
label, and can be used in more than one location for data which share the same dimension.