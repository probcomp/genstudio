# Gen Studio 
_Visualization tools for GenJAX._

-----

`genstudio.plot` provides a composable way to create interactive plots using [Observable Plot](https://observablehq.com/plot/)
and [AnyWidget](https://github.com/manzt/anywidget), built on the work of [pyobsplot](https://github.com/juba/pyobsplot).

Key features:

- Create plot specifications declaratively by combining marks, options and transformations.
- Render specs to interactive plot widgets, with lazy evaluation and caching.
- Inline specification of extra dimensions (eg. time, sample) to be controlled by a slider or viewed in a grid.
- Includes shortcuts for common options like grid lines, color legends, margins.

Runnable examples are in `notebooks/plot_examples.py`. See [Observable Plot](https://observablehq.com/plot/) for detailed examples of all mark types.

## Installation 

genstudio is published to the same artifact registry as genjax, so you can follow [these instructions](https://github.com/probcomp/genjax?tab=readme-ov-file#quickstart) but use `genstudio` for the package name.

```
genstudio = {version = "v2024.05.23.085705", source = "gcp"}
```


## Usage


Given the following setup:

```py 
import genstudio.plot as Plot
import numpy as np

def normal_100():
  return np.random.normal(loc=0, scale=1, size=1000)
```

### Providing data

Data is provided in the first argument. Three formats are accepted:
- A dict of arrays, directly mapping axes to values.
  ```py 
  Plot.dot({'x': [1, 2], 
            'y': [100, 200]})
  ```
- An array of dicts, whose properties are mapped to axes in the options dict.
  ```py
  Plot.dot([{'t': 1, 'foo': 100}, {'t': 2, 'foo': 200}], 
            {'x': 't', 'y': 'foo'})
  ```
- Some marks support a list of [x, y] points.
  ```py
  Plot.dot([[1, 100], [2, 200]])
  ```  

### Mark examples

1. Histogram

```py
Plot.histogram(normal_100())
```
Specify a number of bins:
```py
Plot.histogram(normal_100(), thresholds=20)
```
2. Scatter plot 

```py
Plot.dot({'x': normal_100(), 'y': normal_100()}) + Plot.frame()
```

3. Compose plots and options

```py 
circle = Plot.dot([[0, 0]], r=100)
circle + Plot.frame() + {'inset': 50}
```

4. Display plot documentation:

```py 
Plot.doc(Plot.line)
```

## With GenJAX

Use `Plot.get_address` to read a path into a genjax trace. Use `Plot.Dimension` to match a segment of the path which has many values, eg:

```py
Plot.get_address(traces, ["ys", Plot.Dimension("samples"), "y", "value"])

// this would correspond to data in a shape like ["ys", i, "y", "value"]
```

`Plot.Dimension` adds a slider to the plot, allowing the user to scrub over data. The key will be used as the slider's label, and can be used in more than one location for data which share the same dimension.