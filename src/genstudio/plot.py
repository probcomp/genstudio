# %%
# ruff: noqa: F401
import copy
import json
import random
from typing import Any, Dict, List, Union, TypeAlias, Sequence

import genstudio.plot_defs as plot_defs
from genstudio.js_modules import JSCall, JSRef, js
from genstudio.plot_defs import (
    area,
    areaX,
    areaY,
    arrow,
    auto,
    autoSpec,
    axisFx,
    axisFy,
    axisX,
    axisY,
    barX,
    barY,
    bin,
    binX,
    binY,
    bollinger,
    bollingerX,
    bollingerY,
    boxX,
    boxY,
    cell,
    cellX,
    cellY,
    centroid,
    circle,
    cluster,
    column,
    contour,
    crosshair,
    crosshairX,
    crosshairY,
    delaunayLink,
    delaunayMesh,
    density,
    differenceY,
    dodgeX,
    dodgeY,
    dotX,
    dotY,
    filter,
    find,
    formatIsoDate,
    formatMonth,
    formatWeekday,
    geo,
    geoCentroid,
    graticule,
    gridFx,
    gridFy,
    gridX,
    gridY,
    group,
    groupX,
    groupY,
    groupZ,
    hexagon,
    hexbin,
    hexgrid,
    hull,
    image,
    initializer,
    interpolatorBarycentric,
    interpolatorRandomWalk,
    legend,
    line,
    linearRegressionX,
    linearRegressionY,
    lineX,
    lineY,
    link,
    map,
    mapX,
    mapY,
    marks,
    normalize,
    normalizeX,
    normalizeY,
    plot,
    pointer,
    pointerX,
    pointerY,
    raster,
    rect,
    rectX,
    rectY,
    reverse,
    scale,
    select,
    selectFirst,
    selectLast,
    selectMaxX,
    selectMaxY,
    selectMinX,
    selectMinY,
    shiftX,
    shuffle,
    sort,
    sphere,
    spike,
    stackX,
    stackX1,
    stackX2,
    stackY,
    stackY1,
    stackY2,
    text,
    textX,
    textY,
    tickX,
    tickY,
    tip,
    transform,
    tree,
    treeLink,
    treeNode,
    valueof,
    vector,
    vectorX,
    vectorY,
    voronoi,
    voronoiMesh,
    window,
    windowX,
    windowY,
)
from genstudio.layout import Column, Row, Slider, Hiccup
from genstudio.plot_spec import PlotSpec, new
from genstudio.util import configure, deep_merge
from genstudio.widget import cache

# This module provides a composable way to create interactive plots using Observable Plot
# and AnyWidget, built on the work of pyobsplot.
#
# See:
# - https://observablehq.com/plot/
# - https://github.com/manzt/anywidget
# - https://github.com/juba/pyobsplot
#
#
# Key features:
# - Create plot specifications declaratively by combining marks, options and transformations
# - Compose plot specs using + operator to layer marks and merge options
# - Render specs to interactive plot widgets, with lazy evaluation and caching
# - Easily create grids to compare small multiples
# - Includes shortcuts for common options like grid lines, color legends, margins

d3 = JSRef("d3")
Math = JSRef("Math")
View = JSRef("View")


def repeat(data):
    """
    For passing columnar data to Observable.Plot which should repeat/cycle.
    eg. for a set of 'xs' that are to be repeated for each set of `ys`.
    """
    return View.repeat(data)


class Dimensioned:
    def __init__(self, value, path):
        self.value = value
        self.dimensions = [
            rename_key(segment, ..., "key")
            for segment in path
            if isinstance(segment, dict)
        ]

    def shape(self):
        shape = ()
        current_value = self.value
        for dimension in self.dimensions:
            if "leaves" not in dimension:
                shape += (len(current_value),)
                current_value = current_value[0]
        return shape

    def names(self):
        return [
            dimension.get("key", dimension.get("leaves"))
            for dimension in self.dimensions
        ]

    def __repr__(self):
        return f"<Dimensioned shape={self.shape()}, names={self.names()}>"

    def size(self, name):
        names = self.names()
        shape = self.shape()
        if name in names:
            return shape[names.index(name)]
        raise ValueError(f"Dimension with name '{name}' not found")

    def flatten(self):
        # flattens the data in python, rather than js.
        # currently we are not using/recommending this
        # but it may be useful later or for debugging.
        leaf = (
            self.dimensions[-1]["leaves"]
            if isinstance(self.dimensions[-1], dict) and "leaves" in self.dimensions[-1]
            else None
        )
        dimensions = self.dimensions[:-1] if leaf else self.dimensions

        def _flatten(value, dims, prefix=None):
            if not dims:
                value = {leaf: value} if leaf else value
                return [prefix | value] if prefix else [value]
            results = []
            dim_key = dims[0]["key"]
            for i, v in enumerate(value):
                new_prefix = {**prefix, dim_key: i} if prefix else {dim_key: i}
                results.extend(_flatten(v, dims[1:], new_prefix))
            return results

        return _flatten(self.value, dimensions)

    def to_json(self):
        return {"value": self.value, "dimensions": self.dimensions}


def dimensions(data, dimensions=[], leaves=None):
    """
    Attaches dimension metadata, for further processing in JavaScript.
    """
    dimensions = [{"key": d} for d in dimensions]
    dimensions = [*dimensions, {"leaves": leaves}] if leaves else dimensions
    return Dimensioned(data, dimensions)


def rename_key(d, prev_k, new_k):
    return {k if k != prev_k else new_k: v for k, v in d.items()}


def get_choice(ch, path):
    ch = ch.get_sample() if getattr(ch, "get_sample", None) else ch

    def _get(value, path):
        if not path:
            return value
        segment = path[0]
        if not isinstance(segment, dict):
            return _get(value(segment), path[1:])
        elif ... in segment:
            v = value.get_value()
            if hasattr(value, "get_submap") and v is None:
                v = value.get_submap(...)
            return _get(v, path[1:])
        elif "leaves" in segment:
            return value
        else:
            raise TypeError(
                f"Invalid path segment, expected ... or 'leaves' key, got {segment}"
            )

    value = _get(ch, path)
    value = value.get_value() if hasattr(value, "get_value") else value

    if any(isinstance(elem, dict) for elem in path):
        return Dimensioned(value, path)
    else:
        return value


def is_choicemap(data):
    current_class = data.__class__
    while current_class:
        if current_class.__name__ == "ChoiceMap":
            return True
        current_class = current_class.__base__
    return False


def get_in(data: Union[Dict, Any], path: List[Union[str, Dict]]) -> Any:
    data = data.get_sample() if getattr(data, "get_sample", None) else data  # type: ignore
    if is_choicemap(data):
        return get_choice(data, path)

    def process_segment(value: Any, remaining_path: List[Union[str, Dict]]) -> Any:
        for i, segment in enumerate(remaining_path):
            if isinstance(segment, dict):
                if ... in segment:
                    if isinstance(value, list):
                        return [
                            process_segment(v, remaining_path[i + 1 :]) for v in value
                        ]
                    else:
                        raise TypeError(
                            f"Expected list at path index {i}, got {type(value).__name__}"
                        )
                elif "leaves" in segment:
                    return value  # Leaves are terminal, no further traversal
                else:
                    raise TypeError(
                        f"Invalid path segment, expected ... or 'leaves' key, got {segment}"
                    )
            else:
                value = value[segment]
        return value

    value = process_segment(data, path)

    if any(isinstance(elem, dict) for elem in path):
        return Dimensioned(value, path)
    else:
        return value


# Test case to verify traversal of more than one dimension
def test_get_in():
    data = {"a": [{"b": [{"c": 1}, {"c": 2}]}, {"b": [{"c": 3}, {"c": 4}]}]}

    result = get_in(data, ["a", {...: "first"}, "b", {...: "second"}, "c"])
    assert isinstance(
        result, Dimensioned
    ), f"Expected Dimensioned, got {type(result).__name__}"
    assert result.value == [
        [1, 2],
        [3, 4],
    ], f"Expected [[1, 2], [3, 4]], got {result.value}"
    assert isinstance(
        result.dimensions, list
    ), f"Expected dimensions to be a list, got {type(result.dimensions).__name__}"
    assert (
        len(result.dimensions) == 2
    ), f"Expected 2 dimensions, got {len(result.dimensions)}"
    assert (
        [d["key"] for d in result.dimensions] == ["first", "second"]
    ), f"Expected dimension keys to be ['first', 'second'], got {[d['key'] for d in result.dimensions]}"

    flattened = get_in(
        data, ["a", {...: "first"}, "b", {...: "second"}, "c", {"leaves": "c"}]
    ).flatten()
    assert flattened == [
        {"first": 0, "second": 0, "c": 1},
        {"first": 0, "second": 1, "c": 2},
        {"first": 1, "second": 0, "c": 3},
        {"first": 1, "second": 1, "c": 4},
    ], f"Expected flattened result to be [{{...}}, ...], got {flattened}"

    def test_deeper_nesting():
        data = {
            "x": [
                {"y": [{"z": [{"a": 5}, {"a": 6}]}, {"z": [{"a": 7}, {"a": 8}]}]},
                {"y": [{"z": [{"a": 9}, {"a": 10}]}, {"z": [{"a": 11}, {"a": 12}]}]},
            ]
        }

        result = get_in(
            data,
            ["x", {...: "level1"}, "y", {...: "level2"}, "z", {...: "level3"}, "a"],
        )
        assert isinstance(result, Dimensioned), "Expected Dimensioned object"
        assert result.value == [
            [[5, 6], [7, 8]],
            [[9, 10], [11, 12]],
        ], f"Expected nested list of values, got {result.value}"
        assert len(result.dimensions) == 3, "Expected 3 dimensions"
        assert [d["key"] for d in result.dimensions] == [
            "level1",
            "level2",
            "level3",
        ], "Dimension keys do not match expected values"

        flattened = get_in(
            data,
            [
                "x",
                {...: "level1"},
                "y",
                {...: "level2"},
                "z",
                {...: "level3"},
                "a",
                {"leaves": "a"},
            ],
        ).flatten()
        assert flattened == [
            {"level1": 0, "level2": 0, "level3": 0, "a": 5},
            {"level1": 0, "level2": 0, "level3": 1, "a": 6},
            {"level1": 0, "level2": 1, "level3": 0, "a": 7},
            {"level1": 0, "level2": 1, "level3": 1, "a": 8},
            {"level1": 1, "level2": 0, "level3": 0, "a": 9},
            {"level1": 1, "level2": 0, "level3": 1, "a": 10},
            {"level1": 1, "level2": 1, "level3": 0, "a": 11},
            {"level1": 1, "level2": 1, "level3": 1, "a": 12},
        ], f"Expected flattened result to be [{{...}}, ...], got {flattened}"

    test_deeper_nesting()

    print("tests passed")


def ellipse(values, options: dict[str, Any] = {}, **kwargs) -> PlotSpec:
    return PlotSpec(
        JSCall("View", "MarkSpec", ["ellipse", values, {**options, **kwargs}])
    )


def scaled_circle(x, y, r, **kwargs):
    return ellipse([[x, y]], r=r, **kwargs)


def constantly(x):
    """
    Returns a javascript function which always returns `x`.

    Typically used to specify a constant property for all values passed to a mark,
    eg. plot.dot(values, fill=plot.constantly('My Label')). In this example, the
    fill color will be assigned (from a color scale) and show up in the color legend.
    """
    x = json.dumps(x)
    return js(f"()=>{x}")


def Grid(plotspecs, plot_opts={}, layout_opts={}):
    return Hiccup(
        View.Grid,
        {
            "specs": plotspecs,
            "plotOptions": plot_opts,
            "layoutOptions": layout_opts,
        },
    )


def small_multiples(plotspecs, plot_opts={}, layout_opts={}):
    return Grid(
        plotspecs,
        plot_opts={**plot_opts, "smallMultiples": True},
        layout_opts=layout_opts,
    )


def dot(values, options={}, **kwargs):
    return plot_defs.dot(values, {"fill": "currentColor", **options, **kwargs})


dot.__doc__ = plot_defs.dot.__doc__


def histogram(
    values,
    mark="rectY",
    thresholds="auto",
    layout={"width": 200, "height": 200, "inset": 0},
    **plot_opts,
):
    """
    Create a histogram plot from the given values.

    Args:

    values (list or array-like): The data values to be binned and plotted.
    mark (str): 'rectY' or 'dot'.
    thresholds (str, int, list, or callable, optional): The thresholds option may be specified as a named method or a variety of other ways:

    - 'auto' (default): Scott’s rule, capped at 200.
    - 'freedman-diaconis': The Freedman–Diaconis rule.
    - 'scott': Scott’s normal reference rule.
    - 'sturges': Sturges’ formula.
    - A count (int) representing the desired number of bins.
    - An array of n threshold values for n - 1 bins.
    - An interval or time interval (for temporal binning).
    - A function that returns an array, count, or time interval.

     Returns:
      PlotSpec: A plot specification for a histogram with the y-axis representing the count of values in each bin.
    """
    opts = deep_merge({"x": {"thresholds": thresholds}, "tip": True}, plot_opts)
    if mark == "rectY":
        return rectY(values, binX({"y": "count"}, opts)) + ruleY([0]) + layout
    elif mark == "dot":
        return dot(values, binX({"r": "count"}, opts))


def frame(options={}, **kwargs):
    return plot_defs.frame({"stroke": "#dddddd", **options, **kwargs})


def ruleY(values, options: dict[str, Any] = {}, **kwargs):
    return plot_defs.ruleY(values or [0], options, **kwargs)


def ruleX(values, options: dict[str, Any] = {}, **kwargs):
    return plot_defs.ruleX(values or [0], options, **kwargs)


def identity():
    """Returns a JavaScript identity function.

    This function creates a JavaScript snippet that represents an identity function,
    which returns its input unchanged.

    Returns:
        dict: A dictionary representing the JavaScript identity function.
    """
    return js("(x) => x")


# The following convenience dicts can be added directly to PlotSpecs to declare additional behaviour.


def grid_y():
    return {"y": {"grid": True}}


def grid_x():
    return {"x": {"grid": True}}


def grid():
    return {"grid": True}


def color_legend():
    return {"color": {"legend": True}}


def clip():
    return {"clip": True}


def title(title):
    return {"title": title}


def subtitle(subtitle):
    return {"subtitle": subtitle}


def caption(caption):
    return {"caption": caption}


def width(width):
    return {"width": width}


def height(height):
    return {"height": height}


def size(size, height=None):
    return {"width": size, "height": height or size}


def aspect_ratio(r):
    return {"aspectRatio": r}


def inset(i):
    return {"inset": i}


def color_scheme(name):
    # See https://observablehq.com/plot/features/scales#color-scales
    return {"color": {"scheme": name}}


def domainX(d):
    return {"x": {"domain": d}}


def domainY(d):
    return {"y": {"domain": d}}


def domain(xd, yd=None):
    return {"x": {"domain": xd}, "y": {"domain": yd or xd}}


def color_map(mappings):
    # these will be merged & so are composable. in plot.js they are
    # converted to a {color: {domain: [...], range: [...]}} object.
    return {"color_map": mappings}


def margin(*args):
    """
    Set margin values for a plot using CSS-style margin shorthand.

    Supported arities:
        margin(all)
        margin(vertical, horizontal)
        margin(top, horizontal, bottom)
        margin(top, right, bottom, left)

    """
    if len(args) == 1:
        return {"margin": args[0]}
    elif len(args) == 2:
        return {
            "marginTop": args[0],
            "marginBottom": args[0],
            "marginLeft": args[1],
            "marginRight": args[1],
        }
    elif len(args) == 3:
        return {
            "marginTop": args[0],
            "marginLeft": args[1],
            "marginRight": args[1],
            "marginBottom": args[2],
        }
    elif len(args) == 4:
        return {
            "marginTop": args[0],
            "marginRight": args[1],
            "marginBottom": args[2],
            "marginLeft": args[3],
        }
    else:
        raise ValueError(f"Invalid number of arguments: {len(args)}")


# barX
# For reference - other options supported by plots
example_plot_options = {
    "title": "TITLE",
    "subtitle": "SUBTITLE",
    "caption": "CAPTION",
    "width": "100px",
    "height": "100px",
    "grid": True,
    "inset": 10,
    "aspectRatio": 1,
    "style": {"font-size": "100px"},  # css string also works
    "clip": True,
}

# dot([[0, 0], [0, 1], [1, 1], [2, 3], [4, 2], [4, 0]])


def doc(fn):
    """
    Decorator to display the docstring of a python function formatted as Markdown.

    Args:
        fn: The function whose docstring to display.

    Returns:
        A JSCall instance
    """

    if fn.__doc__:
        name = fn.__name__
        doc = fn.__doc__.strip()  # Strip leading/trailing whitespace
        # Dedent the docstring to avoid unintended code blocks
        doc = "\n".join(line.strip() for line in doc.split("\n"))
        module = fn.__module__
        module = "Plot" if fn.__module__.endswith("plot_defs") else module
        title = f"<span style='padding-right: 10px;'>{module}.{name}</span>"
        return View.md(
            f"""
<div class="doc-header">{title}</div>
<div class="doc-content">

{doc}

</div>
"""
        )
    else:
        return View.md("No docstring available.")


def state(name: str) -> dict[str, str]:
    return js(f"$state.{name}")


# %%


def Frames(frames, key=None, **opts):
    """
    Create an animated plot that cycles through a list of frames.

    Args:
        frames (list): A list of plot specifications or renderable objects to animate.
        **opts: Additional options for the animation, such as fps (frames per second).

    Returns:
        A Hiccup-style representation of the animated plot.
    """
    if key is None:
        key = f"frame_animation_{random.randint(1, 1000000)}"
        return Hiccup(View.Frames, {"state_key": key, "frames": frames}) | Slider(
            key, [0, len(frames) - 1], **opts
        )
    else:
        return Hiccup(View.Frames, {"state_key": key, "frames": frames})
