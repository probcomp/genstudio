import copy
from genstudio.widget import Widget
from genstudio.js_modules import JSRef
from typing import Any, Dict, List, Sequence, Optional, Union


SpecInput = Union[
    "PlotSpec", Sequence[Union["PlotSpec", Dict[str, Any]]], Dict[str, Any]
]
Mark = Dict[str, Any]

View = JSRef("View")


class LayoutItem:
    def to_json(self) -> Any:
        return NotImplemented

    def __and__(self, other):
        return Row(self, other)

    def __rand__(self, other):
        return Row(other, self)

    def __or__(self, other):
        return Column(self, other)

    def __ror__(self, other):
        return Column(other, self)

    def _repr_mimebundle_(self, **kwargs):
        return Widget(self.to_json())._repr_mimebundle_(**kwargs)


class Hiccup(LayoutItem, list):
    """Wraps a Hiccup-style list to be rendered as an interactive widget in the JavaScript runtime."""

    def __init__(self, *args):
        if len(args) == 0:
            super().__init__()
        elif len(args) == 1 and isinstance(args[0], (list, tuple)):
            super().__init__(args[0])
        else:
            super().__init__(args)

    def _repr_mimebundle_(self, **kwargs: Any):
        """Renders the Hiccup list as an interactive widget in the JavaScript runtime."""
        return Widget(self)._repr_mimebundle_(**kwargs)

    def to_json(self):
        return self


def flatten_layout_items(items, layout_class):
    flattened = []
    options = {}
    for item in items:
        if isinstance(item, layout_class):
            flattened.extend(item.items)
            options.update(item.options)
        elif isinstance(item, dict):
            options.update(item)
        else:
            flattened.append(item)
    return flattened, options


class Row(LayoutItem):
    def __init__(self, *items):
        self.items, self.options = flatten_layout_items(items, Row)

    def to_json(self) -> Hiccup:
        return Hiccup(View.Row, self.options, *self.items)


class Column(LayoutItem):
    def __init__(self, *items):
        self.items, self.options = flatten_layout_items(items, Column)

    def to_json(self) -> Hiccup:
        return Hiccup(View.Column, self.options, *self.items)


class Slider(LayoutItem):
    def __init__(self, key, range, label=None, **kwargs):
        self.config = {
            "state_key": key,
            "range": [0, range] if isinstance(range, int) else range,
            "label": label,
            "kind": "Slider",
            **kwargs,
        }

    def to_json(self):
        return View.Reactive(self.config)


def _deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries. Mutates dict1.
    Values in dict2 overwrite values in dict1. If both values are dictionaries, recursively merge them.
    """

    for k, v in dict2.items():
        if k in dict1 and isinstance(dict1[k], dict) and isinstance(v, dict):
            dict1[k] = _deep_merge(dict1[k], v)
        elif isinstance(v, dict):
            dict1[k] = copy.deepcopy(v)
        else:
            dict1[k] = v
    return dict1


def _add_list(
    spec: Dict[str, Any], marks: List[Mark], to_add: Sequence[SpecInput]
) -> None:
    # mutates spec & marks, returns nothing
    for new_spec in to_add:
        if isinstance(new_spec, dict):
            _add_dict(spec, marks, new_spec)
        elif isinstance(new_spec, PlotSpec):
            _add_dict(spec, marks, new_spec.spec)
        elif isinstance(new_spec, (list, tuple)):
            _add_list(spec, marks, new_spec)
        else:
            raise ValueError(f"Invalid plot specification: {new_spec}")


def _add_dict(spec: Dict[str, Any], marks: List[Mark], to_add: Dict[str, Any]) -> None:
    # mutates spec & marks, returns nothing
    if "pyobsplot-type" in to_add:
        marks.append(to_add)
    else:
        _deep_merge(spec, to_add)
        new_marks = to_add.get("marks", None)
        if new_marks:
            spec["marks"] = marks
            _add_list(spec, marks, new_marks)


def _add(
    spec: Dict[str, Any],
    marks: List[Mark],
    to_add: Union[SpecInput, Sequence[SpecInput]],
) -> None:
    # mutates spec & marks, returns nothing
    if isinstance(to_add, (list, tuple)):
        _add_list(spec, marks, to_add)
    elif isinstance(to_add, dict):
        _add_dict(spec, marks, to_add)
    elif isinstance(to_add, PlotSpec):
        _add_dict(spec, marks, to_add.spec)
    else:
        raise TypeError(
            f"Unsupported operand type(s) for +: 'PlotSpec' and '{type(to_add).__name__}'"
        )


class PlotSpec(LayoutItem):
    """
    Represents a specification for an plot (in Observable Plot).

    PlotSpecs can be composed using the + operator. When combined, marks accumulate
    and plot options are merged. Lists of marks or dicts of plot options can also be
    added directly to a PlotSpec.

    IPython plot widgets are created lazily when the spec is viewed in a notebook,
    and then cached for efficiency.

    Args:
        *specs: PlotSpecs, lists of marks, or dicts of plot options to initialize with.
        **kwargs: Additional plot options passed as keyword arguments.
    """

    def __init__(self, *specs: SpecInput, **kwargs: Any) -> None:
        marks: List[Mark] = []
        self.spec: Dict[str, Any] = {"marks": []}
        if specs:
            _add_list(self.spec, marks, specs)
        if kwargs:
            _add_dict(self.spec, marks, kwargs)
        self.spec["marks"] = marks
        self._plot: Optional[Widget] = None

    def __add__(self, to_add: SpecInput) -> "PlotSpec":
        """
        Combine this PlotSpec with another PlotSpec, list of marks, or dict of options.

        Args:
            to_add: The PlotSpec, list of marks, or dict of options to add.

        Returns:
            A new PlotSpec with the combined marks and options.
        """
        spec = self.spec.copy()
        marks = spec["marks"].copy()
        _add(spec, marks, to_add)
        spec["marks"] = marks
        return PlotSpec(spec)

    def plot(self) -> Widget:
        """
        Lazily generate & cache the widget for this PlotSpec.
        """
        if self._plot is None:
            self._plot = Widget(View.PlotSpec(self.spec))
        return self._plot

    def reset(self, *specs: SpecInput, **kwargs: Any) -> None:
        """
        Reset this PlotSpec's options and marks to those from the given specs.

        Reuses the existing plot widget.

        Args:
            *specs: PlotSpecs, lists of marks, or dicts of plot options to reset to.
            **kwargs: Additional options to reset.
        """
        self.spec = PlotSpec(*specs, **kwargs).spec
        self.plot().data = self.spec

    def update(
        self, *to_add: SpecInput, marks: Optional[List[Mark]] = None, **kwargs: Any
    ) -> None:
        """
        Update this PlotSpec's options and marks in-place.

        Reuses the existing plot widget.

        Args:
            *specs: PlotSpecs, lists of marks, or dicts of plot options to update with.
            marks (list, optional): List of marks to replace existing marks with.
                If provided, overwrites rather than adds to existing marks.
            **kwargs: Additional options to update.
        """
        if to_add:
            _add(self.spec, self.spec["marks"], to_add)
        if marks is not None:
            self.spec["marks"] = marks
        self.spec.update(kwargs)
        self.plot().data = self.spec

    def _repr_mimebundle_(self, **kwargs):
        return self.plot()._repr_mimebundle_(**kwargs)

    def to_json(self):
        return View.PlotSpec(self.spec)


def new(*specs, **kwargs):
    """Create a new PlotSpec from the given specs and options."""
    return PlotSpec(*specs, **kwargs)
