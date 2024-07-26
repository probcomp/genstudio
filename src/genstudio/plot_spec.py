from genstudio.layout import LayoutItem, View
from typing import TypeAlias, Union, Sequence, Any
from genstudio.util import deep_merge

SpecInput: TypeAlias = Union[
    "PlotSpec", Sequence[Union["PlotSpec", dict[str, Any]]], dict[str, Any]
]

Mark = dict[str, Any]


def _add_list(spec: dict[str, Any], marks: list[Mark], to_add: Sequence[Any]) -> None:
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


def _add_dict(spec: dict[str, Any], marks: list[Mark], to_add: dict[str, Any]) -> None:
    # mutates spec & marks, returns nothing
    if "pyobsplot-type" in to_add:
        marks.append(to_add)
    else:
        deep_merge(spec, to_add)
        new_marks = to_add.get("marks", None)
        if new_marks:
            spec["marks"] = marks
            _add_list(spec, marks, new_marks)


def _add(
    spec: dict[str, Any],
    marks: list[Mark],
    to_add: Any,
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
        super().__init__()
        marks: list[Mark] = []
        self.spec: dict[str, Any] = {"marks": []}
        if specs:
            _add_list(self.spec, marks, specs)
        if kwargs:
            _add_dict(self.spec, marks, kwargs)
        self.spec["marks"] = marks

    def __add__(self, to_add: Any) -> "PlotSpec":
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

    def for_json(self) -> Any:
        return View.PlotSpec(self.spec)


def new(*specs: Any, **kwargs: Any) -> PlotSpec:
    """Create a new PlotSpec from the given specs and options."""
    return PlotSpec(*specs, **kwargs)
