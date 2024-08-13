from genstudio.layout import LayoutItem, View
from genstudio.js_modules import JSCall
from typing import TypeAlias, Union, Sequence, Any
import uuid

SpecInput: TypeAlias = Union[
    "PlotSpec",
    "MarkSpec",
    Sequence[Union["PlotSpec", "MarkSpec", dict[str, Any]]],
    dict[str, Any],
]

Mark = dict[str, Any]


class MarkSpec:
    def __init__(self, name, data, options):
        self.id = str(uuid.uuid4())
        self.ast = JSCall("View", "MarkSpec", [name, data, options])

    def for_json(self, cache=None, **kwargs) -> Any:
        if cache is None:
            return self.ast
        cache.add(self.id, self.ast, cache=cache, **kwargs)
        return cache.entry(self.id)


def flatten_layers(layers: Sequence[Any]) -> list[Any]:
    """
    Merge layers into a flat structure.
    """
    return [
        item
        for layer in layers
        for item in (
            flatten_layers(layer) if isinstance(layer, (list, tuple)) else [layer]
        )
    ]


class PlotSpec(LayoutItem):
    """
    Represents a specification for a plot (in Observable Plot).

    PlotSpecs can be composed using the + operator. When combined, layers accumulate.
    Lists of marks or dicts of plot options can also be added directly to a PlotSpec.

    Args:
        *specs: PlotSpecs, lists of marks, or dicts of plot options to initialize with.
        **kwargs: Additional plot options passed as keyword arguments.
    """

    def __init__(self, *specs: SpecInput, **kwargs: Any) -> None:
        super().__init__()
        self.layers: list[dict[str, Any]] = flatten_layers(specs)
        if kwargs:
            self.layers.append(kwargs)

    def __add__(self, *to_add: Any) -> "PlotSpec":
        """
        Combine this PlotSpec with another PlotSpec, list of marks, or dict of options.

        Args:
            to_add: The PlotSpec, list of marks, or dict of options to add.

        Returns:
            A new PlotSpec with the combined layers.
        """
        new_spec = PlotSpec()
        new_spec.layers = self.layers + flatten_layers(to_add)
        return new_spec

    def __radd__(self, to_add: Any) -> "PlotSpec":
        new_spec = PlotSpec()
        new_spec.layers = flatten_layers(to_add) + self.layers
        return new_spec

    def for_json(self, cache=None, widget=None) -> Any:
        return View.PlotSpec({"layers": self.layers})


def new(*specs: Any, **kwargs: Any) -> PlotSpec:
    """Create a new PlotSpec from the given specs and options."""
    return PlotSpec(*specs, **kwargs)
