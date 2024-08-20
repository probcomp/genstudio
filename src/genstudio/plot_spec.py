from genstudio.layout import LayoutItem, View
from genstudio.js_modules import JSCall
from genstudio.util import CONFIG
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

    def cache_id(self):
        return self.id

    def for_json(self) -> Any:
        return self.ast


def flatten_layers(layers: Sequence[Any]) -> list[Any]:
    """
    Merge layers into a flat structure, including PlotSpec instances.
    """
    flattened = []
    for layer in layers:
        if isinstance(layer, (list, tuple)):
            flattened.extend(flatten_layers(layer))
        elif isinstance(layer, PlotSpec):
            flattened.extend(layer.layers)
        else:
            flattened.append(layer)
    return flattened


class PlotSpec(LayoutItem):
    """
    Represents a specification for a plot (in Observable Plot).

    PlotSpec can be composed using the + operator. When combined, layers accumulate.
    Lists of marks or dicts of plot options can also be added directly to a PlotSpec.

    Args:
        *specs: PlotSpec, lists of marks, or dicts of plot options to initialize with.
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

    def __radd__(self, *to_add: Any) -> "PlotSpec":
        new_spec = PlotSpec()
        new_spec.layers = flatten_layers(to_add) + self.layers
        return new_spec

    def for_json(self) -> Any:
        return View.PlotSpec({"layers": [CONFIG["defaults"]] + self.layers})


def new(*specs: Any, **kwargs: Any) -> PlotSpec:
    """Create a new PlotSpec from the given specs and options."""
    return PlotSpec(*specs, **kwargs)
