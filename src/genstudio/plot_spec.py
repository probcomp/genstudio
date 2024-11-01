import uuid
from typing import Any, Sequence, TypeAlias, Union

from genstudio.layout import JSRef, JSCall, LayoutItem
from genstudio.util import CONFIG

SpecInput: TypeAlias = Union[
    "PlotSpec",
    "MarkSpec",
    Sequence[Union["PlotSpec", "MarkSpec", JSCall, dict[Any, Any]]],
    JSCall,
    dict[Any, Any],
]


class MarkSpec:
    def __init__(self, name, data, options):
        self._state_key = str(uuid.uuid4())
        self.ast = JSCall("MarkSpec", [name, data, options])

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


_PlotSpec = JSRef("PlotSpec")


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
        self.layers: list[JSCall | dict[Any, Any]] = flatten_layers(specs)
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
        return _PlotSpec({"layers": [CONFIG["defaults"]] + self.layers})


def new(*specs: Any, **kwargs: Any) -> PlotSpec:
    """Create a new PlotSpec from the given specs and options."""
    return PlotSpec(*specs, **kwargs)
