import datetime
import json
import uuid

from typing import Any, Iterable, Callable, Dict

import anywidget
import traitlets

from genstudio.util import PARENT_PATH


class Cached:
    def __init__(self, data):
        self.data = data


def cache(data):
    return Cached(data)


def to_json(data, widget=None):
    def default(obj):
        if isinstance(obj, Cached):
            if widget is not None:
                id = str(uuid.uuid4())
                widget.data_cache[id] = obj.data
                return {"__type__": "cached", "id": id}
            else:
                return obj.data
        if hasattr(obj, "to_json"):
            return obj.to_json()
        if hasattr(obj, "tolist"):
            return obj.tolist()
        if isinstance(obj, Iterable):
            # Check if the iterable might be exhaustible
            if not hasattr(obj, "__len__") and not hasattr(obj, "__getitem__"):
                print(
                    f"Warning: Potentially exhaustible iterator encountered: {type(obj).__name__}"
                )
            return list(obj)
        elif isinstance(obj, (datetime.date, datetime.datetime)):
            return {"pyobsplot-type": "datetime", "value": obj.isoformat()}
        elif callable(obj):
            return callback_to_json(obj, widget)
        else:
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    return json.dumps(data, default=default)


def callback_to_json(f, widget):
    if widget is not None:
        id = str(uuid.uuid4())
        widget.callback_registry[id] = f
        return {"__type__": "callback", "id": id}


class Widget(anywidget.AnyWidget):
    _esm = PARENT_PATH / "js/widget_build.js"
    _css = PARENT_PATH / "widget.css"
    callback_registry: Dict[str, Callable] = {}
    ast = traitlets.Any().tag(sync=True, to_json=to_json)
    data_cache = traitlets.Dict().tag(sync=True)

    def __init__(self, ast: Any):
        super().__init__()
        self.ast = ast
        self.data_cache = {}

    def _repr_mimebundle_(self, **kwargs):  # type: ignore
        return super()._repr_mimebundle_(**kwargs)

    @anywidget.experimental.command  # type: ignore
    def handle_callback(
        self, params: dict[str, Any], buffers: list[bytes]
    ) -> tuple[str, list[bytes]]:
        f = self.callback_registry[params["id"]]
        if f is not None:
            f(params["event"])
        return "ok", []
