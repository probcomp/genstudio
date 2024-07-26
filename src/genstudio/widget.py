import datetime
import json
import uuid

from typing import Any, Iterable, Callable, Dict

import anywidget
import traitlets

from genstudio.util import PARENT_PATH


class Cache:
    def __init__(self):
        self.cache = {}

    def add(self, obj):
        id = str(uuid.uuid4())
        self.cache[id] = obj
        return id

    def get(self, id):
        return self.cache.get(id)

    def for_json(self):
        return {
            id: {"value": obj.data, "static": obj.static}
            for id, obj in self.cache.items()
        }


class Cached:
    def __init__(self, data, static):
        self.data = data
        self.static = static


def cache(data, static=True):
    return Cached(data, static=static)


def to_json(data, widget=None, cache=None):
    def default(obj):
        if isinstance(obj, Cached):
            if cache is not None:
                for cache_id, cache_obj in cache.cache.items():
                    if cache_obj is obj:
                        return {"__type__": "cached", "id": cache_id}

                id = cache.add(obj)
                return {"__type__": "cached", "id": id}
            else:
                return obj.data
        if hasattr(obj, "for_json"):
            return obj.for_json()
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
            return callback_for_json(obj, widget)
        else:
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    return json.dumps(data, default=default)


def callback_for_json(f, widget):
    if widget is not None:
        id = str(uuid.uuid4())
        widget.callback_registry[id] = f
        return {"__type__": "callback", "id": id}


def to_json_with_cache(data, widget=None):
    cache = Cache()
    return to_json({"ast": data, "cache": cache}, widget=widget, cache=cache)


class Widget(anywidget.AnyWidget):
    _esm = PARENT_PATH / "js/widget_build.js"
    _css = PARENT_PATH / "widget.css"
    callback_registry: Dict[str, Callable] = {}
    data = traitlets.Any().tag(sync=True, to_json=to_json_with_cache)

    def __init__(self, ast: Any):
        super().__init__()
        self.data = ast

    def set_ast(self, ast: Any):
        self.data = ast

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
