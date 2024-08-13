import datetime
import orjson
import uuid

from typing import Any, Iterable, Callable, Dict

import anywidget
import traitlets

from genstudio.util import PARENT_PATH, CONFIG


class Cache:
    def __init__(self):
        self.cache = {}

    def has(self, id):
        return id in self.cache

    def add(self, id, value, static=False, **kwargs):
        # immediately serialize value so that nested cached values are
        # discovered during the initial data traversal
        self.cache[id] = (orjson.Fragment(to_json(value, **kwargs)), static)

    def entry(self, id):
        return {"__type__": "cached", "id": id}

    def for_json(self, widget=None, cache=None):
        return {
            id: {"value": value, "static": static}
            for id, (value, static) in self.cache.items()
        }


def to_json(data, widget=None, cache=None):
    def default(obj):
        if hasattr(obj, "for_json"):
            return obj.for_json(cache=cache, widget=widget)
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
            if widget is not None:
                id = str(uuid.uuid4())
                widget.callback_registry[id] = obj
                return {"__type__": "callback", "id": id}
            return None
        else:
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    return orjson.dumps(data, default=default).decode("utf-8")


def to_json_with_cache(data, widget=None):
    cache = Cache()
    return to_json({"ast": data, "cache": cache, **CONFIG}, widget=widget, cache=cache)


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
