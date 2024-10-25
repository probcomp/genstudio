import datetime
import orjson
import uuid

from typing import Any, Iterable, Callable, Dict

import anywidget
import traitlets

from genstudio.util import PARENT_PATH, CONFIG


class Cache:
    # a Cache is used to store "refs" off to the side while serializing data.
    def __init__(self):
        self.cache = {}

    def entry(self, id, value, **kwargs):
        if id not in self.cache:
            # perform to_json conversion for cache entries immediately
            self.cache[id] = orjson.Fragment(to_json(value, **kwargs))
        return {"__type__": "ref", "id": id}

    def for_json(self):
        return self.cache


def to_json(data, widget=None, cache=None):
    def default(obj):
        if hasattr(obj, "ref_id"):
            if cache is not None:
                return cache.entry(
                    obj.ref_id(), obj.for_json(), widget=widget, cache=cache
                )
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
            return {"__type__": "datetime", "value": obj.isoformat()}
        elif callable(obj):
            if widget is not None:
                id = str(uuid.uuid4())
                widget.callback_registry[id] = obj
                return {"__type__": "callback", "id": id}
            return None
        else:
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    return orjson.dumps(data, default=default).decode("utf-8")


def to_json_with_cache(data: Any, widget: "Widget | None" = None):
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
            f({**params["event"], "widget": self})
        return "ok", []

    def update_state(self, *updates):
        # an update can be a dict of {id, value} to reset, or
        # a list, [id, operation, payload] where operations are
        # "append", "concat", "reset", or "setAt". The payload for
        # "setAt" should be a list [index, value].
        def entry_id(key):
            return key if isinstance(key, str) else key.id

        out = []
        for entry in updates:
            if isinstance(entry, dict):
                for key, value in entry.items():
                    out.append([entry_id(key), "reset", value])
            else:
                out.append([entry_id(entry[0]), entry[1], entry[2]])
        self.send({"type": "update_state", "updates": to_json(out, widget=self)})
