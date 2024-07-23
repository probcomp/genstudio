import datetime
import json
import uuid

from typing import Any, Iterable

import anywidget
import traitlets


from genstudio.util import PARENT_PATH


def to_json(data, widget=None):
    def default(obj):
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
        widget.callbacks[id] = f
        return {"type": "callback", "id": id}


class Widget(anywidget.AnyWidget):
    _esm = PARENT_PATH / "js/widget_build.js"
    _css = PARENT_PATH / "widget.css"
    callbacks = {}
    data = traitlets.Any().tag(sync=True, to_json=to_json)

    def __init__(self, data):
        super().__init__()
        self.data = data

    def _repr_mimebundle_(self, **kwargs):  # type: ignore
        return super()._repr_mimebundle_(**kwargs)

    @anywidget.experimental.command  # type: ignore
    def callback(
        self, params: dict[str, Any], buffers: list[bytes]
    ) -> tuple[str, list[bytes]]:
        f = self.callbacks[params["id"]]
        if f is not None:
            f(params)
        return f"Callback {id} processed", []
