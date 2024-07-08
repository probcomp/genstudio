# %%
import base64
import json
import datetime
import anywidget
import traitlets
from genstudio.util import PARENT_PATH
from typing import Iterable
from IPython.display import display


def to_json(data, _widget):
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
        else:
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    return json.dumps(data, default=default)


class Widget(anywidget.AnyWidget):
    _esm = PARENT_PATH / "js/widget_build.js"
    _css = PARENT_PATH / "widget.css"
    data = traitlets.Any().tag(sync=True, to_json=to_json)
    image_requests = traitlets.List(traitlets.Unicode()).tag(sync=True)
    images = traitlets.Dict().tag(sync=True)
    _displayed = False

    def ensure_displayed(self):
        """Ensure the widget is displayed in the output."""
        if not self._displayed:
            display(self)
            self._displayed = True

    def __init__(self, data):
        super().__init__()
        self.data = data
        self.observe(self._on_images_change, names="images")
        self._pending_saves = {}

    def save_image(self, path, callback=None):
        self.ensure_displayed()
        format = path.split(".")[-1].lower()
        if format not in ["png", "jpeg", "svg"]:
            raise ValueError(f"Unsupported format: {format}")

        if format not in self.image_requests:
            self.image_requests = self.image_requests + [format]
        self._pending_saves[format] = (path, callback)

    def _on_images_change(self, change):
        for format, (path, callback) in list(self._pending_saves.items()):
            if format in change.new:
                self._save_image(format, path, callback)
                del self._pending_saves[format]

    def _save_image(self, format, path, callback):
        with open(path, "wb") as f:
            f.write(base64.b64decode(self.images[format].split(",")[1]))
        if callback:
            callback(path)

    def _repr_mimebundle_(self, **kwargs):
        self._displayed = True
        return super()._repr_mimebundle_(**kwargs)
