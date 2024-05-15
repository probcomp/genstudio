import json
import datetime
import jax.numpy as jnp
import numpy as np
import anywidget
import traitlets
from pathlib import Path
import importlib.util

# necessary for VS Code IPython interactive contexts 
PARENT_PATH = Path(importlib.util.find_spec("gen.studio.widget").origin).parent

def to_json(data, _widget):
    def default(obj):
        if hasattr(obj, "to_json"):
            return obj.to_json()
        if isinstance(obj, (jnp.ndarray, np.ndarray)):
            return obj.tolist()
        elif isinstance(obj, (datetime.date, datetime.datetime)):
            return {"pyobsplot-type": "datetime", "value": obj.isoformat()}
        else:
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    return json.dumps(data, default=default)

class Widget(anywidget.AnyWidget):
    _esm = PARENT_PATH / "widget.js"
    data = traitlets.Any().tag(sync=True, to_json=to_json)

    def __init__(self, data):
        super().__init__(data=data)

    @anywidget.experimental.command
    def ping(self, msg, buffers):
        return "pong", None