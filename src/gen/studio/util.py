# %%
import datetime
import importlib.util
import json
import pathlib
import re
from functools import partial
from timeit import default_timer as timer

import anywidget
import jax.numpy as jnp
import numpy as np
import traitlets


class benchmark(object):
    """
    A context manager for simple benchmarking.

    Usage:
        with benchmark("My benchmark"):
            # Code to be benchmarked
            ...

    Args:
        msg (str): The message to display with the benchmark result.
        fmt (str, optional): The format string for the time display. Defaults to "%0.3g".

    http://dabeaz.blogspot.com/2010/02/context-manager-for-timing-benchmarks.html
    """

    def __init__(self, msg, fmt="%0.3g"):
        self.msg = msg
        self.fmt = fmt

    def __enter__(self):
        self.start = timer()
        return self

    def __exit__(self, *args):
        t = timer() - self.start
        print(("%s : " + self.fmt + " seconds") % (self.msg, t))
        self.time = t

PARENT_PATH = pathlib.Path(importlib.util.find_spec("gen.studio.util").origin).parent
OBSERVABLE_PLOT_METADATA = json.load(open(PARENT_PATH / "scripts" / "observable_plot_metadata.json"))


# %%

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

def doc_str(functionName):
    return OBSERVABLE_PLOT_METADATA[functionName]['doc']
    
def doc(plot_fn):
    """
    Decorator to display the docstring of a plot function nicely formatted as Markdown.

    Args:
        plot_fn: The plot function whose docstring to display.

    Returns:
        A JSCall instance
    """

    if plot_fn.__doc__:
        name = plot_fn.__name__
        doc = plot_fn.__doc__
        meta = OBSERVABLE_PLOT_METADATA.get(name, None)
        title = (
            f"<span style='font-size: 20px; padding-right: 10px;'>Plot.{name}</span>"
        )
        url = (
            f"https://observablehq.com/plot/{meta['kind']}/{re.search(r'([a-z]+)', name).group(1)}"
            if meta
            else None
        )
        return js_call("View", "md", 
            f"""
<div style="display: block; gap: 10px; border-bottom: 1px solid #ddd; padding: 10px 0;">
{title} 
<a style='color: #777; text-decoration: none;' href="{url}">Examples &#8599;</a></div>

"""
            + doc
        )
    else:
        return js_call("View", "md", "No docstring available.")
    
class JSCall(dict):
    """Represents a JavaScript function call."""
    def __init__(self, module, name, args):
        super().__init__(
            {"pyobsplot-type": "function", "module": module, "name": name, "args": args}
        )
        
    def doc(self):
        return doc(self)
    def _repr_mimebundle_(self, **kwargs):
        return Widget(self)._repr_mimebundle_(**kwargs)

def js_call(module, name, *args):
    """Represents a JavaScript function call."""
    return JSCall(module, name, args)


def js_ref(module, name):
    """Represents a reference to a JavaScript module or name."""
    return JSRef(module=module, name=name)


def js(txt: str) -> dict:
    """Represents raw JavaScript code to be evaluated."""
    return {"pyobsplot-type": "js", "value": txt}

class JSRef(dict):
    """Refers to a JavaScript module or name. When called, returns a function call representation."""
    def __init__(self, module, name=None, inner=lambda fn, *args: fn(*args), doc=None):
        self.__name__ = name
        self.__doc__ = doc
        self.inner = inner
        super().__init__({"pyobsplot-type": "ref", "module": module, "name": name})
    def doc(self):
        return doc(self)

    def __call__(self, *args, **kwargs):
        """Invokes the wrapped JavaScript function in the runtime with the provided arguments."""
        return self.inner(
            partial(js_call, self["module"], self["name"]), *args, **kwargs
        )

    def __getattr__(self, name):
        """Returns a reference to a nested property or method of the JavaScript object."""
        if name[0] == '_':
            return super().__getattribute__(name)
        elif self["name"] is None:
            return JSRef(self["module"], name)
        else:
            raise ValueError("Only module.name paths are currently supported")
            # return JSRef(f"{self['module']}.{self['name']}", name)

class Hiccup(list):
    """Wraps a Hiccup-style list to be rendered as an interactive widget in the JavaScript runtime."""
    def __init__(self, contents):
        super().__init__(contents)

    def _repr_mimebundle_(self, **kwargs):
        """Renders the Hiccup list as an interactive widget in the JavaScript runtime."""
        return Widget(self)._repr_mimebundle_(**kwargs)


def hiccup(x):
    """Constructs a Hiccup object from the provided list to be rendered in the JavaScript runtime."""
    return Hiccup(x)


# %%
