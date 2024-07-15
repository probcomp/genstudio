import datetime
import json
import uuid
from typing import Iterable

import anywidget
import traitlets

from genstudio.util import PARENT_PATH


def to_json(data):
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


def html_snippet(data, id):
    serialized_data = to_json(data)

    # Read and inline the JS and CSS files
    with open(PARENT_PATH / "js/widget_build.js", "r") as js_file:
        js_content = js_file.read()
    with open(PARENT_PATH / "widget.css", "r") as css_file:
        css_content = css_file.read()

    html_content = f"""
    <style>{css_content}</style>
    <div id="{id}"></div>
    <script type="module">
        {js_content}
        const container = document.getElementById('{id}');
        const data = {serialized_data};
        renderData(container, data);
    </script>
    """

    return html_content


def html_standalone(data, id):
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>GenStudio Widget</title>
    </head>
    <body>
        {html_snippet(data, id)}
    </body>
    </html>
    """


def save_html(data, id, path):
    html = html_standalone(data, id)
    with open(path, "w") as f:
        f.write(html)


class HTML:
    def __init__(self, data):
        self.data = data
        self.id = f"genstudio-widget-{uuid.uuid4().hex}"

    def save(self, path):
        save_html(self.data, self.id, path)

    def _repr_mimebundle_(self, **kwargs):
        html_content = html_snippet(self.data, self.id)
        return {"text/html": html_content}, {}


class Widget(anywidget.AnyWidget):
    _esm = PARENT_PATH / "js/widget_build.js"
    _css = PARENT_PATH / "widget.css"
    data = traitlets.Any().tag(sync=True, to_json=lambda x, _: to_json(x))

    def __init__(self, data):
        super().__init__()
        self.data = data

    def _repr_mimebundle_(self, **kwargs):  # type: ignore
        return super()._repr_mimebundle_(**kwargs)

    def html(self):
        """Return an HTML representation of the widget."""
        return HTML(self.data)
