import copy
import uuid
from typing import Any, Sequence, TypeAlias, Union

from html2image import Html2Image
from PIL import Image

from genstudio.js_modules import JSRef
from genstudio.util import PARENT_PATH
from genstudio.widget import Widget, to_json

SpecInput: TypeAlias = Union[
    "PlotSpec", Sequence[Union["PlotSpec", dict[str, Any]]], dict[str, Any]
]

Mark = dict[str, Any]

View = JSRef("View")


def html_snippet(data, id=None):
    id = id or f"genstudio-widget-{uuid.uuid4().hex}"
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


def html_standalone(data, id=None):
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


class HTML:
    def __init__(self, data):
        self.data = data
        self.id = f"genstudio-widget-{uuid.uuid4().hex}"

    def _repr_mimebundle_(self, **kwargs):
        html_content = html_snippet(self.data, self.id)
        return {"text/html": html_content}, {}


class LayoutItem:
    def __init__(self):
        self._html: HTML | None = None
        self._widget: Widget | None = None
        self._display_preference = "html"

    def display_as(self, display_preference) -> "LayoutItem":
        if display_preference not in ["html", "widget"]:
            raise ValueError("display_pref must be either 'html' or 'widget'")
        self._display_preference = display_preference
        return self

    def to_json(self) -> dict[str, Any]:
        raise NotImplementedError("Subclasses must implement to_json method")

    def __and__(self, other: "LayoutItem") -> "Row":
        return Row(self, other)

    def __rand__(self, other: "LayoutItem") -> "Row":
        return Row(other, self)

    def __or__(self, other: "LayoutItem") -> "Column":
        return Column(self, other)

    def __ror__(self, other: "LayoutItem") -> "Column":
        return Column(other, self)

    def _repr_mimebundle_(self, **kwargs: Any) -> Any:
        if self._display_preference == "widget":
            return self.widget()._repr_mimebundle_(**kwargs)
        else:
            return self.html()._repr_mimebundle_(**kwargs)

    def html(self) -> HTML:
        """
        Lazily generate & cache the HTML for this LayoutItem.
        """
        if self._html is None:
            self._html = HTML(self.to_json())
        return self._html

    def widget(self) -> Widget:
        """
        Lazily generate & cache the widget for this LayoutItem.
        """
        if self._widget is None:
            self._widget = Widget(self.to_json())
        return self._widget

    def save_html(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(html_standalone(self.to_json()))
        print(f"HTML saved to {path}")

    def save_image(self, path, width=500, height=1000):
        # Save image using headless browser
        hti = Html2Image()
        hti.size = (width, height)
        hti.screenshot(html_str=html_standalone(self.to_json()), save_as=path)
        # Crop transparent regions
        img = Image.open(path)
        content = img.getbbox()
        img = img.crop(content)
        img.save(path)
        print(f"Image saved to {path}")

    def reset(self, other: "LayoutItem") -> None:
        """
        Render a new LayoutItem to this LayoutItem's widget.

        Args:
            new_item: A LayoutItem to reset to.
        """
        new_data = other.to_json()
        self.widget().data = new_data
        self.html().data = new_data


class Hiccup(LayoutItem):
    """Wraps a Hiccup-style list to be rendered as an interactive widget in the JavaScript runtime."""

    def __init__(self, *args: Any) -> None:
        LayoutItem.__init__(self)
        self.callbacks = {}
        if len(args) == 0:
            self.data: list[Any] | tuple[Any, ...] | None = None
        elif len(args) == 1 and isinstance(args[0], (list, tuple)):
            self.data = args[0]
        else:
            self.data = args

    def _process_callbacks(self, data):
        def process_item(item):
            if isinstance(item, dict) and "onClick" in item:
                callback_id = str(uuid.uuid4())
                self.callbacks[callback_id] = item["onClick"]
                return {**item, "onClick": {"type": "callback", "id": callback_id}}
            elif isinstance(item, (list, tuple)):
                return [process_item(subitem) for subitem in item]
            return item

        return process_item(data)

    def to_json(self) -> Any:
        return self._process_callbacks(self.data)

    def handle_callback(self, callback_id, *args):
        if callback_id in self.callbacks:
            self.callbacks[callback_id](*args)


def flatten_layout_items(
    items: Sequence[Any], layout_class: type
) -> tuple[list[Any], dict[str, Any]]:
    flattened: list[Any] = []
    options: dict[str, Any] = {}
    for item in items:
        if isinstance(item, layout_class):
            flattened.extend(item.items)
            options.update(item.options)
        elif isinstance(item, dict):
            options.update(item)
        else:
            flattened.append(item)
    return flattened, options


class Row(LayoutItem):
    def __init__(self, *items: Any):
        super().__init__()
        self.items, self.options = flatten_layout_items(items, Row)

    def to_json(self) -> Any:
        return Hiccup(View.Row, self.options, *self.items)


class Column(LayoutItem):
    def __init__(self, *items: Any):
        super().__init__()
        self.items, self.options = flatten_layout_items(items, Column)

    def to_json(self) -> Any:
        return Hiccup(View.Column, self.options, *self.items)


class Slider(LayoutItem):
    def __init__(
        self,
        key: str,
        range: int | Sequence[int],
        label: str | None = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.config: dict[str, Any] = {
            "state_key": key,
            "range": [0, range] if isinstance(range, int) else range,
            "label": label,
            "kind": "Slider",
            **kwargs,
        }

    def to_json(self) -> Any:
        return View.Reactive(self.config)


def _deep_merge(dict1: dict[str, Any], dict2: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively merge two dictionaries. Mutates dict1.
    Values in dict2 overwrite values in dict1. If both values are dictionaries, recursively merge them.
    """

    for k, v in dict2.items():
        if k in dict1 and isinstance(dict1[k], dict) and isinstance(v, dict):
            dict1[k] = _deep_merge(dict1[k], v)
        elif isinstance(v, dict):
            dict1[k] = copy.deepcopy(v)
        else:
            dict1[k] = v
    return dict1


def _add_list(spec: dict[str, Any], marks: list[Mark], to_add: Sequence[Any]) -> None:
    # mutates spec & marks, returns nothing
    for new_spec in to_add:
        if isinstance(new_spec, dict):
            _add_dict(spec, marks, new_spec)
        elif isinstance(new_spec, PlotSpec):
            _add_dict(spec, marks, new_spec.spec)
        elif isinstance(new_spec, (list, tuple)):
            _add_list(spec, marks, new_spec)
        else:
            raise ValueError(f"Invalid plot specification: {new_spec}")


def _add_dict(spec: dict[str, Any], marks: list[Mark], to_add: dict[str, Any]) -> None:
    # mutates spec & marks, returns nothing
    if "pyobsplot-type" in to_add:
        marks.append(to_add)
    else:
        _deep_merge(spec, to_add)
        new_marks = to_add.get("marks", None)
        if new_marks:
            spec["marks"] = marks
            _add_list(spec, marks, new_marks)


def _add(
    spec: dict[str, Any],
    marks: list[Mark],
    to_add: Any,
) -> None:
    # mutates spec & marks, returns nothing
    if isinstance(to_add, (list, tuple)):
        _add_list(spec, marks, to_add)
    elif isinstance(to_add, dict):
        _add_dict(spec, marks, to_add)
    elif isinstance(to_add, PlotSpec):
        _add_dict(spec, marks, to_add.spec)
    else:
        raise TypeError(
            f"Unsupported operand type(s) for +: 'PlotSpec' and '{type(to_add).__name__}'"
        )


class PlotSpec(LayoutItem):
    """
    Represents a specification for an plot (in Observable Plot).

    PlotSpecs can be composed using the + operator. When combined, marks accumulate
    and plot options are merged. Lists of marks or dicts of plot options can also be
    added directly to a PlotSpec.

    IPython plot widgets are created lazily when the spec is viewed in a notebook,
    and then cached for efficiency.

    Args:
        *specs: PlotSpecs, lists of marks, or dicts of plot options to initialize with.
        **kwargs: Additional plot options passed as keyword arguments.
    """

    def __init__(self, *specs: SpecInput, **kwargs: Any) -> None:
        super().__init__()
        marks: list[Mark] = []
        self.spec: dict[str, Any] = {"marks": []}
        if specs:
            _add_list(self.spec, marks, specs)
        if kwargs:
            _add_dict(self.spec, marks, kwargs)
        self.spec["marks"] = marks

    def __add__(self, to_add: Any) -> "PlotSpec":
        """
        Combine this PlotSpec with another PlotSpec, list of marks, or dict of options.

        Args:
            to_add: The PlotSpec, list of marks, or dict of options to add.

        Returns:
            A new PlotSpec with the combined marks and options.
        """
        spec = self.spec.copy()
        marks = spec["marks"].copy()
        _add(spec, marks, to_add)
        spec["marks"] = marks
        return PlotSpec(spec)

    def to_json(self) -> Any:
        return View.PlotSpec(self.spec)


def new(*specs: Any, **kwargs: Any) -> PlotSpec:
    """Create a new PlotSpec from the given specs and options."""
    return PlotSpec(*specs, **kwargs)
