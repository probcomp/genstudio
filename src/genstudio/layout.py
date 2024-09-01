import uuid
from typing import Any, List, Optional, Tuple, Sequence, Union

from html2image import Html2Image
from PIL import Image

from genstudio.util import PARENT_PATH, CONFIG
from genstudio.widget import Widget, to_json_with_cache


def html_snippet(ast, id=None):
    id = id or f"genstudio-widget-{uuid.uuid4().hex}"
    data = to_json_with_cache(ast)

    # Read and inline the JS and CSS files
    with open(PARENT_PATH / "js/widget_build.js", "r") as js_file:
        js_content = js_file.read()
    with open(PARENT_PATH / "widget.css", "r") as css_file:
        css_content = css_file.read()

    html_content = f"""
    <style>{css_content}</style>
    <div class="bg-white p3" id="{id}"></div>

    <script type="application/json">
        {data}
    </script>

    <script type="module">
        {js_content}
        const container = document.getElementById('{id}');
        const jsonString = container.nextElementSibling.textContent;
        renderData(container, {{jsonString}});
    </script>
    """

    return html_content


def html_standalone(ast, id=None):
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>GenStudio Widget</title>
    </head>
    <body>
        {html_snippet(ast, id)}
    </body>
    </html>
    """


class HTML:
    def __init__(self, ast):
        self.ast = ast
        self.id = f"genstudio-widget-{uuid.uuid4().hex}"

    def set_ast(self, ast):
        self.ast = ast

    def _repr_mimebundle_(self, **kwargs):
        html_content = html_snippet(self.ast, self.id)
        return {"text/html": html_content}, {}


class LayoutItem:
    def __init__(self):
        self._html: HTML | None = None
        self._widget: Widget | None = None
        self._display_as = None

    def display_as(self, display_as) -> "LayoutItem":
        if display_as not in ["html", "widget"]:
            raise ValueError("display_pref must be either 'html' or 'widget'")
        self._display_as = display_as
        return self

    def for_json(self) -> dict[str, Any]:
        raise NotImplementedError("Subclasses must implement for_json method")

    def __and__(self, other: Any) -> "Row":
        return Row(self, other)

    def __rand__(self, other: Any) -> "Row":
        return Row(other, self)

    def __or__(self, other: Any) -> "Column":
        return Column(self, other)

    def __ror__(self, other: Any) -> "Column":
        return Column(other, self)

    def _repr_mimebundle_(self, **kwargs: Any) -> Any:
        display_as = self._display_as or CONFIG["display_as"]
        if display_as == "widget":
            return self.widget()._repr_mimebundle_(**kwargs)
        else:
            return self.html()._repr_mimebundle_(**kwargs)

    def html(self) -> HTML:
        """
        Lazily generate & cache the HTML for this LayoutItem.
        """
        if self._html is None:
            self._html = HTML(self.for_json())
        return self._html

    def widget(self) -> Widget:
        """
        Lazily generate & cache the widget for this LayoutItem.
        """
        if self._widget is None:
            self._widget = Widget(self)
        return self._widget

    def save_html(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(html_standalone(self.for_json()))
        print(f"HTML saved to {path}")

    def save_image(self, path, width=500, height=1000):
        # Save image using headless browser
        hti = Html2Image()
        hti.size = (width, height)
        hti.screenshot(html_str=html_standalone(self.for_json()), save_as=path)
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
        display_as = self._display_as or CONFIG["display_as"]
        if display_as != "widget":
            print(
                "Warning: Resetting a non-widget LayoutItem. This will not update the display."
            )
        new_ast = other.for_json()
        self.widget().set_ast(new_ast)
        self.html().set_ast(new_ast)


class JSCall(LayoutItem):
    """Represents a JavaScript function call."""

    def __init__(self, module: str, name: str, args: Union[List[Any], Tuple[Any, ...]]):
        super().__init__()
        self.module = module
        self.name = name
        self.args = args

    def for_json(self) -> dict:
        return {
            "__type__": "function",
            "module": self.module,
            "name": self.name,
            "args": self.args,
        }


class JSRef(LayoutItem):
    """Refers to a JavaScript module or name. When called, returns a function call representation."""

    def __init__(
        self,
        module: str,
        name: Optional[str] = None,
        label: Optional[str] = None,
        doc: Optional[str] = None,
    ):
        super().__init__()
        self.__name__ = name or label
        self.__doc__ = doc
        self.module = module
        self.name = name

    def __call__(self, *args: Any) -> Any:
        """Invokes the wrapped JavaScript function in the runtime with the provided arguments."""
        if self.name is None:
            raise ValueError("Cannot call a JSRef with no name")
        return JSCall(self.module, self.name, args)

    def __getattr__(self, name: str) -> "JSRef":
        """Returns a reference to a nested property or method of the JavaScript object."""
        if name == "cache_id":
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute 'cache_id'"
            )
        if self.name is None:
            return JSRef(self.module, name)
        else:
            raise ValueError("Only module.name paths are currently supported")
            # return JSRef(f"{self.module}.{self.name}", name)

    def for_json(self) -> dict:
        return {"__type__": "ref", "module": self.module, "name": self.name}


def js_ref(module: str, name: str) -> "JSRef":
    """Represents a reference to a JavaScript module or name."""
    return JSRef(module=module, name=name)


class JSCode(LayoutItem):
    def __init__(self, code: str):
        super().__init__()
        self.code = code

    def for_json(self) -> dict:
        return {"__type__": "js", "value": self.code}


def js(txt: str) -> JSCode:
    """Represents raw JavaScript code to be evaluated as a LayoutItem."""
    return JSCode(txt)


class Hiccup(LayoutItem):
    """Wraps a Hiccup-style list to be rendered as an interactive widget in the JavaScript runtime."""

    def __init__(self, *args: Any) -> None:
        LayoutItem.__init__(self)
        if len(args) == 0:
            self.child = None
        elif len(args) == 1:
            self.child = args[0]
        else:
            self.child = args

    def for_json(self) -> Any:
        print("Hiccup child", self.child)
        return self.child


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


View = JSRef("View")


class Row(LayoutItem):
    def __init__(self, *items: Any):
        super().__init__()
        self.items, self.options = flatten_layout_items(items, Row)

    def for_json(self) -> Any:
        return Hiccup(View.Row, self.options, *self.items)


class Column(LayoutItem):
    def __init__(self, *items: Any):
        super().__init__()
        self.items, self.options = flatten_layout_items(items, Column)

    def for_json(self) -> Any:
        return Hiccup(View.Column, self.options, *self.items)


def unwrap_for_json(x):
    while hasattr(x, "for_json"):
        x = x.for_json()
    return x


class CachedObject(LayoutItem):
    def __init__(self, value):
        self.id = str(uuid.uuid1())
        self.value = value

    def cache_id(self):
        return self.id

    def for_json(self):
        return unwrap_for_json(self.value)

    def _repr_mimebundle_(self, **kwargs: Any) -> Any:
        if hasattr(self.value, "_repr_mimebundle_"):
            return self.value._repr_mimebundle_(**kwargs)
        return super()._repr_mimebundle_(**kwargs)


def cache(value: Any) -> CachedObject:
    return CachedObject(value)
