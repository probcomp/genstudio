from genstudio.widget import Widget
from typing import Any, Dict, List, Optional, Tuple, Union


class JSCall(dict):
    """Represents a JavaScript function call."""

    def __init__(self, module: str, name: str, args: Union[List[Any], Tuple[Any, ...]]):
        if not isinstance(args, (list, tuple)):
            print("not a list", type(args))
            raise TypeError("args must be a list")
        super().__init__(
            {"pyobsplot-type": "function", "module": module, "name": name, "args": args}
        )

    def _repr_mimebundle_(self, **kwargs: Any):
        return Widget(self)._repr_mimebundle_(**kwargs)


def js_ref(module: str, name: str) -> "JSRef":
    """Represents a reference to a JavaScript module or name."""
    return JSRef(module=module, name=name)


def js(txt: str) -> Dict[str, str]:
    """Represents raw JavaScript code to be evaluated."""
    return {"pyobsplot-type": "js", "value": txt}


class JSRef(dict):
    """Refers to a JavaScript module or name. When called, returns a function call representation."""

    def __init__(
        self,
        module: str,
        name: Optional[str] = None,
        label: Optional[str] = None,
        doc: Optional[str] = None,
    ):
        self.__name__ = name or label
        self.__doc__ = doc
        super().__init__({"pyobsplot-type": "ref", "module": module, "name": name})

    def __call__(self, *args: Any) -> Any:
        """Invokes the wrapped JavaScript function in the runtime with the provided arguments."""
        return JSCall(self["module"], self["name"], args)

    def __getattr__(self, name: str) -> "JSRef":
        """Returns a reference to a nested property or method of the JavaScript object."""
        if name[0] == "_":
            return super().__getattribute__(name)
        elif self["name"] is None:
            return JSRef(self["module"], name)
        else:
            raise ValueError("Only module.name paths are currently supported")
            # return JSRef(f"{self['module']}.{self['name']}", name)
