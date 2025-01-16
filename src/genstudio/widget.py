import datetime
import uuid
from typing import Any, Callable, Dict, Iterable, List, Optional, Union
from types import SimpleNamespace

import anywidget
import numpy as np
import traitlets
import warnings

from genstudio.util import CONFIG, WIDGET_URL, CSS_URL


class SubscriptableNamespace(SimpleNamespace):
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)


class CollectedState:
    # collect initial state while serializing data.
    def __init__(self):
        self.syncedKeys = set()
        self.initialState = {}
        self.initialStateJSON = {}
        self.listeners = {"py": {}, "js": {}}
        self.imports = []  # List of import specs

    def state_entry(self, state_key, value, sync=False, **kwargs):
        if sync:
            self.syncedKeys.add(state_key)
        if state_key not in self.initialStateJSON:
            self.initialState[state_key] = value
            self.initialStateJSON[state_key] = to_json(value, **kwargs)
        return {"__type__": "ref", "state_key": state_key}

    def add_import(self, spec: dict):
        """Add an import specification.

        Args:
            spec: Import specification with format, source info, and options
        """
        self.imports.append(spec)
        return None

    def _add_listener(self, state_key, listener):
        listeners = [listener] if not isinstance(listener, list) else listener
        for listener in listeners:
            target = "py" if callable(listener) else "js"
            self.listeners[target].setdefault(state_key, []).append(listener)

    def add_listeners(self, listeners):
        for state_key, listener in listeners.items():
            self.syncedKeys.add(state_key)
            self._add_listener(state_key, listener)
        return None


def serialize_binary_data(
    buffers: Optional[List[bytes | bytearray | memoryview]], entry
):
    if buffers is None:
        return entry

    buffers.append(entry["data"])
    index = len(buffers) - 1
    return {
        **entry,
        "__buffer_index__": index,
        "data": None,
    }


def to_json(
    data,
    collected_state=None,
    widget=None,
    buffers: Optional[List[bytes | bytearray | memoryview]] = None,
):
    # Handle NaN at top level
    if isinstance(data, float):
        if np.isnan(data):
            return None
        return data

    # Handle basic JSON-serializable types first since they're most common
    if isinstance(data, (str, int, bool)):
        return data

    # Handle None case
    if data is None:
        return None

    # Handle binary data
    if isinstance(data, (bytes, bytearray, memoryview)):
        if buffers is not None:
            # Store binary data in buffers and return reference
            buffer_index = len(buffers)
            buffers.append(data)
            return {"__type__": "buffer", "index": buffer_index}
        return data

    # Handle datetime objects early since isinstance check is fast
    if isinstance(data, (datetime.date, datetime.datetime)):
        return {"__type__": "datetime", "value": data.isoformat()}

    # Handle state-related objects
    if collected_state is not None:
        if hasattr(data, "_state_imports"):
            for spec in data._state_imports:
                collected_state.add_import(spec)
            return None
        if hasattr(data, "_state_key"):
            return collected_state.state_entry(
                state_key=data._state_key,
                value=data.for_json(),
                sync=getattr(data, "_state_sync", False),
                widget=widget,
                collected_state=collected_state,
            )
        if hasattr(data, "_state_listeners"):
            collected_state.add_listeners(data._state_listeners)
            return None

    # Handle numpy and jax arrays
    if isinstance(data, np.ndarray) or type(data).__name__ in (
        "DeviceArray",
        "Array",
        "ArrayImpl",
    ):
        try:
            if data.ndim == 0:  # It's a scalar
                return data.item()
        except AttributeError:
            pass

        bytes_data = data.tobytes()
        return serialize_binary_data(
            buffers,
            {
                "__type__": "ndarray",
                "data": bytes_data,
                "dtype": str(data.dtype),
                "shape": data.shape,
            },
        )

    # Handle objects with custom serialization
    if hasattr(data, "for_json") and callable(data.for_json):
        return to_json(
            data.for_json(),
            collected_state=collected_state,
            widget=widget,
            buffers=buffers,
        )

    # Handle objects with attributes_dict method
    if hasattr(data, "attributes_dict") and callable(data.attributes_dict):
        return to_json(
            data.attributes_dict(),
            collected_state=collected_state,
            widget=widget,
            buffers=buffers,
        )

    # Handle containers
    if isinstance(data, dict):
        # if "__type__" in data and data["__type__"] == "ndarray":
        #     raise ValueError("Found __type__ in dict - this indicates double serialization")
        return {
            k: to_json(v, collected_state, widget, buffers) for k, v in data.items()
        }

    if isinstance(data, (list, tuple)):
        return [to_json(x, collected_state, widget, buffers) for x in data]

    if isinstance(data, Iterable):
        if not hasattr(data, "__len__") and not hasattr(data, "__getitem__"):
            warnings.warn(
                "Potentially exhaustible iterator encountered: generator", UserWarning
            )
        return [to_json(x, collected_state, widget, buffers) for x in data]

    # Handle callable objects
    if callable(data):
        if widget is not None:
            id = str(uuid.uuid4())
            widget.callback_registry[id] = data
            return {"__type__": "callback", "id": id}
        return None

    # Raise error for unsupported types
    raise TypeError(f"Object of type {type(data)} is not JSON serializable")


def to_json_with_initialState(
    ast: Any,
    widget: "Widget | None" = None,
    buffers: List[bytes | bytearray | memoryview] | None = None,
):
    collected_state = CollectedState()
    ast = to_json(ast, widget=widget, collected_state=collected_state, buffers=buffers)

    json = to_json(
        {
            "ast": ast,
            "initialState": collected_state.initialStateJSON,
            "syncedKeys": collected_state.syncedKeys,
            "listeners": collected_state.listeners["js"],
            "imports": collected_state.imports,
            **CONFIG,
        },
        buffers=buffers,
    )

    if widget is not None:
        widget.state.init_state(collected_state)
    return json


def entry_id(key):
    return key if isinstance(key, str) else key._state_key


def normalize_updates(
    updates: Iterable[Union[List[Any], Dict[str, Any]]],
) -> List[List[Any]]:
    out = []
    for entry in updates:
        if isinstance(entry, dict):
            for key, value in entry.items():
                out.append([entry_id(key), "reset", value])
        else:
            out.append([entry_id(entry[0]), entry[1], entry[2]])
    return out


def apply_updates(state: Dict[str, Any], updates: List[List[Any]]) -> None:
    for name, operation, payload in updates:
        if operation == "append":
            if name not in state:
                state[name] = []
            state[name] = state[name] + [payload]
        elif operation == "concat":
            if name not in state:
                state[name] = []
            state[name] = state[name] + list(payload)
        elif operation == "reset":
            state[name] = payload
        elif operation == "setAt":
            index, value = payload
            if name not in state:
                state[name] = []
            state[name] = state[name][:index] + [value] + state[name][index + 1 :]
        else:
            raise ValueError(f"Unknown operation: {operation}")


def deserialize_buffer_entry(data: dict, buffers: List[bytes]) -> Any:
    """Parse a buffer entry, converting to numpy array if needed."""
    buffer_idx = data["__buffer_index__"]
    if "__type__" in data and data["__type__"] == "ndarray":
        # Convert buffer to numpy array
        buffer = buffers[buffer_idx]
        dtype = data.get("dtype", "float64")
        shape = data.get("shape", [len(buffer)])
        return np.frombuffer(buffer, dtype=dtype).reshape(shape)
    return buffers[buffer_idx]


def replace_buffers(data: Any, buffers: List[bytes]) -> Any:
    """Replace buffer indices with actual buffer data in a nested data structure."""
    if not buffers:
        return data

    # Fast path for direct buffer reference
    if isinstance(data, dict):
        if "__buffer_index__" in data:
            return deserialize_buffer_entry(data, buffers)

        # Process dictionary values in-place
        for k, v in data.items():
            if isinstance(v, dict) and "__buffer_index__" in v:
                data[k] = deserialize_buffer_entry(v, buffers)
            elif isinstance(v, (dict, list, tuple)):
                data[k] = replace_buffers(v, buffers)
        return data

    # Fast path for non-container types
    if not isinstance(data, (dict, list, tuple)):
        return data

    if isinstance(data, list):
        # Mutate list in-place
        for i, x in enumerate(data):
            if isinstance(x, dict) and "__buffer_index__" in x:
                data[i] = deserialize_buffer_entry(x, buffers)
            elif isinstance(x, (dict, list, tuple)):
                data[i] = replace_buffers(x, buffers)
        return data

    # Handle tuples
    result = list(data)
    modified = False
    for i, x in enumerate(data):
        if isinstance(x, dict) and "__buffer_index__" in x:
            result[i] = deserialize_buffer_entry(x, buffers)
            modified = True
        elif isinstance(x, (dict, list, tuple)):
            new_val = replace_buffers(x, buffers)
            if new_val is not x:
                result[i] = new_val
                modified = True

    if modified:
        return tuple(result)
    return data


class WidgetState:
    def __init__(self, widget):
        self._state = {}
        self._widget = widget
        self._syncedKeys = set()
        self._listeners = {}
        self._processing_listeners = (
            set()
        )  # Track which listeners are currently processing

    def __getattr__(self, name):
        if name in self._state:
            return self._state[name]
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self._state[name] = value
            self.update([name, "reset", value])

    def notify_listeners(self, updates: List[List[Any]]) -> None:
        for name, operation, value in updates:
            for listener in self._listeners.get(name, []):
                # Skip if this listener is already being processed
                if listener in self._processing_listeners:
                    continue
                try:
                    self._processing_listeners.add(listener)
                    listener(
                        self._widget,
                        SubscriptableNamespace(id=name, value=self._state[name]),
                    )
                finally:
                    self._processing_listeners.remove(listener)

    # update values from python - send to js
    def update(self, *updates: Union[List[Any], Dict[str, Any]]) -> None:
        normalized_updates = normalize_updates(updates)

        # apply updates locally for synced state
        synced_updates = [
            [name, op, payload]
            for name, op, payload in normalized_updates
            if entry_id(name) in self._syncedKeys
        ]
        apply_updates(self._state, synced_updates)

        # send all updates to JS regardless of sync status
        buffers: List[bytes | bytearray | memoryview] = []

        json_updates = to_json(normalized_updates, widget=self, buffers=buffers)
        self._widget.send(
            {"type": "update_state", "updates": json_updates}, buffers=buffers
        )

        self.notify_listeners(synced_updates)

    # accept updates from js - notify callbacks
    def accept_js_updates(self, updates: List[List[Any]]) -> None:
        apply_updates(self._state, updates)
        self.notify_listeners(updates)

    def init_state(self, collected_state):
        self._listeners = collected_state.listeners["py"]
        self._syncedKeys = syncedKeys = collected_state.syncedKeys

        for key, value in collected_state.initialState.items():
            if key in syncedKeys and key not in self._state:
                self._state[key] = value


class Widget(anywidget.AnyWidget):
    _esm = WIDGET_URL
    _css = CSS_URL
    callback_registry: Dict[str, Callable] = {}
    data = traitlets.Any().tag(sync=True, to_json=to_json_with_initialState)

    def __init__(self, ast: Any):
        self.state = WidgetState(self)
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
            event = replace_buffers(params["event"], buffers)
            event = SubscriptableNamespace(**event)
            f(self, event)
        return "ok", []

    @anywidget.experimental.command  # type: ignore
    def handle_updates(
        self, params: dict[str, Any], buffers: list[bytes]
    ) -> tuple[str, list[bytes]]:
        updates = replace_buffers(params["updates"], buffers)
        self.state.accept_js_updates(updates)
        return "ok", []
