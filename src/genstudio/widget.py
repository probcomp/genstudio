import datetime
import uuid

from typing import Any, Iterable, Callable, Dict

import anywidget
import traitlets

from genstudio.util import PARENT_PATH, CONFIG


class CollectedState:
    # collect initial state while serializing data.
    def __init__(self):
        self.syncedKeys = set()
        self.initialState = {}
        self.initialStateJSON = {}
        self.stateListeners = {}

    def state_entry(self, state_key, value, sync=False, **kwargs):
        if sync:
            self.syncedKeys.add(state_key)
        if state_key not in self.initialStateJSON:
            self.initialState[state_key] = value
            self.initialStateJSON[state_key] = to_json(value, **kwargs)
        return {"__type__": "ref", "state_key": state_key}

    def add_listeners(self, listeners):
        for state_key, listener in listeners.items():
            self.syncedKeys.add(state_key)
            if state_key not in self.stateListeners:
                self.stateListeners[state_key] = []
            self.stateListeners[state_key].append(listener)
        return None


def to_json(data, collected_state=None, widget=None):
    # Handle basic JSON-serializable types first since they're most common
    if isinstance(data, (str, int, float, bool)):
        return data

    # Handle None case
    if data is None:
        return None

    # Handle datetime objects early since isinstance check is fast
    if isinstance(data, (datetime.date, datetime.datetime)):
        return {"__type__": "datetime", "value": data.isoformat()}

    # Handle state-related objects
    if collected_state is not None:
        if hasattr(data, "_state_key"):
            return collected_state.state_entry(
                state_key=data._state_key,
                value=data.for_json(),
                sync=getattr(data, "_state_sync", False),
                widget=widget,
                collected_state=collected_state,
            )
        if hasattr(data, "_state_listeners"):
            return collected_state.add_listeners(data._state_listeners)

    # Handle objects with custom serialization
    if hasattr(data, "for_json"):
        return to_json(data.for_json(), collected_state=collected_state, widget=widget)
    if hasattr(data, "tolist"):
        return data.tolist()

    # Handle containers
    if isinstance(data, dict):
        return {k: to_json(v, collected_state, widget) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [to_json(x, collected_state, widget) for x in data]
    if isinstance(data, Iterable):
        if not hasattr(data, "__len__") and not hasattr(data, "__getitem__"):
            print(
                f"Warning: Potentially exhaustible iterator encountered: {type(data).__name__}"
            )
        return [to_json(x, collected_state, widget) for x in data]

    # Handle callable objects
    if callable(data):
        if widget is not None:
            id = str(uuid.uuid4())
            widget.callback_registry[id] = data
            return {"__type__": "callback", "id": id}
        return None

    # Raise error for unsupported types
    raise TypeError(f"Object of type {type(data)} is not JSON serializable")


def to_json_with_initialState(ast: Any, widget: "Widget | None" = None):
    collected_state = CollectedState()
    ast = to_json(ast, widget=widget, collected_state=collected_state)

    json = to_json(
        {
            "ast": ast,
            "initialState": collected_state.initialStateJSON,
            "syncedKeys": collected_state.syncedKeys,
            **CONFIG,
        }
    )

    if widget is not None:
        widget.state.init_state(collected_state)
    return json


def entry_id(key):
    return key if isinstance(key, str) else key._state_key


def normalize_updates(updates):
    out = []
    for entry in updates:
        if isinstance(entry, dict):
            for key, value in entry.items():
                out.append([entry_id(key), "reset", value])
        else:
            out.append([entry_id(entry[0]), entry[1], entry[2]])
    return out


def apply_updates(state, updates):
    for name, operation, payload in updates:
        if operation == "append":
            if name not in state:
                state[name] = []
            state[name].append(payload)
        elif operation == "concat":
            if name not in state:
                state[name] = []
            state[name].extend(payload)
        elif operation == "reset":
            state[name] = payload
        elif operation == "setAt":
            index, value = payload
            if name not in state:
                state[name] = []
            state[name][index] = value
        else:
            raise ValueError(f"Unknown operation: {operation}")


class WidgetState:
    def __init__(self, widget):
        self._state = {}
        self._widget = widget
        self._syncedKeys = set()
        self._listeners = {}

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

    def notify_listeners(self, updates):
        for name, operation, value in updates:
            for listener in self._listeners.get(name, []):
                listener(self._widget, {"id": name, "value": self._state[name]})

    # update values from python - send to js
    def update(self, *updates):
        updates = normalize_updates(updates)

        # apply updates locally for synced state
        synced_updates = [
            [name, op, payload]
            for name, op, payload in updates
            if entry_id(name) in self._syncedKeys
        ]
        apply_updates(self._state, synced_updates)

        # send all updates to JS regardless of sync status
        self._widget.send(
            {"type": "update_state", "updates": to_json(updates, widget=self)}
        )

        self.notify_listeners(synced_updates)

    # accept updates from js - notify callbacks
    def accept_js_updates(self, updates):
        apply_updates(self._state, updates)
        self.notify_listeners(updates)

    def init_state(self, collected_state):
        self._listeners = collected_state.stateListeners or {}
        self._syncedKeys = syncedKeys = collected_state.syncedKeys

        for key, value in collected_state.initialState.items():
            if key in syncedKeys and key not in self._state:
                self._state[key] = value


class Widget(anywidget.AnyWidget):
    _esm = PARENT_PATH / "js/widget_build.js"
    _css = PARENT_PATH / "widget.css"
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
            f(self, params["event"])
        return "ok", []

    @anywidget.experimental.command  # type: ignore
    def handle_updates(
        self, params: dict[str, Any], buffers: list[bytes]
    ) -> tuple[str, list[bytes]]:
        self.state.accept_js_updates(params["updates"])

        return "ok", []
