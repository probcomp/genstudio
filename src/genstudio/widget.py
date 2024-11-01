import datetime
import orjson
import uuid

from typing import Any, Iterable, Callable, Dict

import anywidget
import traitlets

from genstudio.util import PARENT_PATH, CONFIG


class CollectedState:
    # collect initial state while serializing data.
    def __init__(self):
        self.initialState = {}
        self.initialStateJSON = {}
        self.listeners = {}

    def state_entry(self, state_key, value, sync=False, **kwargs):
        if state_key not in self.initialState:
            _entry = {"sync": sync, "value": value}
            self.initialState[state_key] = _entry
            # perform to_json conversion immediately
            self.initialStateJSON[state_key] = orjson.Fragment(
                to_json(_entry, **kwargs)
            )
        return {"__type__": "ref", "state_key": state_key}

    def add_listeners(self, listeners):
        for state_key, listener in listeners.items():
            if state_key not in self.listeners:
                self.listeners[state_key] = []
            self.listeners[state_key].append(listener)
        return None


def to_json(data, collected_state=None, widget=None):
    def default(obj):
        if collected_state is not None:
            if hasattr(obj, "_state_key"):
                return collected_state.state_entry(
                    state_key=obj._state_key(),
                    value=obj.for_json(),
                    sync=getattr(obj, "ref_sync", False),
                    widget=widget,
                    collected_state=collected_state,
                )
            if hasattr(obj, "_state_listeners"):
                return collected_state.add_listeners(obj.state_listeners)
        if hasattr(obj, "for_json"):
            return obj.for_json()
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
            return {"__type__": "datetime", "value": obj.isoformat()}
        elif callable(obj):
            if widget is not None:
                id = str(uuid.uuid4())
                widget.callback_registry[id] = obj
                return {"__type__": "callback", "id": id}
            return None
        else:
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    return orjson.dumps(data, default=default).decode("utf-8")


def to_json_with_initialState(ast: Any, widget: "Widget | None" = None):
    collected_state = CollectedState()
    astJSON = orjson.Fragment(
        to_json(ast, widget=widget, collected_state=collected_state)
    )
    listenersJSON = orjson.Fragment(
        to_json(
            collected_state.listeners, widget=widget, collected_state=collected_state
        )
    )

    json = to_json(
        {
            "ast": astJSON,
            "initialState": collected_state.initialStateJSON,
            "listeners": listenersJSON,
            **CONFIG,
        }
    )

    if widget is not None:
        widget.set_initialState(collected_state.initialState)
    return json


def entry_id(key):
    return key if isinstance(key, str) else key.id


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
        self._on_change = {}
        self._synced_vars = set()

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

    def notify_callbacks(self, updates):
        for name, operation, value in updates:
            callback = self._on_change.get(name)
            if callback:
                callback({"id": name, "value": self._state[name]})

    # update values from python - send to js
    def update(self, *updates):
        updates = normalize_updates(updates)

        # apply updates locally for synced state
        synced_updates = [
            [name, op, payload]
            for name, op, payload in updates
            if entry_id(name) in self._synced_vars
        ]
        apply_updates(self._state, synced_updates)

        # send all updates to JS regardless of sync status
        self._widget.send(
            {"type": "update_state", "updates": to_json(updates, widget=self)}
        )

        self.notify_callbacks(synced_updates)

    # accept updates from js - notify callbacks
    def accept_js_updates(self, updates):
        apply_updates(self._state, updates)
        self.notify_callbacks(updates)

    def onChange(self, callbacks):
        self._on_change.update(callbacks)

    def backfill(self, initialState):
        for key, entry in initialState.items():
            if entry["sync"]:
                self._synced_vars.add(key)
                if key not in self._state:
                    self._state[key] = entry["value"]
            else:
                if key in self._state:
                    del self._state[key]
                self._synced_vars.discard(key)


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

    def set_initialState(self, initialState):
        self.state.backfill(initialState)

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
