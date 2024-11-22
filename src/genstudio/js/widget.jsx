import * as AnyWidgetReact from "@anywidget/react";
import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import * as mobx from "mobx";
import * as mobxReact from "mobx-react-lite";
import * as React from "react";
import * as ReactDOM from "react-dom/client";
import * as api from "./api";
import {evaluateNdarray, inferDtype, estimateJSONSize} from "./binary"
import { $StateContext, CONTAINER_PADDING } from "./context";
import { serializeEvent, useCellUnmounted, tw } from "./utils";

const { createRender, useModelState, useModel, useExperimental } = AnyWidgetReact;
const { useState, useMemo, useCallback, useEffect } = React;

function resolveReference(path, obj) {
  return path.split('.').reduce((acc, key) => acc[key], obj);
}

function resolveRef(node, $state) {
  if (node && typeof node === 'object' && node["__type__"] === "ref") {
    return resolveRef($state[node.state_key], $state);
  }
  return node;
}

export function evaluate(node, $state, experimental, buffers) {
  if (node === null || typeof node !== 'object') return node;
  if (Array.isArray(node)) return node.map(item => evaluate(item, $state, experimental, buffers));
  if (node.constructor !== Object) {
    if (node instanceof DataView) {
      return node;
    }
    return node;
  }

  switch (node["__type__"]) {
    case "function":
      const fn = resolveReference(node.path, api);
      if (!fn) {
        console.error('Function not found', node);
        return null;
      }
      // functions marked as macros are passed unevaluated args + $state (for selective evaluation)
      const args = fn.macro ? [$state, ...node.args] : evaluate(node.args, $state, experimental, buffers)
      if (fn.prototype?.constructor === fn) {
        return new fn(...args);
      } else {
        return fn(...args);
      }
    case "js_ref":
      return resolveReference(node.path, api);
    case "js_source":
      const source = node.expression ? `return ${node.value.trimLeft()}` : node.value;
      const params = (node.params || []).map(p => evaluate(p, $state, experimental, buffers));
      const paramVars = params.map((_, i) => `p${i}`);
      const code = source.replace(/%(\d+)/g, (_, i) => `p${parseInt(i) - 1}`);
      return (new Function('$state', 'd3', 'Plot', ...paramVars, code))($state, d3, Plot, ...params);
    case "datetime":
      return new Date(node.value);
    case "ref":
      return $state.computed(node.state_key);
    case "callback":
      if (experimental) {
        return (e) => experimental.invoke("handle_callback", { id: node.id, event: serializeEvent(e) });
      } else {
        return undefined;
      }
    case "ndarray":
      if (node.data?.__type__ === 'buffer') {
        node.data = buffers[node.data.index]
      }
      return evaluateNdarray(node);
    default:
      return Object.fromEntries(
        Object.entries(node).map(([key, value]) => [key, evaluate(value, $state, experimental, buffers)])
      );
  }
}

function applyUpdate($state, init, op, payload) {
  const evaluatedPayload = $state.evaluate(payload);
  switch (op) {
    case "append":
      return [...init, evaluatedPayload];
    case "concat":
      return [...init, ...evaluatedPayload];
    case "reset":
      return evaluatedPayload;
    case "setAt":
      const [i, v] = evaluatedPayload;
      const newArray = init.slice();
      newArray[i] = v;
      return newArray;
    default:
      throw new Error(`Unknown operation: ${op}`);
  }
}

// normalize updates to handle both dict and array formats
function normalizeUpdates(updates) {
  return updates.flatMap(entry => {
    if (entry.constructor === Object) {
      return Object.entries(entry).map(([key, value]) => [key, 'reset', value]);
    }
    // handle array format [key, operation, payload]
    const [key, operation, payload] = entry;
    return [[typeof key === 'string' ? key : key.id, operation, payload]];
  });
}

function collectBuffers(data) {
  const buffers = [];

  function traverse(value) {
    // Handle ArrayBuffer and TypedArray instances
    if (value instanceof ArrayBuffer || ArrayBuffer.isView(value)) {
      const index = buffers.length;
      buffers.push(value);

      // Add metadata about the array type
      const metadata = {
        "__buffer_index__": index,
        "__type__": "ndarray",
        "dtype": inferDtype(value),
      };

      // Add shape if available
      if (value instanceof ArrayBuffer) {
        metadata.shape = [value.byteLength];
      } else {
        metadata.shape = [value.length];
      }

      return metadata;
    }

    // Handle arrays recursively
    if (Array.isArray(value)) {
      return value.map(traverse);
    }

    // Handle objects recursively
    if (value && typeof value === 'object') {
      const result = {};
      for (const [key, val] of Object.entries(value)) {
        result[key] = traverse(val);
      }
      return result;
    }

    // Return primitives as-is
    return value;
  }

  return [traverse(data), buffers];
}

/**
 * Creates a reactive state store with optional sync capabilities
 * @param {Object.<string, any>} initialState
 * @param {Object} experimental - The experimental interface for sync operations
 * @returns {Proxy} A proxied state store with reactive capabilities
 */
export function createStateStore({ initialState, syncedKeys, experimental, buffers }) {
  syncedKeys = new Set(syncedKeys)
  const initialStateMap = mobx.observable.map(initialState, { deep: false });
  const computeds = {};

  const stateHandler = {
    get(target, key) {
      if (key in target) return target[key];
      return target.computed(key);
    },
    set: mobx.action((_target, key, value) => {
      const newValue = typeof value === 'function' ? value(initialStateMap.get(key)) : value;
      initialStateMap.set(key, newValue);

      // Send sync update if this key should be synced
      if (experimental && syncedKeys.has(key)) {
        const [updates, buffers] = collectBuffers([[key, "reset", newValue]]);
        experimental.invoke("handle_updates", {
          updates: updates
        }, {buffers});
      }
      return true;
    }),
    ownKeys(_target) {
      return Array.from(initialStateMap.keys());
    },
    getOwnPropertyDescriptor(_target, key) {
      return {
        enumerable: true,
        configurable: true,
        value: this.get(_target, key)
      };
    }
  };

  const applyUpdates = (updates) => {
    for (const update of updates) {
      const [key, operation, payload] = update
      const init = $state[key]
      initialStateMap.set(key, applyUpdate($state, init, operation, payload));
    }
  }

  const notifyPython = (updates) => {
    if (!experimental) return;
    const syncUpdates = updates.filter((([key]) => syncedKeys.has(key)))
    if (syncUpdates?.length > 0) {
      const [processedUpdates, buffers] = collectBuffers(syncUpdates);
      experimental.invoke("handle_updates", {
        updates: processedUpdates
      }, {buffers});
    }
  }

  const $state = new Proxy({
    evaluate: (ast) => evaluate(ast, $state, experimental, buffers),

    backfill: function (data) {
      syncedKeys = new Set(data.syncedKeys)
      for (const [key, value] of Object.entries(data.initialState)) {
        if (!initialStateMap.has(key)) {
          initialStateMap.set(key, value);
        }
      }
    },

    resolveRef: function (node) { return resolveRef(node, $state); },

    computed: function (key) {
      if (!(key in computeds)) {
        computeds[key] = mobx.computed(() => $state.evaluate(initialStateMap.get(key)));
      }
      return computeds[key].get();
    },

    updateLocal: mobx.action(applyUpdates),

    update: mobx.action((...updates) => {
      const all_updates = normalizeUpdates(updates)
      applyUpdates(all_updates)
      notifyPython(all_updates)
    })
  }, stateHandler);

  return $state;
}

/**
 * Recursively replaces buffer index references with actual buffer data in a nested data structure.
 * Mirrors the functionality of replace_buffers() in widget.py.
 *
 * @param {Object|Array} data - The data structure containing buffer references
 * @param {Array<Buffer>} buffers - Array of binary buffers
 * @returns {Object|Array} The data structure with buffer references replaced with actual data
 */
const replaceBuffers = (data, buffers) => {
  if (!buffers || !buffers.length) return data;

  if (data && typeof data === 'object') {
    if ('__buffer_index__' in data) {
      data.data = buffers[data.__buffer_index__];
      delete data.__buffer_index__;
      return data;
    }

    if (Array.isArray(data)) {
      data.forEach(item => replaceBuffers(item, buffers));
    } else {
      Object.values(data).forEach(value => replaceBuffers(value, buffers));
    }
  }
  return data;
}

export const StateProvider = mobxReact.observer(
  function (data) {
    const { ast, initialState, experimental, model } = data
    const $state = useMemo(() => createStateStore(data), [])

    // a currentAst state field managed by the following useEffect hook,
    // to ensure that an ast is only rendered after $state has been populated
    // with the associated initialState entries.
    const [currentAst, setCurrentAst] = useState(null)

    useEffect(() => {
      // when the widget is reset with a new ast/initialState, add missing entries
      // to the initialState and then reset the current ast.
      $state.backfill(data)
      setCurrentAst(ast)
    }, [ast, initialState])

    useEffect(() => {
      // if we have an AnyWidget model (ie. we are in widget model),
      // listen for `update_state` events.
      if (model) {
        const cb = (msg, buffers) => {
          if (msg.type === 'update_state') {
            $state.updateLocal(replaceBuffers(msg.updates, buffers))
          }
        }
        model.on("msg:custom", cb);
        return () => model.off("msg:custom", cb)
      }
    }, [initialState, model])

    if (!currentAst) return;

    return (
      <$StateContext.Provider value={$state}>
        <api.Node value={currentAst} />
      </$StateContext.Provider>
    );
  }
)

function Viewer(data) {
  const [el, setEl] = useState();
  const elRef = useCallback((element) => element && setEl(element), [setEl])
  const isUnmounted = useCellUnmounted(el?.parentNode);

  if (isUnmounted || !data) {
    return null;
  }

  return (
    <div className="genstudio-container" style={{ "padding": CONTAINER_PADDING }} ref={elRef}>
      {el && <StateProvider {...data}/>}
      {data.size && data.dev && <div className={tw("text-xl p-3")}>{data.size}</div>}
    </div>
  );
}

function parseJSON(jsonString) {
  if (jsonString === null) return null;
  try {
    return JSON.parse(jsonString);
  } catch (error) {
    console.error("Error parsing JSON:", jsonString);
    console.error(error);
    return error;
  }
}

function FileViewer() {
  const [data, setData] = useState(null);
  const [dragActive, setDragActive] = useState(false);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (file) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const data = parseJSON(e.target.result);
      if (data instanceof Error) {
        alert("Error parsing JSON file. Please ensure it's a valid JSON.");
      } else {
        setData({
          ...data,
          size: estimateJSONSize(e.target.result)
        });
      }
    };
    reader.readAsText(file);
  };

  return (
    <div className={tw("p-3")}>
      <div
        className={tw(`border-2 border-dashed rounded-lg p-5 text-center ${dragActive ? 'border-blue-500' : 'border-gray-300'}`)}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <label htmlFor="file-upload" className={tw("text-sm inline-block px-3 py-2 mb-2 text-white bg-blue-600 rounded-full cursor-pointer hover:bg-blue-700")}>
          Choose a JSON file
        </label>
        <input
          type="file"
          id="file-upload"
          accept=".json"
          onChange={handleChange}
          className={tw("hidden")}
        />
        <p className={tw("text-sm text-gray-600")}>or drag and drop a JSON file here</p>
      </div>
      {data && (
        <div className={tw("mt-4")}>
          <h2 className={tw("text-lg mb-3")}>Loaded JSON Data:</h2>
          <Viewer {...data} />
        </div>
      )}
    </div>
  );
}

function AnyWidgetApp() {
  const [data, _setData] = useModelState("data");
  const experimental = useExperimental();
  const model = useModel();

  return <Viewer {...data} experimental={experimental} model={model} />;
}

export const renderData = (element, data, buffers) => {
  const root = ReactDOM.createRoot(element);
  root.render(<Viewer {...data} buffers={buffers} />);
};

export const renderFile = (element) => {
  const root = ReactDOM.createRoot(element);
  root.render(<FileViewer />);
};

export default {
  render: createRender(AnyWidgetApp),
  renderData,
  renderFile
}
