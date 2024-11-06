import * as AnyWidgetReact from "@anywidget/react";
import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import * as mobx from "mobx";
import * as mobxReact from "mobx-react-lite";
import * as React from "react";
import * as ReactDOM from "react-dom/client";
import * as api from "./api";
import { $StateContext, CONTAINER_PADDING } from "./context";
import { serializeEvent, useCellUnmounted, useElementWidth, tw } from "./utils";

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

function reshapeArray(flat, dims, offset = 0) {
  const [dim, ...restDims] = dims;

  if (restDims.length === 0) {
    const start = offset;
    const end = offset + dim;
    return flat.slice(start, end);
  }

  const stride = restDims.reduce((a, b) => a * b, 1);
  return Array.from({ length: dim }, (_, i) =>
    reshapeArray(flat, restDims, offset + i * stride)
  );
}

function evaluateNdarray(node) {
  const { data, dtype, shape } = node;
  console.log("JS side - ndarray info:");
  console.log("  Shape:", shape);
  console.log("  Dtype:", dtype);
  console.log("  Data type:", typeof data);
  console.log("  Data constructor:", data?.constructor?.name);
  console.log("  DataView byteLength:", data?.byteLength);
  console.log("  DataView buffer size:", data?.buffer?.byteLength);
  console.log("  DataView:", data)

  console.time("evaluateNdarray")
  const dtypeMap = {
    'float32': Float32Array,
    'float64': Float64Array,
    'int8': Int8Array,
    'int16': Int16Array,
    'int32': Int32Array,
    'uint8': Uint8Array,
    'uint16': Uint16Array,
    'uint32': Uint32Array,
  };
  const ArrayConstructor = dtypeMap[dtype] || Float64Array;

  // Create typed array directly from the DataView's buffer
  const flatArray = new ArrayConstructor(
    data.buffer,
    data.byteOffset,
    data.byteLength / ArrayConstructor.BYTES_PER_ELEMENT
  );
  // If 1D, return the typed array directly
  if (shape.length <= 1) {
    return { data: flatArray, shape, dtype };
  }



  const reshapedArray = reshapeArray(flatArray, shape);
  console.timeEnd("evaluateNdarray")

  return reshapedArray;
}

export function evaluate(node, $state, experimental) {
  if (node === null || typeof node !== 'object') return node;
  if (Array.isArray(node)) return node.map(item => evaluate(item, $state, experimental));
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
      const args = fn.macro ? node.args : evaluate(node.args, $state, experimental)
      if (fn.prototype?.constructor === fn) {
        return new fn(...args);
      } else {
        return fn(...args);
      }
    case "js_ref":
      return resolveReference(node.path, api);
    case "js_source":
      const source = node.expression ? `return ${node.value}` : node.value;
      const params = (node.params || []).map(p => evaluate(p, $state, experimental));
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
      return evaluateNdarray(node);
    default:
      return Object.fromEntries(
        Object.entries(node).map(([key, value]) => [key, evaluate(value, $state, experimental)])
      );
  }
}

function applyUpdate($state, init, op, payload) {
  const evaluatedPayload = $state.evaluate(payload);
  console.log("applying update", evaluatedPayload)
  switch (op) {
    case "append":
      return [...init, evaluatedPayload];
    case "concat":
      return [...init, ...evaluatedPayload];
    case "reset":
      return evaluatedPayload;
    case "setAt":
      const [i, v] = evaluatedPayload;
      const newArray = [...init];
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

// TODO
// support sending buffers back to python
function collectBuffers(data) {
  return [data, []]
}
// function collectBuffers(data) {
//   const buffers = [];

//   function traverse(obj) {
//     if (!obj || typeof obj !== 'object') return obj;

//     if (obj instanceof DataView || obj instanceof ArrayBuffer) {
//       const index = buffers.length;
//       buffers.push(obj instanceof DataView ? obj : new DataView(obj));
//       return { "__buffer_index__": index };
//     }

//     if (Array.isArray(obj)) {
//       return obj.map(traverse);
//     }

//     return Object.fromEntries(
//       Object.entries(obj).map(([k, v]) => [k, traverse(v)])
//     );
//   }

//   const processed = traverse(data);
//   return [processed, buffers];
// }

/**
 * Creates a reactive state store with optional sync capabilities
 * @param {Object.<string, any>} initialState
 * @param {Object} experimental - The experimental interface for sync operations
 * @returns {Proxy} A proxied state store with reactive capabilities
 */
export function createStateStore({ initialState, syncedKeys, experimental }) {
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
          updates: JSON.stringify(updates)
        }, buffers);
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
      initialStateMap.set(key, applyUpdate($state, initialStateMap.get(key), operation, payload));
    }
  }

  const notifyPython = (updates) => {
    if (!experimental) return;
    const syncUpdates = updates.filter((([key]) => syncedKeys.has(key)))
    if (syncUpdates?.length > 0) {
      console.log("Notify python")
      const [processedUpdates, buffers] = collectBuffers(syncUpdates);
      experimental.invoke("handle_updates", {
        updates: processedUpdates
      }, buffers);
    }
  }

  const $state = new Proxy({
    evaluate: (ast) => evaluate(ast, $state, experimental),

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
  console.time("replaceBuffers")
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
  console.timeEnd("replaceBuffers")
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
          console.log("msg", msg)
          if (msg.type === 'update_state') {
            console.log("update_state", msg, buffers)
            let updates;
            try {
              updates = replaceBuffers(msg.updates, buffers);
            } catch (err) {
              console.error("Error replacing buffers:", err);
              updates = msg.updates;
            }

            $state.updateLocal(updates)
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

function estimateJSONSize(jsonString) {
  if (!jsonString) return '0 B';

  // Use TextEncoder to get accurate byte size for UTF-8 encoded string
  const encoder = new TextEncoder();
  const bytes = encoder.encode(jsonString).length;

  // Convert bytes to KB or MB
  if (bytes < 1024) {
    return `${bytes} B`;
  } else if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(2)} KB`;
  } else {
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  }
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

export const renderData = (element, data) => {
  const root = ReactDOM.createRoot(element);
  if (typeof data === 'string') {
    root.render(<Viewer {...JSON.parse(data)} />);
  } else {
    root.render(<Viewer {...data} />);
  }
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
