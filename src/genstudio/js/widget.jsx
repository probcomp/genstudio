import * as AnyWidgetReact from "@anywidget/react";
import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import * as mobx from "mobx";
import * as mobxReact from "mobx-react-lite";
import * as React from "react";
import * as ReactDOM from "react-dom/client";
import * as api from "./api";
import { evaluateNdarray, inferDtype, estimateJSONSize } from "./binary";
import { $StateContext, CONTAINER_PADDING } from "./context";
import { serializeEvent, useCellUnmounted, tw } from "./utils";

const { createRender, useModelState, useModel, useExperimental } =
  AnyWidgetReact;
const { useState, useMemo, useCallback, useEffect } = React;

function resolveReference(path, obj) {
  return path.split(".").reduce((acc, key) => acc[key], obj);
}

function resolveRef(node, $state) {
  if (node && typeof node === "object" && node["__type__"] === "ref") {
    return resolveRef($state[node.state_key], $state);
  }
  return node;
}


async function createEvalEnv() {

  const env = { d3, Plot: api, html: api.html, React };

  // Process imports in order
  for (const [name, spec] of imports) {
    try {
      if (spec.type === 'module') {
        // ESM module from URL
        env[name] = await import(spec.url);
      } else if (spec.type === 'source') {
        if (spec.module) {
          // ES Module style source
          const blob = new Blob(
            [spec.source],
            { type: 'text/javascript' }
          );
          const url = URL.createObjectURL(blob);
          try {
            env[name] = await import(url);
          } finally {
            URL.revokeObjectURL(url);
          }
        } else {
          // CommonJS style source
          const moduleFactory = new Function(
            'React', 'd3', 'Plot',
            ...Object.keys(env),
            `
            const exports = {};
            const module = { exports };
            ${spec.source}
            return module.exports;
            `
          );
          env[name] = moduleFactory(
            React, d3, Plot,
            ...Object.values(env)
          );
        }
      }
    } catch (e) {
      console.error(`Failed to process import ${name}:`, e);
    }
  }
  return env;
}

export function evaluate(node, $state, experimental, buffers) {
  if (node === null || typeof node !== "object") return node;
  if (Array.isArray(node))
    return node.map((item) => evaluate(item, $state, experimental, buffers));
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
        console.error("Function not found", node);
        return null;
      }
      // functions marked as macros are passed unevaluated args + $state (for selective evaluation)
      const args = fn.macro
        ? [$state, ...node.args]
        : evaluate(node.args, $state, experimental, buffers);
      if (fn.prototype?.constructor === fn) {
        return new fn(...args);
      } else {
        return fn(...args);
      }
    case "js_ref":
      return resolveReference(node.path, api);
    case "js_source":
      const source = node.expression
        ? `return ${node.value.trimLeft()}`
        : node.value;
      const params = (node.params || []).map((p) =>
        evaluate(p, $state, experimental, buffers)
      );
      const paramVars = params.map((_, i) => `p${i}`);
      const code = source.replace(/%(\d+)/g, (_, i) => `p${parseInt(i) - 1}`);

      // Use the evaluation environment
      const env = $state.evalEnv;

      return new Function("$state", ...Object.keys(env), ...paramVars, code)(
        $state,
        ...Object.values(env),
        ...params
      );
    case "datetime":
      return new Date(node.value);
    case "ref":
      return $state.__computed(node.state_key);
    case "callback":
      if (experimental) {
        return (e) =>
          experimental.invoke("handle_callback", {
            id: node.id,
            event: serializeEvent(e),
          });
      } else {
        return undefined;
      }
    case "ndarray":
      if (node.data?.__type__ === "buffer") {
        node.data = buffers[node.data.index];
      }
      return evaluateNdarray(node);
    default:
      return Object.fromEntries(
        Object.entries(node).map(([key, value]) => [
          key,
          evaluate(value, $state, experimental, buffers),
        ])
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
  return updates.flatMap((entry) => {
    if (entry.constructor === Object) {
      return Object.entries(entry).map(([key, value]) => [key, "reset", value]);
    }
    // handle array format [key, operation, payload]
    const [key, operation, payload] = entry;
    return [[typeof key === "string" ? key : key.id, operation, payload]];
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
        __buffer_index__: index,
        __type__: "ndarray",
        dtype: inferDtype(value),
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
    if (value && typeof value === "object") {
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
   * Gets a deeply nested value from the state store using dot notation.
   *
   * Traverses nested objects/arrays by splitting the property path on dots.
   * For array access, converts string indices to integers.
   * Falls back to stateHandler.get() for non-nested properties.
   *
   * @param {Object} target - The state store target object
   * @param {string} prop - The property path using dot notation (e.g. "a.b.c" or "points.0.x")
   * @returns {any} The value at the specified path
   * @throws {Error} If the path cannot be resolved
   */
function getDeep(stateHandler, target, prop) {
  if (prop.includes(".")) {
    const parts = prop.split(".");
    const topKey = parts[0];
    const rest = parts.slice(1);

    // Then traverse the remaining path
    return rest.reduce((obj, key) => {
      if (Array.isArray(obj) || ArrayBuffer.isView(obj)) {
        return obj[parseInt(key)];
      }
      return obj[key];
    }, stateHandler.get(target, topKey));
  }
  return stateHandler.get(target, prop);
}

/**
 * Sets a deeply nested value in the state store using dot notation.
 * Creates new objects/arrays along the path to maintain proper reactivity.
 *
 * @param {Object} target - The state store target object
 * @param {string} prop - The property path using dot notation (e.g. "a.b.c" or "points.0.x")
 * @param {any} value - The value to set
 * @returns {boolean} True if the set operation succeeded
 */
function setDeep(stateHandler, target, prop, value) {
  if (!prop.includes(".")) {
    return stateHandler.set(target, prop, value);
  }

  const parts = prop.split(".");
  const first = parts[0];
  const current = stateHandler.get(target, first);

  // Build up the new object/array structure from bottom up
  let result = value;
  for (let i = parts.length - 1; i > 0; i--) {
    const key = parts[i];
    const parentKey = parts[i-1];
    const index = parseInt(key);
    const isArray = !isNaN(index);
    const parentIndex = parseInt(parentKey);
    const isParentArray = !isNaN(parentIndex);

    // Get the parent container we'll be modifying
    const parent = i === 1 ? current :
      (isParentArray ?
        (parentKey in current ? current[parentIndex] : []) :
        (parentKey in current ? current[parentKey] : {}));

    // Create the new container with our value
    if (isArray) {
      if (ArrayBuffer.isView(parent)) {
        const newArray = new parent.constructor(parent);
        newArray[index] = result;
        result = newArray;
      } else {
        result = Object.assign([...parent], {[index]: result});
      }
    } else {
      result = {...parent, [key]: result};
    }
  }

  return stateHandler.set(target, first, result);
}

/**
 * Creates a reactive state store with optional sync capabilities
 * @param {Object.<string, any>} initialState
 * @param {Object} experimental - The experimental interface for sync operations
 * @returns {Proxy} A proxied state store with reactive capabilities
 */
export function createStateStore({ initialState, syncedKeys, listeners = {}, experimental, buffers, evalEnv }) {
  syncedKeys = new Set(syncedKeys)
  const initialStateMap = mobx.observable.map(initialState, { deep: false });
  const computeds = {};
  const reactions = {};

  const stateHandler = {
    get(target, key) {
      if (key in target) return target[key];
      return target.__computed(key);
    },
    set: (_target, key, value) => {
      const newValue =
        typeof value === "function" ? value(initialStateMap.get(key)) : value;
      const updates = applyUpdates([[key, "reset", newValue]]);
      notifyPython(updates);
      return true;
    },
    ownKeys(_target) {
      return Array.from(initialStateMap.keys());
    },
    getOwnPropertyDescriptor(_target, key) {
      return {
        enumerable: true,
        configurable: true,
        value: this.get(_target, key),
      };
    },
  };

  // Track the current transaction depth and accumulated updates
  let updateDepth = 0;
  let transactionUpdates = null;

  function notifyPython(updates) {
    if (!experimental || !updates) return;
    updates = updates.filter(([key]) => syncedKeys.has(key));
    if (!updates.length) return;

    // if we're already in a transaction, just add to it.
    if (transactionUpdates) {
      transactionUpdates.push(...updates);
      return;
    }
    const [processedUpdates, buffers] = collectBuffers(updates);
    experimental.invoke(
      "handle_updates",
      {
        updates: processedUpdates,
      },
      { buffers }
    );
  }

  // notify python when computed state changes.
  // these are dependent reactions which will run within applyUpdates.
  const listenToComputed = (key, value) => {
    reactions[key]?.(); // clean up existing reaction, if it exists.
    const isComputed =
      value?.constructor === Object && value.__type__ === "js_source";
    if (syncedKeys.has(key) && isComputed) {
      reactions[key] = mobx.reaction(
        () => $state[key],
        (value) => notifyPython([[key, "reset", value]]),
        { fireImmediately: true }
      );
    }
  };

  const applyUpdates = (updates) => {
    // Track update depth and initialize accumulator at root level
    updateDepth++;
    const isRoot = updateDepth === 1;
    if (isRoot) {
      transactionUpdates = [];
    }

    // Add initial updates to accumulated list
    transactionUpdates.push(...updates);

    // Run updates within a mobx action to batch reactions
    mobx.action(() => {
      for (const update of updates) {
        const [key, operation, payload] = update;
        const init = $state[key];
        initialStateMap.set(key, applyUpdate($state, init, operation, payload));
      }
    })();

    // Notify JS listeners which may trigger more updates
    notifyJs(updates);

    updateDepth--;

    // Only return accumulated updates at root level
    if (isRoot) {
      const rootUpdates = transactionUpdates;
      transactionUpdates = null;
      return rootUpdates;
    }

    return null;
  };

  // notify js listeners when updates occur
  const notifyJs = (updates) => {
    updates.forEach(([key]) => {
      const keyListeners = listeners[key];
      if (keyListeners) {
        const value = $state[key];
        keyListeners.forEach((callback) => callback({ value }));
      }
    });
  };

  const $state = new Proxy(
    {
      evaluate: (ast) => evaluate(ast, $state, experimental, buffers),
      evalEnv,
      __backfill: function (data) {
        syncedKeys = new Set(data.syncedKeys);
        for (const [key, value] of Object.entries(data.initialState)) {
          if (!initialStateMap.has(key)) {
            initialStateMap.set(key, value);
          }
          listenToComputed(key, value);
        }
      },

      __resolveRef: function (node) {
        return resolveRef(node, $state);
      },

      __computed: function (key) {
        if (!(key in computeds)) {
          computeds[key] = mobx.computed(() => {
            return $state.evaluate(initialStateMap.get(key));
          });
        }
        return computeds[key].get();
      },

      __updateLocal: mobx.action(applyUpdates),

      update: (...updates) => {
        updates = applyUpdates(normalizeUpdates(updates));
        notifyPython(updates);
      },
    },
    {
      ...stateHandler,
      get: getDeep.bind(null, stateHandler),
      set: setDeep.bind(null, stateHandler),
    }
  );

  listeners = $state.evaluate(listeners);

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

  if (data && typeof data === "object") {
    if ("__buffer_index__" in data) {
      data.data = buffers[data.__buffer_index__];
      delete data.__buffer_index__;
      return data;
    }

    if (Array.isArray(data)) {
      data.forEach((item) => replaceBuffers(item, buffers));
    } else {
      Object.values(data).forEach((value) => replaceBuffers(value, buffers));
    }
  }
  return data;
};

export const StateProvider = mobxReact.observer(
  function (data) {
    const { ast, imports, initialState, model } = data
    const [evalEnv, setEnv] = useState(null);

    useEffect(() => {
      createEvalEnv(imports || {}).then(setEnv);
    }, [imports]);

    const $state = useMemo(
      () =>
        createStateStore({
          ...data,
          evalEnv,
        }),
      [evalEnv]
    );

    const [currentAst, setCurrentAst] = useState(null);

    useEffect(() => {
      // wait for env to load (async)
      if (!evalEnv) return;

      // when the widget is reset with a new ast/initialState, add missing entries
      // to the initialState and then reset the current ast.
      $state.backfill(data)
      setCurrentAst(ast)
    }, [ast, initialState, evalEnv])

  useEffect(() => {
    // if we have an AnyWidget model (ie. we are in widget model),
    // listen for `update_state` events.
    if (model) {
      const cb = (msg, buffers) => {
        if (msg.type === "update_state") {
          $state.__updateLocal(replaceBuffers(msg.updates, buffers));
        }
      };
      model.on("msg:custom", cb);
      return () => model.off("msg:custom", cb);
    }
  }, [initialState, model, $state]);

  if (!currentAst) return;

  return (
    <$StateContext.Provider value={$state}>
      <api.Node value={currentAst} />
    </$StateContext.Provider>
  );
});

function Viewer(data) {
  const [el, setEl] = useState();
  const elRef = useCallback((element) => element && setEl(element), [setEl]);
  const isUnmounted = useCellUnmounted(el?.parentNode);

  if (isUnmounted || !data) {
    return null;
  }

  return (
    <div
      className="genstudio-container"
      style={{ padding: CONTAINER_PADDING }}
      ref={elRef}
    >
      {el && <StateProvider {...data} />}
      {data.size && data.dev && (
        <div className={tw("text-xl p-3")}>{data.size}</div>
      )}
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
          size: estimateJSONSize(e.target.result),
        });
      }
    };
    reader.readAsText(file);
  };

  return (
    <div className={tw("p-3")}>
      <div
        className={tw(
          `border-2 border-dashed rounded-lg p-5 text-center ${
            dragActive ? "border-blue-500" : "border-gray-300"
          }`
        )}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <label
          htmlFor="file-upload"
          className={tw(
            "text-sm inline-block px-3 py-2 mb-2 text-white bg-blue-600 rounded-full cursor-pointer hover:bg-blue-700"
          )}
        >
          Choose a JSON file
        </label>
        <input
          type="file"
          id="file-upload"
          accept=".json"
          onChange={handleChange}
          className={tw("hidden")}
        />
        <p className={tw("text-sm text-gray-600")}>
          or drag and drop a JSON file here
        </p>
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
  renderFile,
};
