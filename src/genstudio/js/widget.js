import { WidthContext, CONTAINER_PADDING, $StateContext } from "./context";
import { html, useCellUnmounted, useElementWidth, serializeEvent } from "./utils";
import { AnyWidgetReact, React, ReactDOM, Plot, d3, mobx, mobxReact } from "./imports";
const { createRender, useModelState, useModel, useExperimental } = AnyWidgetReact;
const { useState, useMemo, useCallback, useEffect } = React;
import * as api from "./api";

function resolveReference(path, obj) {
  return path.split('.').reduce((acc, key) => acc[key], obj);
}

function resolveRef(node, $state) {
  if (node && typeof node === 'object' && node["__type__"] === "ref") {
    return resolveRef($state[node.id], $state);
  }
  return node;
}

export function evaluate(node, $state, experimental) {
  if (node === null || typeof node !== 'object') return node;
  if (Array.isArray(node)) return node.map(item => evaluate(item, $state, experimental));
  if (node.constructor !== Object) {
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
      return (new Function('$state', 'd3', 'Plot', source))($state, d3, Plot);
    case "datetime":
      return new Date(node.value);
    case "ref":
      return $state.computed(node.id);
    case "callback":
      if (experimental) {
        return (e) => experimental.invoke("handle_callback", { id: node.id, event: serializeEvent(e) });
      } else {
        return undefined;
      }
    default:
      return Object.fromEntries(
        Object.entries(node).map(([key, value]) => [key, evaluate(value, $state, experimental)])
      );
  }
}

function applyOperation($state, init, op, payload) {
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
      const newArray = [...init];
      newArray[i] = v;
      return newArray;
    default:
      throw new Error(`Unknown operation: ${op}`);
  }
}

// This function creates a reactive state management system.
// It maintains a graph of AST fragments, evaluates them on-demand,
// and allows for efficient updates to the underlying ASTs.
// The system supports lazy evaluation and interdependent state management.
export function createStateStore(initialState, experimental) {
  const initialStateMap = mobx.observable.map(initialState, { deep: false });
  const computeds = {};

  const stateHandler = {
    get(target, key) {
      if (key in target) return target[key];
      return target.computed(key);
    },
    set: mobx.action((_target, key, value) => {
      initialStateMap.set(key, typeof value === 'function' ? value(initialStateMap.get(key)) : value);
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

  const $state = new Proxy({
    evaluate: (ast) => evaluate(ast, $state, experimental),

    backfill: function(cache) {
      for (const key in cache) {
        if (!initialStateMap.has(key)) {
          initialStateMap.set(key, cache[key]);
        }
      }
    },

    resolveRef: function(node) { return resolveRef(node, $state); },

    computed: function(key) {
      if (!(key in computeds)) {
        computeds[key] = mobx.computed(() => $state.evaluate(initialStateMap.get(key)));
      }
      return computeds[key].get();
    },

    update: mobx.action(updates => {
      for (const [key, operation, payload] of updates) {
        initialStateMap.set(key, applyOperation($state, initialStateMap.get(key), operation, payload));
      }
    })
  }, stateHandler);

  return $state;
}

export const StateProvider = mobxReact.observer(
  function ({ ast, cache, experimental, model }) {

    const $state = useMemo(() => createStateStore(cache, experimental), [])

    // a currentAst state field managed by the following useEffect hook,
    // to ensure that an ast is only rendered after $state has been populated
    // with the associated cache entries.
    const [currentAst, setCurrentAst] = useState(null)

    useEffect(() => {
      // when the widget is reset with a new ast/cache, add missing entries
      // to the cache and then reset the current ast.
      $state.backfill(cache)
      setCurrentAst(ast)
    }, [ast, cache])

    useEffect(() => {
      // if we have an AnyWidget model (ie. we are in widget model),
      // listen for `update_state` events.
      if (model) {
        const cb = (msg) => {
          if (msg.type === 'update_state') {
            $state.update(JSON.parse(msg.updates))
          }
        }
        model.on("msg:custom", cb);
        return () => model.off("msg:custom", cb)
      }
    }, [cache, model])

    if (!currentAst) return;

    return html`
    <${$StateContext.Provider} value=${$state}>
      <${api.Node} value=${currentAst} />
    </${$StateContext.Provider}>
  `;
  }
)

function DataViewer(data) {
  const [el, setEl] = useState();
  const elRef = useCallback((element) => element && setEl(element), [setEl])
  const width = useElementWidth(el)
  const isUnmounted = useCellUnmounted(el?.parentNode);

  if (isUnmounted || !data) {
    return null;
  }

  const adjustedWidth = width ? width - CONTAINER_PADDING : undefined;

  return html`
    <${WidthContext.Provider} value=${adjustedWidth}>
      <div className="genstudio-container" style=${{ "padding": CONTAINER_PADDING }} ref=${elRef}>
        ${el && html`<${StateProvider} ...${data}/>`}
      </div>
      ${data.size && data.dev && html`<div className=${tw("text-xl p-3")}>${data.size}</div>`}
    </${WidthContext.Provider}>
  `;
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



function Viewer({ jsonString, ...data }) {
  const parsedData = useMemo(() => {
    if (jsonString) {
      try {
        return { ...data, size: estimateJSONSize(jsonString), ...JSON.parse(jsonString) };
      } catch (error) {
        console.error("Error parsing JSON:", error);
        return null;
      }
    }
    return data;
  }, [jsonString]);
  return html`<${DataViewer} ...${parsedData} />`;
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

  return html`
    <div className=${tw("p-3")}>
      <div
        className=${tw(`border-2 border-dashed rounded-lg p-5 text-center ${dragActive ? 'border-blue-500' : 'border-gray-300'}`)}
        onDragEnter=${handleDrag}
        onDragLeave=${handleDrag}
        onDragOver=${handleDrag}
        onDrop=${handleDrop}
      >
        <label htmlFor="file-upload" className=${tw("text-sm inline-block px-3 py-2 mb-2 text-white bg-blue-600 rounded-full cursor-pointer hover:bg-blue-700")}>
          Choose a JSON file
        </label>
        <input
          type="file"
          id="file-upload"
          accept=".json"
          onChange=${handleChange}
          className=${tw("hidden")}
        />
        <p className=${tw("text-sm text-gray-600")}>or drag and drop a JSON file here</p>
      </div>
      ${data && html`
        <div className=${tw("mt-4")}>
          <h2 className=${tw("text-lg mb-3")}>Loaded JSON Data:</h2>
          <${Viewer} ...${data} />
        </div>
      `}
    </div>
  `;
}

function AnyWidgetApp() {
  let [jsonString] = useModelState("data");
  const experimental = useExperimental();
  const model = useModel();
  return html`<${Viewer} ...${{ jsonString, experimental, model }} />`;
}

export const renderData = (element, data) => {
  const root = ReactDOM.createRoot(element);
  if (typeof data === 'string') {
    root.render(html`<${Viewer} jsonString=${data} />`);
  } else {
    root.render(html`<${Viewer} ...${data} />`);
  }
};

export const renderFile = (element) => {
  const root = ReactDOM.createRoot(element);
  root.render(html`<${FileViewer} />`);
};

export default {
  render: createRender(AnyWidgetApp),
  renderData,
  renderFile
}
