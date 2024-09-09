import { WidthContext, EvaluateContext, CONTAINER_PADDING, $StateContext, AUTOGRID_MIN as AUTOGRID_MIN_WIDTH } from "./context";
import { html, useCellUnmounted, useElementWidth, serializeEvent } from "./utils";
import { AnyWidgetReact, React, ReactDOM, Plot, d3, mobx, mobxReact } from "./imports";
const { createRender, useModelState, useModel, useExperimental } = AnyWidgetReact;
const { useState, useMemo, useCallback, useEffect } = React;
import * as api from "./api";

const layoutComponents = new Set(['Hiccup', 'Grid', 'Row', 'Column']);

export function collectReactiveInitialState(ast) {
  let initialState = {};

  function traverse(node) {
    if (!node) return;
    if (typeof node === 'object' && node['__type__'] === 'function') {
      if (node.path === 'Reactive') {
        const { state_key, init } = new api.Reactive(node.args[0], initialState).options;
        initialState[state_key] = init;
      } else if (layoutComponents.has(node.path)) {
        node.args.forEach(traverse);
      }
    } else if (Array.isArray(node)) {
      node.forEach(traverse);
    }
  }

  traverse(ast);
  return initialState;
}
function resolveReference(path, obj) {
  return path.split('.').reduce((acc, key) => acc[key], obj);
}

function resolveCached(node, cache) {
  if (node && typeof node === 'object' && node["__type__"] === "cached") {
    return cache.get(node.id);
  }
  return node;
}

export function evaluate(node, cache, $state, experimental) {

  if (node === null || typeof node !== 'object') return node;
  if (Array.isArray(node)) return node.map(item => evaluate(item, cache, $state, experimental));
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
      const args = fn.macro ? node.args : evaluate(node.args, cache, $state, experimental)
      if (fn.prototype?.constructor === fn) {
        return new fn(...args);
      } else {
        return fn(...args);
      }
    case "ref":
      return resolveReference(node.path, api);
    case "js":
      return (new Function('$state', "$cache", 'd3', 'Plot', `return ${node.value}`))($state, cache, d3, Plot);
    case "datetime":
      return new Date(node.value);
    case "cached":
      return cache.get(node.id);
    case "callback":
      if (experimental) {
        return (e) => experimental.invoke("handle_callback", { id: node.id, event: serializeEvent(e) });
      } else {
        return undefined;
      }
    default:
      return Object.fromEntries(
        Object.entries(node).map(([key, value]) => [key, evaluate(value, cache, $state, experimental)])
      );
  }
}

export function useStateStore(ast) {
  const initialState = useMemo(() => collectReactiveInitialState(ast), [ast]);

  const store = useMemo(() => {
    const state = mobx.observable(initialState);
    return new Proxy(state, {
      set: mobx.action((target, prop, value) => {
        if (typeof value === 'function') {
          target[prop] = value(target[prop]);
        } else {
          target[prop] = value;
        }
        return true;
      })
    });
  }, []);

  useEffect(() => {
    mobx.action(() => {
      Object.keys(initialState).forEach(key => {
        if (!(key in store)) {
          store[key] = initialState[key];
        }
      });
    })
  }, [initialState])

  return store
}

function applyOperation(cacheStore, initialValue, operation, payload) {
  const evaluatedPayload = cacheStore.evaluate(payload);
  switch (operation) {
    case "append":
      return [...initialValue, evaluatedPayload];
    case "concat":
      return [...initialValue, ...evaluatedPayload];
    case "reset":
      return evaluatedPayload;
    default:
      throw new Error(`Unknown operation: ${operation}`);
  }
}

function createCacheStore(initialCache, $state, experimental) {
  const store = {
    initialCache,
    updates: mobx.observable.map({}),
    $state,
    experimental,
    evaluate: function(ast) {
      return evaluate(ast, this, $state, experimental);
    },
    computedCache: {},
    get: function (key) {
      return this.computedCache[key].get();
    }
  };

  for (const [key, initialValue] of Object.entries(store.initialCache)) {
    store.computedCache[key] = mobx.computed(() => {
      const updatesList = store.updates.get(key) || [];
      return updatesList.reduce((acc, [operation, payload]) =>
        applyOperation(store, acc, operation, payload), store.evaluate(initialValue));
    });
  }

  store.addUpdates = mobx.action(updates => {
    for (const update of updates) {
      const [id, operation, payload] = update;
      if (id.startsWith("$state")) {
        const key = id.slice(7)
        $state[key] = (prevValue) => applyOperation(store, prevValue, operation, payload)
      } else {
        const currentUpdates = store.updates.get(id) || [];
        store.updates.set(id, [...currentUpdates, [operation, payload]]);
      }
    }
  })

  return store;
}

export const StateProvider = mobxReact.observer(
  function ({ ast, cache, experimental, model }) {
    const $state = useStateStore(ast);
    const $cache = useMemo(() => createCacheStore(cache, $state, experimental), [cache])
    const EVAL = useCallback((ast) => evaluate(ast, $cache, $state, experimental),
      [$cache, $state, experimental])
    EVAL.resolveCached = (node) => resolveCached(node, $cache)

    useEffect(() => {
      const cb = (msg) => {
        if (msg.type === 'update_cache') {
          $cache.addUpdates(JSON.parse(msg.updates))
        }
      }
      model?.on("msg:custom", cb);
      return () => model?.off("msg:custom", cb)
    }, [cache, model])

    return html`
    <${$StateContext.Provider} value=${$state}>
      <${EvaluateContext.Provider} value=${EVAL}>
        <${api.Node} value=${ast} />
      </${EvaluateContext.Provider}>
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
