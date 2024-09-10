import { WidthContext, CONTAINER_PADDING, $StateContext } from "./context";
import { html, useCellUnmounted, useElementWidth, serializeEvent } from "./utils";
import { AnyWidgetReact, React, ReactDOM, Plot, d3, mobx, mobxReact } from "./imports";
const { createRender, useModelState, useModel, useExperimental } = AnyWidgetReact;
const { useState, useMemo, useCallback, useEffect } = React;
import * as api from "./api";

function resolveReference(path, obj) {
  return path.split('.').reduce((acc, key) => acc[key], obj);
}

function resolveCached(node, $state) {
  if (node && typeof node === 'object' && node["__type__"] === "cached") {
    return resolveCached($state.cached(node.id), $state);
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
    case "ref":
      return resolveReference(node.path, api);
    case "js":
      return (new Function('$state', 'd3', 'Plot', `return ${node.value}`))($state, d3, Plot);
    case "datetime":
      return new Date(node.value);
    case "cached":
      return $state.cached(node.id);
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
    default:
      throw new Error(`Unknown operation: ${op}`);
  }
}

export function createStateStore(initialValues, experimental) {
  // An AST is paired with two kinds of dynamic state: $state variables
  // and cache entries. $state variables are primitive (non-evaluated) values
  // which cannot have dependencies. Cache entries are AST fragments which can
  // depend on other cache entries as well as $state.

  // $state entries will be represented as `mobx.observable.box` instances,
  // created lazily when accessed for the first time.
  const stateEntries = {}

  // cache entries will be represented as `mobx.computed` instances, created lazily,
  // along with a list of updates to be applied to the initial value.
  const cacheEntries = {}
  const cacheUpdates = mobx.observable.map({})

  const stateBox = function(key) {
    // return a mobx "box" for each $state entry
    if (!(key in stateEntries)) {
      stateEntries[key] = mobx.observable.box(initialValues[key], {deep: false});
    }
    return stateEntries[key];
  }

  const $state = {
    evaluate: function(ast) {
      return evaluate(ast, this, experimental);
    },
    backfill: function(cache) {
      // adds state/cache entries to the initial state
      Object.assign(initialValues, cache)
    },
    resolveCached: function (node) {
      return resolveCached(node, this)
    },
    cached: function (key) {
      if (key.startsWith('$state.')) {
        // $state is initially populated from cache entries, identified only by a
        // `$state` prefix in their id.
        return stateBox(key).get()
      }
      if (!(key in cacheEntries)) {
        cacheEntries[key] = mobx.computed(() => {
          const updatesList = cacheUpdates.get(key) || [];
          return updatesList.reduce((acc, [operation, payload]) =>
            applyOperation(this, acc, operation, payload), this.evaluate(initialValues[key]));
        });
      }
      return cacheEntries[key].get();
    }
  };

  $state.addUpdates = mobx.action(updates => {
    // Cache entries are computed by taking an initial value and applying each of the updates which have
    // occurred. This is assumed to be adequately performant as the updates are typically simple/cheap.
    for (const update of updates) {
      const [id, operation, payload] = update;
      if (id.startsWith("$state.")) {
        const box = stateBox(id)
        box.set(applyOperation($state, box.get(), operation, payload))
      } else {
        const currentUpdates = cacheUpdates.get(id) || [];
        cacheUpdates.set(id, [...currentUpdates, [operation, payload]]);
      }
    }
  })

  // We return a proxy object for interacting directly with $state variables
  return new Proxy($state, {
    get: (_, key) => {
      return key in $state ? $state[key] : stateBox('$state.' + key).get()
    },
    set: mobx.action((_, key, value) => {
      const box = stateBox('$state.' + key);
      box.set(typeof value === 'function' ? value(box.get()) : value)
      return true;
    })
  });
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
      // listen for `update_cache` events.
      if (model) {
        const cb = (msg) => {
          if (msg.type === 'update_cache') {
            $state.addUpdates(JSON.parse(msg.updates))
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
