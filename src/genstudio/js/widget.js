import { $StateContext, WidthContext, AUTOGRID_MIN as AUTOGRID_MIN_WIDTH } from "./context";
import { MarkSpec, PlotSpec, PlotWrapper, DEFAULT_PLOT_OPTIONS } from "./plot";
import { flatten, html, useCellUnmounted, useElementWidth, serializeEvent } from "./utils";
import { AnyWidgetReact, Plot, d3, MarkdownIt, React, ReactDOM } from "./imports";
const { createRender, useModelState, useModel, useExperimental } = AnyWidgetReact
const { useState, useEffect, useContext, useMemo, useCallback } = React
import bylight from "bylight";

const TACHYONS_CSS_URL = "https://cdn.jsdelivr.net/gh/tachyons-css/tachyons@6b8c744afadaf506cb12f9a539b47f9b412ed500/css/tachyons.css"
const DEFAULT_GRID_GAP = "10px"
export const CONTAINER_PADDING = 10;

const md = new MarkdownIt({
  html: true,
  linkify: true,
  typographer: true
});

function renderMarkdown(text) {
  return html`<div className='prose' dangerouslySetInnerHTML=${{ __html: md.render(text) }} />`;
}
function ReactiveSlider(options) {
  let { state_key, fps, label, step = 1, loop = true, tail, rangeMin, rangeMax } = options;
  const [$state, set$state] = useContext($StateContext);
  const availableWidth = useContext(WidthContext);
  const isAnimated = typeof fps === 'number' && fps > 0;
  const [isPlaying, setIsPlaying] = useState(isAnimated);

  const sliderValue = clamp($state[state_key] ?? rangeMin, rangeMin, rangeMax);

  useEffect(() => {
    if (isAnimated && isPlaying) {
      const intervalId = setInterval(() => {
        $state[state_key] = (prevValue) => {
          const nextValue = prevValue + step;
          if (nextValue > rangeMax) {
            if (tail) {
              return rangeMax;
            } else if (loop) {
              return rangeMin;
            } else {
              setIsPlaying(false);
              return rangeMax;
            }
          }
          return nextValue;
        };
      }, 1000 / fps);
      return () => clearInterval(intervalId);
    }
  }, [isPlaying, fps, state_key, rangeMin, rangeMax, step, loop, tail]);

  const handleSliderChange = useCallback((value) => {
    setIsPlaying(false);
    $state[state_key] = Number(value);
  }, [set$state, state_key]);

  const togglePlayPause = useCallback(() => setIsPlaying((prev) => !prev), []);
  if (options.kind !== 'Slider') return;
  return html`
    <div className="f1 flex flex-column mv2 gap2" style=${{ width: availableWidth }}>
      <div className="flex items-center justify-between">
        <span className="flex g2">
          <label>${label}</label>
          <span>${$state[state_key]}</span>
        </span>
        ${isAnimated && html`
          <div onClick=${togglePlayPause} className="pointer">
            ${isPlaying ? pauseIcon : playIcon}
          </div>
        `}
      </div>
      <input
        type="range"
        min=${rangeMin}
        max=${rangeMax}
        step=${step}
        value=${sliderValue}
        onChange=${(e) => handleSliderChange(e.target.value)}
        className="w-100 outline-0"
      />
    </div>
  `;
}

function clamp(value, min, max) {
  if (value < min) return min;
  if (value > max) return max;
  return value;
}

class Reactive {
  constructor(data) {
    let { init, range, rangeFrom, tail, step } = data;

    let rangeMin, rangeMax;
    if (rangeFrom) {
      // determine range dynamically based on last index of rangeFrom
      rangeMin = 0;
      rangeMax = rangeFrom.length - 1;
    } else if (typeof range === 'number') {
      // range may be specified as the upper end of a [0, n] integer range.
      rangeMin = 0;
      rangeMax = range;
    } else {
      [rangeMin, rangeMax] = range;
    }
    init = init || rangeMin;
    step = step || 1;

    // Assert that init is set
    if (init === undefined) {
      throw new Error("Reactive: 'init', 'rangeFrom' or 'range' must be defined");
    }
    this.options = {
      ...data,
      rangeMin,
      rangeMax,
      tail,
      init,
      step
    };
  }

  render() {
    return ReactiveSlider(this.options);
  }
}

const playIcon = html`<svg viewBox="0 0 24 24" width="24" height="24"><path fill="currentColor" d="M8 5v14l11-7z"></path></svg>`;
const pauseIcon = html`<svg viewBox="0 24 24" width="24" height="24"><path fill="currentColor" d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"></path></svg>`;

function Frames(props) {
  const { state_key, frames } = props
  const [$state] = useContext($StateContext);
  if (!Array.isArray(frames)) {
    return html`<div className="red">Error: 'frames' must be an array.</div>`;
  }

  const index = $state[state_key];
  if (!Number.isInteger(index) || index < 0 || index >= frames.length) {
    return html`<div className="red">Error: Invalid index. $state[${state_key}] (${index}) must be a valid index of the frames array (length: ${frames.length}).</div>`;
  }

  return html`<${Node} value=${frames[index]} />`;
}

class Bylight {
  constructor({ patterns, source, ...props }) {
    this.patterns = patterns;
    this.source = source;
    this.props = props;
  }

  render() {
    const preRef = React.useRef(null);

    React.useEffect(() => {
      if (preRef.current && this.patterns) {
        bylight.highlight(preRef.current, this.patterns);
      }
    }, [this.source, this.patterns]);

    return React.createElement('pre', {
      ref: preRef,
      className: this.props.className
    }, this.source);
  }
}

const scope = {
  d3,
  Plot, React, ReactDOM,
  View: {
    PlotSpec: (x) => new PlotSpec(x),
    MarkSpec: (name, data, options) => new MarkSpec(name, data, options),
    md: renderMarkdown,
    repeat: (data) => (_, i) => data[i % data.length],
    Hiccup,
    Grid,
    Row,
    Column,
    flatten,
    Frames,
    Reactive: (options) => new Reactive(options),
    Bylight: (source, patterns, props) => new Bylight({ source, patterns, ...(props || {}) })
  }
}


const layoutComponents = new Set(['Hiccup', 'Grid', 'Row', 'Column']);

function collectReactiveInitialState(ast) {
  let initialState = {};

  function traverse(node) {
    if (!node) return;
    if (typeof node === 'object' && node['__type__'] === 'function') {
      if (node.name === 'Reactive') {
        const {state_key, init} = new Reactive(node.args[0], initialState).options;
        initialState[state_key] = init;
      } else if (layoutComponents.has(node.name)) {
        node.args.forEach(traverse);
      }
    } else if (Array.isArray(node)) {
      node.forEach(traverse);
    }
  }

  traverse(ast);
  return initialState;
}

export function evaluate(node, cache, $state, experimental) {
  if (node === null || typeof node !== 'object') return node;
  if (Array.isArray(node)) return node.map(item => evaluate(item, cache, $state, experimental));

  switch (node["__type__"]) {
    case "function":
      const fn = node.name ? scope[node.module][node.name] : scope[node.module];
      if (!fn) {
        console.error('Function not found', node);
        return null;
      }
      return fn(...evaluate(node.args, cache, $state, experimental));
    case "ref":
      return node.name ? scope[node.module][node.name] : scope[node.module];
    case "js":
      return (new Function('$state', 'd3', 'Plot', `return ${node.value}`))($state, d3, Plot);
    case "datetime":
      return new Date(node.value);
    case "cached":
      return cache[node.id];
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


function evaluateCache(cache, $state, experimental) {
  const evaluatedCache = {};
  const evaluating = new Set();

  function evaluateCacheEntry(key) {
    if (key in evaluatedCache) return evaluatedCache[key];
    if (evaluating.has(key)) {
      console.warn(`Circular reference detected in cache for key: ${key}`);
      return undefined;
    }

    evaluating.add(key);
    const result = evaluate(cache[key], evaluatedCache, $state, experimental);
    evaluating.delete(key);
    evaluatedCache[key] = result;
    return result;
  }

  for (const key in cache) {
    evaluateCacheEntry(key);
  }

  return evaluatedCache;
}

function Grid({ children, style, minWidth = AUTOGRID_MIN_WIDTH, gap = DEFAULT_GRID_GAP, aspectRatio = 1 }) {
  const availableWidth = useContext(WidthContext);
  const effectiveMinWidth = Math.min(minWidth, availableWidth);
  const gapSize = parseInt(gap);

  const numColumns = Math.max(1, Math.min(Math.floor(availableWidth / effectiveMinWidth), children.length));
  const itemWidth = (availableWidth - (numColumns - 1) * gapSize) / numColumns;
  const itemHeight = itemWidth / aspectRatio;
  const numRows = Math.ceil(children.length / numColumns);
  const layoutHeight = numRows * itemHeight + (numRows - 1) * gapSize;

  const containerStyle = {
    display: 'grid',
    gap,
    gridTemplateColumns: `repeat(${numColumns}, 1fr)`,
    gridAutoRows: `${itemHeight}px`,
    height: `${layoutHeight}px`,
    width: `${availableWidth}px`,
    overflowX: 'auto',
    ...style
  };

  return html`
    <${WidthContext.Provider} value=${itemWidth}>
      <div style=${containerStyle}>
        ${children.map((value, index) => html`<${Node} key=${index}
                                                       style=${{ width: itemWidth }}
                                                       value=${value}/>`)}
      </div>
    </>
  `;
}

function Row({ children, ...props }) {
  const availableWidth = useContext(WidthContext);
  const childCount = React.Children.count(children);
  const childWidth = availableWidth / childCount;

  return html`
    <div ...${props} className="layout-row">
      <${WidthContext.Provider} value=${childWidth}>
        ${React.Children.map(children, (child, index) => html`
          <div className="row-item" key=${index}>
            ${child}
          </div>
        `)}
      </${WidthContext.Provider}>
    </div>
  `;
}

function Column({ children, ...props }) {
  return html`
    <div ...${props} className="layout-column">
      ${children}
    </div>
  `;
}


function Node({ value }) {
  if (Array.isArray(value)) {
    return (['string', 'function'].includes(typeof value[0])) ? Hiccup(...value) : Hiccup("div", ...value);
  } else if (typeof value === 'object' && value !== null && 'render' in value) {
    return value.render();
  } else {
    return value;
  }
}

function Hiccup(tag, props, ...children) {
  if (props?.constructor !== Object) {
    children.unshift(props);
    props = {};
  }

  let baseTag = tag;
  if (typeof tag === 'string') {
    let id, classes
    [baseTag, ...classes] = tag.split('.');
    [baseTag, id] = baseTag.split('#');

    if (id) { props.id = id; }

    if (classes.length > 0) {
      props.className = `${props.className || ''} ${classes.join(' ')}`.trim();
    }
  }

  return baseTag instanceof PlotSpec
    ? baseTag.render()
    : children.length > 0
      ? html`<${baseTag} ...${props}>
          ${children.map((child, index) => html`<${Node} key=${index} value=${child}/>`)}
        </>`
      : html`<${baseTag} ...${props} />`;
}

function useReactiveState(ast) {
  const initialState = useMemo(() => collectReactiveInitialState(ast), [ast]);
  const [state, setState] = useState(initialState);
  const initialStateKeys = useMemo(() => Object.keys(initialState).sort().join(','), [initialState]);
  useEffect(() => setState(initialState), [initialStateKeys]);

  const stateProxy = new Proxy(state, {
    set(_, prop, value) {
      setState(prevState => ({
        ...prevState,
        [prop]: typeof value === 'function' ? value(prevState[prop]) : value
      }));
      return true;
    }
  });

  return useMemo(() => [stateProxy, setState], [state]);
}

function StateProvider({ ast, cache, experimental, model }) {
  const stateArray = useReactiveState(ast);
  const [$state] = stateArray;

  const [evaluatedAst, setEvaluatedAst] = useState();
  const evaluateData = () => {
    const evaluatedCache = evaluateCache(cache, $state, experimental)
    const evaluatedAst = evaluate(ast, evaluatedCache, $state, experimental)
    setEvaluatedAst(evaluatedAst)
  }
  useEffect(evaluateData, [ast, cache, $state, experimental]);

  useEffect(() => {
    const cb = (msg) => {
      if (msg.type === 'update_cache') {
        const updates = evaluate(JSON.parse(msg.updates), cache, $state, experimental);
        for (const [id, operation, payload] of updates) {
          const prevValue = cache[id];
          let nextValue;
          switch (operation) {
            case "append":
              nextValue = [...prevValue, payload];
              break;
            case "concat":
              nextValue = [...prevValue, ...payload];
              break;
            case "reset":
              nextValue = payload;
              break;
          }
          cache[id] = nextValue
        }
        evaluateData()
      }
    }
    model.on("msg:custom", cb);
    return () => model.off("msg:custom", cb)
  }, [cache, model])

  if (!evaluatedAst) return;
  return html`
    <${$StateContext.Provider} value=${stateArray}>
      <${Node} value=${evaluatedAst} />
    </>
  `;
}

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
      ${data.size && data.dev && html`<div className="f1 p3">${data.size}</div>`}
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
    <div className="pa3">
      <div
        className=${`ba b--dashed br3 pa5 tc ${dragActive ? 'b--blue' : 'b--black-20'}`}
        onDragEnter=${handleDrag}
        onDragLeave=${handleDrag}
        onDragOver=${handleDrag}
        onDrop=${handleDrop}
      >
        <label htmlFor="file-upload" className="f5 link dim br-pill ph3 pv2 mb2 dib white bg-dark-blue pointer">
          Choose a JSON file
        </label>
        <input
          type="file"
          id="file-upload"
          accept=".json"
          onChange=${handleChange}
          style=${{ display: 'none' }}
        />
        <p className="f6 black-60">or drag and drop a JSON file here</p>
      </div>
      ${data && html`
        <div className="mt4">
          <h2 className="f4 mb3">Loaded JSON Data:</h2>
          <${Viewer} ...${data} />
        </div>
      `}
    </div>
  `;
}

function addCSSLink(url) {
  const linkId = 'tachyons-css';
  if (!document.getElementById(linkId)) {
    const link = document.createElement('link');
    link.id = linkId;
    link.rel = 'stylesheet';
    link.href = url;
    document.head.appendChild(link);
  }
}

function AnyWidgetApp() {
  addCSSLink(TACHYONS_CSS_URL)
  let [jsonString] = useModelState("data");
  const experimental = useExperimental();
  const model = useModel();
  return html`<${Viewer} jsonString=${jsonString} experimental=${experimental}, model=${model} />`;
}

export const renderData = (element, data) => {
  addCSSLink(TACHYONS_CSS_URL);
  const root = ReactDOM.createRoot(element);
  if (typeof data === 'string') {
    root.render(html`<${Viewer} jsonString=${data} />`);
  } else {
    root.render(html`<${Viewer} ...${data} />`);
  }
};

export const renderFile = (element) => {
  addCSSLink(TACHYONS_CSS_URL);
  const root = ReactDOM.createRoot(element);
  root.render(html`<${FileViewer} />`);
};

export default {
  render: createRender(AnyWidgetApp),
  renderData,
  renderFile
}
