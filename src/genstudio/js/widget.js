import { $StateContext, WidthContext, AUTOGRID_MIN as AUTOGRID_MIN_WIDTH } from "./context";
import { MarkSpec, PlotSpec, PlotWrapper, DEFAULT_PLOT_OPTIONS } from "./plot";
import { flatten, html, useCellUnmounted, useElementWidth, serializeEvent } from "./utils";
import { AnyWidgetReact, Plot, d3, MarkdownIt, React, ReactDOM } from "./imports";
const { createRender, useModelState, useExperimental } = AnyWidgetReact
const { useState, useEffect, useContext, useMemo, useCallback } = React

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

class Reactive {
  constructor(data) {
    this.options = data;
  }

  render() {
    return Slider(this.options);
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
    Slider,
    Frames,
    Reactive: (options) => new Reactive(options)
  }
}

function getFirstDefinedValue(...values) {
  return values.find(value => value !== undefined);
}

function Slider(options) {
  const { state_key, fps, label, range, init, step = 1, loop = true } = options;
  const [$state, set$state] = useContext($StateContext);
  const availableWidth = useContext(WidthContext);
  const isAnimated = typeof fps === 'number' && fps > 0;
  const [isPlaying, setIsPlaying] = useState(isAnimated);

  const [minRange, maxRange] = range[0] < range[1] ? range : [range[1], range[0]];
  const sliderValue = getFirstDefinedValue($state[state_key], init, minRange);

  useEffect(() => {
    if (isAnimated && isPlaying) {
      const intervalId = setInterval(() => {
        set$state((prevState) => {
          const newValue = prevState[state_key] + step;
          if (newValue > maxRange) {
            if (loop) {
              return { ...prevState, [state_key]: minRange };
            } else {
              setIsPlaying(false);
              return { ...prevState, [state_key]: maxRange };
            }
          }
          return { ...prevState, [state_key]: newValue };
        });
      }, 1000 / fps);
      return () => clearInterval(intervalId);
    }
  }, [isPlaying, fps, state_key, minRange, maxRange, step, loop]);

  const handleSliderChange = useCallback((value) => {
    setIsPlaying(false);
    set$state((prevState) => ({ ...prevState, [state_key]: Number(value) }));
  }, [set$state, state_key]);

  const togglePlayPause = useCallback(() => setIsPlaying((prev) => !prev), []);

  return html`
    <div className="f1 flex flex-column mv2 gap2" style=${{ width: availableWidth }}>
      <div className="flex items-center justify-between">
        <span className="flex g2">
          <label>${label}${label ? ':' : ''}</label>
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
        min=${minRange}
        max=${maxRange}
        step=${step}
        value=${sliderValue}
        onChange=${(e) => handleSliderChange(e.target.value)}
        className="w-100 outline-0"
      />
    </div>
  `;
}

const layoutComponents = new Set(['Hiccup', 'Grid', 'Row', 'Column']);

function collectReactiveInitialState(ast) {
  let initialState = {};

  function traverse(node) {
    if (!node) return;
    if (typeof node === 'object' && node['__type__'] === 'function') {
      if (node.name === 'Reactive') {
        const { state_key, init, range } = node.args[0];
        initialState[state_key] = init ?? (typeof range === 'number' ? range : range[0]);
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
        return (e) => experimental.invoke("handle_callback", {id: node.id, event: serializeEvent(e)});
      } else {
        return undefined;
      }
    default:
      return Object.fromEntries(
        Object.entries(node).map(([key, value]) => [key, evaluate(value, cache, $state, experimental)])
      );
  }
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

function Row({children, ...props}) {
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

function Column({children, ...props}) {
  return html`
    <div ...${props} className="layout-column">
      ${children}
    </div>
  `;
}


function Node({ value }) {
  if (Array.isArray(value)) {
    return Hiccup(...value);
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

function useStateWithDeps(initialStateFunction, deps) {
  // useState, recomputes initial state when deps change
  const [state, setState] = useState(null);
  useEffect(() => {
    setState(initialStateFunction);
  }, deps);

  return useMemo(() => [state, setState], [state]);
}

function StateProvider({ ast, cache, experimental }) {
  const stateArray = useStateWithDeps(() => collectReactiveInitialState(ast), [ast]);
  const [$state] = stateArray;

  const [data, setData] = useState();
  useEffect(() => {
    if ($state) {
      setData(evaluate(ast, evaluate(cache, {}, $state, experimental), $state, experimental))
    }
  }, [ast, cache, $state, experimental]);

  if (!data) return;
  return html`
    <${$StateContext.Provider} value=${stateArray}>
      <${Node} value=${data} />
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
        return {...data, size: estimateJSONSize(jsonString), ...JSON.parse(jsonString)};
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
  return html`<${Viewer} jsonString=${jsonString} experimental=${experimental} />`;
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
