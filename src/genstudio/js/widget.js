import { $StateContext, WidthContext, AUTOGRID_MIN } from "./context";
import { MarkSpec, PlotSpec, PlotWrapper, DEFAULT_PLOT_OPTIONS } from "./plot";
import { flatten, html, binding, useCellUnmounted, useElementWidth } from "./utils";
import { AnyWidgetReact, Plot, d3, MarkdownIt, React, ReactDOM } from "./imports";

const { createRender, useModelState } = AnyWidgetReact

const { useState, useEffect, useContext, useMemo, useCallback } = React

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

function firstDefined(...values) {
  return values.find(value => value !== undefined);
}

function Slider(options) {
  const { state_key, fps, label, range, init, step = 1, loop = true } = options;
  const [$state, set$state] = useContext($StateContext);
  const availableWidth = useContext(WidthContext);
  const isAnimated = typeof fps === 'number' && fps > 0;
  const [isPlaying, setIsPlaying] = useState(isAnimated);

  const [minRange, maxRange] = range[0] < range[1] ? range : [range[1], range[0]];
  const sliderValue = firstDefined($state[state_key], init, minRange);

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
    if (typeof node === 'object' && node['pyobsplot-type'] === 'function') {
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

export function evaluate(data, $state) {
  if (data === null || typeof data !== 'object') return data;
  if (Array.isArray(data)) return data.map(item => evaluate(item, $state));

  switch (data["pyobsplot-type"]) {
    case "function":
      const fn = data.name ? scope[data.module][data.name] : scope[data.module];
      if (!fn) {
        console.error('Function not found', data);
        return null;
      }
      return fn(...evaluate(data.args, $state));
    case "ref":
      return data.name ? scope[data.module][data.name] : scope[data.module];
    case "js":
      return (new Function('$state', `return ${data.value}`))($state);
    case "datetime":
      return new Date(data.value);
    default:
      return Object.fromEntries(
        Object.entries(data).map(([key, value]) => [key, evaluate(value, $state)])
      );
  }
}

function Grid({ specs: PlotSpecs, plotOptions, layoutOptions }) {
  const containerWidth = useContext(WidthContext);
  const aspectRatio = plotOptions.aspectRatio || 1;
  const minWidth = Math.min(plotOptions.minWidth || AUTOGRID_MIN, containerWidth);

  // Ensure at least one column, even if containerWidth is less than minWidth
  const numColumns = Math.max(1, Math.min(Math.floor(containerWidth / minWidth), PlotSpecs.length));
  const gap = layoutOptions.gap || "10px";
  const gapSize = parseInt(gap);
  const availableWidth = Math.max(containerWidth - (numColumns - 1) * gapSize, minWidth);
  const itemWidth = availableWidth / numColumns;
  const itemHeight = itemWidth / aspectRatio;

  const mergedPlotOptions = {
    ...DEFAULT_PLOT_OPTIONS,
    ...plotOptions,
    width: itemWidth,
    height: itemHeight
  };

  const numRows = Math.ceil(PlotSpecs.length / numColumns);
  const layoutHeight = numRows * itemHeight + (numRows - 1) * gapSize;

  const mergedLayoutOptions = {
    display: 'grid',
    gap: gap,
    gridTemplateColumns: `repeat(${numColumns}, 1fr)`,
    gridAutoRows: `${itemHeight}px`,
    height: `${layoutHeight}px`,
    width: `${containerWidth}px`, // Ensure the grid doesn't exceed container width
    overflowX: 'auto', // Allow horizontal scrolling if needed
    ...layoutOptions
  };

  return html`
    <${WidthContext.Provider} value=${itemWidth}>
      <div style=${mergedLayoutOptions}>
        ${PlotSpecs.map((item, index) => {
          if (item.spec) {
            return html`<${PlotWrapper} key=${index} spec=${{ ...item.spec, ...mergedPlotOptions }} />`;
          } else {
            return html`<div key=${index} style=${{ width: itemWidth, height: itemHeight }}>${item}</div>`;
          }
        })}
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
  const availableWidth = useContext(WidthContext);

  if (Array.isArray(value)) {
    return Hiccup(...value);
  } else if (value instanceof PlotSpec) {
    const specWithWidth = value.spec.width ? value.spec : { ...value.spec, width: availableWidth };
    return html`<${PlotWrapper} spec=${specWithWidth}/>`;
  } else if (value instanceof MarkSpec) {
    return html`<${PlotWrapper} spec=${{ marks: [value], width: availableWidth }}/>`;
  } else if (value instanceof Reactive) {
    return Slider(value.options);
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
    ? html`<${PlotWrapper} spec=${baseTag.spec}/>`
    : html`<${baseTag} ...${props}>
            ${children.map((child, index) => html`<${Node} key=${index} value=${child}/>`)}
           </>`;
}

function useStableState(initialState) {
  const [state, setState] = useState(initialState);
  return useMemo(() => [state, setState], [state]);
}

function WithState({ data }) {
  const st = useStableState(collectReactiveInitialState(data))
  const [$state] = st
  const interpretedData = useMemo(() => evaluate(data, $state), [data, $state])
  return html`
    <${$StateContext.Provider} value=${st}>
      <${Node} value=${interpretedData} />
    </>
  `
}

function AppWithData({ data }) {
  const [el, setEl] = useState();
  const width = useElementWidth(el)
  const unmounted = useCellUnmounted(el?.parentNode);

  useEffect(() => {}, [el]);

  if (!data || unmounted) {
    return null;
  }

  const adjustedWidth = width ? width - 20 : undefined; // Subtract 20px for left and right padding

  return html`
    <${WidthContext.Provider} value=${adjustedWidth}>
      <div className="genstudio-container p3" ref=${setEl}>
        ${el && html`<${WithState} data=${data}/>`}
      </div>
    </${WidthContext.Provider}>
  `;
}

function AnyWidgetApp() {
  addCSSLink(tachyons_css)
  const [data] = useModelState("data");
  const parsedData = data ? JSON.parse(data) : null

  return html`<${AppWithData} data=${parsedData} />`;
}

function HTMLApp(props) {
  return html`
    <div className="bg-white">
      <${AppWithData} ...${props} />
    </div>
  `;
}

function JSONViewer() {
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
      try {
        const jsonData = JSON.parse(e.target.result);
        setData(jsonData);
      } catch (error) {
        console.error("Error parsing JSON file:", error);
        alert("Error parsing JSON file. Please ensure it's a valid JSON.");
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
          <${AppWithData} data=${data} />
        </div>
      `}
    </div>
  `;
}

const tachyons_css = "https://cdn.jsdelivr.net/gh/tachyons-css/tachyons@6b8c744afadaf506cb12f9a539b47f9b412ed500/css/tachyons.css"

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

export const renderData = (element, data) => {
  addCSSLink(tachyons_css);
  const root = ReactDOM.createRoot(element);
  root.render(React.createElement(HTMLApp, { data }));
};


const estimateJSONSize = (jsonString) => {
  const estimatedSizeInKB = (jsonString.length / 1024).toFixed(2);
  return estimatedSizeInKB < 1000 ?
    `${estimatedSizeInKB} KB` :
    `${(estimatedSizeInKB / 1024).toFixed(2)} MB`;
};

export const renderJSON = (element, jsonString) => {
  console.log(`Loading plot data: ${estimateJSONSize(jsonString)}`);
  addCSSLink(tachyons_css);
  const root = ReactDOM.createRoot(element);
  const data = JSON.parse(jsonString);
  root.render(React.createElement(HTMLApp, { data }));
};

export const renderJSONViewer = (element) => {
  addCSSLink(tachyons_css);
  const root = ReactDOM.createRoot(element);
  root.render(React.createElement(JSONViewer));
};

export default {
  render: createRender(AnyWidgetApp),
  renderData,
  renderJSON,
  renderJSONViewer
}
