import { createRender, useModelState } from "@anywidget/react";
import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import MarkdownIt from "markdown-it";
import * as React from "react";
import * as ReactDOM from "react-dom";
import { WidthContext, AUTOGRID_MIN } from "./context";
import { MarkSpec, PlotSpec, PlotWrapper, DEFAULT_PLOT_OPTIONS } from "./plot";
import { flatten, html, binding, useCellUnmounted, useElementWidth } from "./utils";
import { $StateContext } from "./context";
const { useState, useEffect, useContext, useMemo } = React

const md = new MarkdownIt({
  html: true,
  linkify: true,
  typographer: true
});

function renderMarkdown(text) {
  return html`<div class='prose' dangerouslySetInnerHTML=${{ __html: md.render(text) }} />`;
}

class Reactive {
  constructor(data) {
    this.options = data;
  }
}

const playIcon = html`<svg viewBox="0 0 24 24" width="24" height="24"><path fill="currentColor" d="M8 5v14l11-7z"></path></svg>`;
const pauseIcon = html`<svg viewBox="0 24 24" width="24" height="24"><path fill="currentColor" d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"></path></svg>`;

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
    flatten,
    Slider,
    Reactive: (options) => new Reactive(options)
  }
}

function Slider({ name, fps, label, range, init, step = 1 }) {
  const [$state, set$state] = useContext($StateContext)
  const isAnimated = typeof fps === 'number' && fps > 0
  const [isPlaying, setIsPlaying] = useState(isAnimated);

  useEffect(() => {
    if (isAnimated && isPlaying) {
      const intervalId = setInterval(() => {
        set$state((prevState) => {
          const newValue = prevState[name] + step;
          return {
            ...prevState,
            [name]: newValue < range[1] ? newValue : range[0]
          };
        });
      }, 1000 / fps);
      return () => clearInterval(intervalId);
    }
  }, [isPlaying, fps, name, range, step]);

  const handleSliderChange = (value) => {
    setIsPlaying(false);
    set$state((prevState) => ({ ...prevState, [name]: Number(value) }));
  };

  const togglePlayPause = () => setIsPlaying(!isPlaying);

  const animationControl = isAnimated &&
    html`<div onClick=${togglePlayPause} style=${{ cursor: 'pointer' }}>
      ${isPlaying ? pauseIcon : playIcon}
    </div>`;

  return html`
    <div style=${{ fontSize: "14px", display: 'flex', flexDirection: 'column', margin: '0.5rem 0', gap: '0.5rem' }}>
      <div style=${{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <label>${label}</label>
        <span>${$state[name]}</span>
        ${animationControl}
      </div>
      <input
        type="range"
        min=${range[0]}
        max=${range[1]}
        step=${step}
        value=${$state[name] || init || range[0]}
        onChange=${(e) => handleSliderChange(e.target.value)}
        style=${{ outline: 'none' }}
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
        const { name, init, range } = node.args[0];
        initialState[name] = init ?? (typeof range === 'number' ? range : range[0]);
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

  const numColumns = Math.max(Math.min(Math.floor(containerWidth / minWidth), PlotSpecs.length), 1);
  const itemWidth = containerWidth / numColumns;
  const itemHeight = itemWidth / aspectRatio;

  const mergedPlotOptions = {
    ...DEFAULT_PLOT_OPTIONS,
    width: itemWidth,
    height: itemHeight,
    ...plotOptions
  };

  const numRows = Math.ceil(PlotSpecs.length / numColumns);
  const layoutHeight = numRows * itemHeight;

  const mergedLayoutOptions = {
    display: 'grid',
    gridTemplateColumns: `repeat(${numColumns}, 1fr)`,
    gridAutoRows: `${itemHeight}px`,
    height: `${layoutHeight}px`,
    ...layoutOptions
  };

  return html`
    <div style=${mergedLayoutOptions}>
      ${PlotSpecs.map((item, index) => {
    if (item.spec) {
      return html`<${PlotWrapper} key=${index} spec=${{ ...item.spec, ...mergedPlotOptions }} />`;
    } else {
      return html`<div key=${index} style=${{ width: itemWidth, height: itemHeight }}>${item}</div>`;
    }
  })}
    </div>
  `;
}

function Node({ value }) {
  if (Array.isArray(value)) {
    return Hiccup(...value);
  } else if (value instanceof PlotSpec) {
    return html`<${PlotWrapper} spec=${value.spec}/>`;
  } else if (value instanceof MarkSpec) {
    return html`<${PlotWrapper} spec=${{ marks: [value] }}/>`;
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

function App() {
  const [el, setEl] = useState();
  const [data] = useModelState("data");
  const parsedData = data ? JSON.parse(data) : null
  const width = useElementWidth(el)
  const unmounted = useCellUnmounted(el?.parentNode);

  if (!parsedData || unmounted) return null;

  return html`
    <${WidthContext.Provider} value=${width}>
      <div style=${{ color: '#333' }} ref=${setEl}>
        ${el && html`<${WithState} data=${parsedData}/>`}
      </div>
    </${WidthContext.Provider}>
  `;
}

export default { render: createRender(App) }
