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
    md: (x) => renderMarkdown(x),
    repeat: (data) => (_, i) => data[i % data.length],
    Hiccup,
    AutoGrid,
    flatten,
    Slider,
    Reactive: (options) => new Reactive(options)
  }
}

function Slider({ name, fps, label, range, init }) {
  const [$state, set$state] = React.useContext($StateContext)
  const isAnimated = typeof fps === 'number' && fps > 0
  const [isPlaying, setIsPlaying] = React.useState(isAnimated);

  React.useEffect(() => {
    if (isAnimated) {
      let intervalId;
      if (isPlaying && isAnimated && fps > 0) {
        intervalId = setInterval(() => {
          set$state((prevState) => {
            const nextValue = prevState[name] + 1;
            if (nextValue < range[1]) {
              return { ...prevState, [name]: nextValue };
            } else {
              return { ...prevState, [name]: range[0] };
            }
          });
        }, 1000 / fps);
      }
      return () => clearInterval(intervalId);
    }
  }, [isPlaying, fps, name, range]);

  const handleSliderChange = (value) => {
    setIsPlaying(false);
    set$state((prevState) => {
      return { ...prevState, [name]: Number(value) };
    });
  };

  const togglePlayPause = () => {
    setIsPlaying(!isPlaying);
  };
  const animationControl = isAnimated ? html`<div onClick=${togglePlayPause} style=${{ cursor: 'pointer' }}>${isPlaying ? pauseIcon : playIcon}</div>` : null;

  return html`
      <div style=${{ fontSize: "14px", display: 'flex', flexDirection: 'column', marginTop: '0.5rem', marginBottom: '0.5rem', gap: '0.5rem' }}>
          <div style=${{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <label>${label}</label>
              <span>${$state[name]}</span>
              ${animationControl}

          </div>
          <input
              type="range"
              min=${range[0]}
              max=${range[1] - 1}
              value=${$state[name] || init || 0}
              onChange=${(e) => handleSliderChange(e.target.value)}
              style=${{ outline: 'none' }}
          />
      </div>
  `;
}

const layoutComponents = new Set(['Hiccup', 'Grid', 'Row', 'Column']);

// Function to collect reactive variables from the AST
function collectReactiveInitialState(ast) {
  let initialState = {};

  function traverse(node) {
    if (!node) return;
    if (typeof node === 'object' && node['pyobsplot-type'] === 'function') {
      if (node.name === 'Reactive') {
        const options = node.args[0]
        const key = options.name
        initialState[key] = options.init || typeof options.range == 'number' ? options.range : options.range[0]
      } else if (layoutComponents.has(node.name)) {
        // Traverse arguments of layout components
        node.args.forEach(traverse);
      }
    } else if (Array.isArray(node)) {
      // Traverse array elements
      node.forEach(traverse);
    }
  }

  traverse(ast);
  return initialState;
}

/**
 * Interpret data recursively, evaluating functions.
 */
export function evaluate(data, $state) {
  if (data === null) return null;
  if (Array.isArray(data)) return data.map(item => evaluate(item, $state));
  if (typeof data === "string" || data instanceof String) return data;
  if (data.constructor !== Object) return data;

  switch (data["pyobsplot-type"]) {
    case "function":
      let fn = data.name ? scope[data.module][data.name] : scope[data.module];
      if (!fn) {
        console.error('f not found', data);
      }
      const interpretedArgs = evaluate(data.args, $state);
      // Inject $state into layout components
      if (layoutComponents.has(data.name)) {
        return fn.call(null, ...interpretedArgs, { $state });
      }
      return fn.call(null, ...interpretedArgs);
    case "ref":
      return data.name ? scope[data.module][data.name] : scope[data.module];
    case "js":
      // Use indirect eval to avoid bundling issues
      let indirect_eval = eval;
      // Allow access to $state in js expressions
      return indirect_eval(`(($state) => { return ${data["value"]} })`)($state);
    case "datetime":
      return new Date(data["value"]);
  }

  // recurse into objects
  let ret = {};
  for (const [key, value] of Object.entries(data)) {
    ret[key] = evaluate(value, $state);
  }
  return ret;
}

function AutoGrid({ specs: PlotSpecs, plotOptions, layoutOptions }) {
  // normalizeDomains(PlotSpecs)
  const containerWidth = useContext(WidthContext);
  const aspectRatio = plotOptions.aspectRatio || 1;
  const minWidth = Math.min(plotOptions.minWidth || AUTOGRID_MIN, containerWidth);

  // Compute the number of columns based on containerWidth and minWidth, no more than the number of specs
  const numColumns = Math.max(Math.min(Math.floor(containerWidth / minWidth), PlotSpecs.length), 1);
  const itemWidth = containerWidth / numColumns;
  const itemHeight = itemWidth / aspectRatio;

  // Merge width and height into defaultPlotOptions
  const mergedPlotOptions = {
    ...DEFAULT_PLOT_OPTIONS,
    width: itemWidth,
    height: itemHeight,
    ...plotOptions
  };

  // Compute the total layout height
  const numRows = Math.ceil(PlotSpecs.length / numColumns);
  const layoutHeight = numRows * itemHeight;

  // Update defaultLayoutOptions with the computed layout height
  const defaultLayoutOptions = {
    display: 'grid',
    gridTemplateColumns: `repeat(${numColumns}, 1fr)`,
    gridAutoRows: `${itemHeight}px`,
    height: `${layoutHeight}px`
  };

  // Merge default options with provided options
  const mergedLayoutOptions = { ...defaultLayoutOptions, ...layoutOptions };

  return html`
    <div style=${mergedLayoutOptions}>
      ${PlotSpecs.map(({ spec }, index) => {
    return html`<${PlotWrapper} key=${index} spec=${{ ...spec, ...mergedPlotOptions }} />`;
  })}
    </div>
  `;
}

/**
 * Renders a node. Arrays are parsed as hiccup.
 */
function Node({ value }) {
  if (Array.isArray(value)) {
    return Hiccup.apply(null, value);
  } else if (value instanceof PlotSpec) {
    return html`<${PlotWrapper} spec=${value.spec}/>`;
  } else if (value instanceof MarkSpec) {
    return Node({ value: new PlotSpec({ marks: [value] }) })
  } else if (value instanceof Reactive) {
    return Slider(value.options)
  } else {
    return value
  }
}

function Hiccup(tag, props, ...children) {
  if (props.constructor !== Object) {
    children.unshift(props);
    props = {};
  }
  let baseTag
  if (typeof tag === 'string') {
    let id, classes
    [baseTag, ...classes] = tag.split('.');
    [baseTag, id] = baseTag.split('#');

    if (id) { props.id = id; }

    if (classes.length > 0) {
      props.className = `${props.className || ''} ${classes.join(' ')}`.trim();
    }

  } else {
    baseTag = tag
  }
  children = children.map((child, i) => html`<${Node} key=${i} value=${child} />`)

  return baseTag instanceof PlotSpec ?
    html`<${PlotWrapper} spec=${baseTag.spec}/>` :
    html`<${baseTag} ...${props}>${children}</${tag}>`;
};

function useStableState(initialState) {
  // Return a useState whose array identity only changes
  // when the state has changed
  const [state, setState] = useState(initialState);
  return useMemo(() => [state, setState], [state, setState]);
}

function WithState({ data }) {
  const st = useStableState(collectReactiveInitialState(data))
  const [$state] = st
  const interpretedData = useMemo(() => evaluate(data, $state), [data, $state])
  return html`
    <${$StateContext.Provider} value=${st}>
      <${Node} value=${interpretedData} />
    </${$StateContext.Provider}>
  `
}

function App() {
  const [el, setEl] = useState();
  const [data, _] = useModelState("data");
  const parsedData = data ? JSON.parse(data) : null
  const width = useElementWidth(el)
  const unmounted = useCellUnmounted(el?.parentNode);
  if (!parsedData || unmounted) {
    return null;
  }
  return html`<${WidthContext.Provider} value=${width}>
      <div style=${{ color: '#333' }} ref=${setEl}>
        ${el ? html`<${WithState} data=${parsedData}/>` : null}
      </div>
    </>`
}

export default { render: createRender(App) }
