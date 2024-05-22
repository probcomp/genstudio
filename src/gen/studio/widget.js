import { createRender, useModelState } from "https://esm.sh/@anywidget/react@0.0.7"
import * as React from 'https://esm.sh/react@18.3.1'
import * as ReactDOM from 'https://esm.sh/react-dom@18.3.1'
import htm from 'https://esm.sh/htm@3.1.1'
import * as Plot from "https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6.14/+esm"
import * as d3 from "https://cdn.jsdelivr.net/npm/d3@7.9.0/+esm"
import MarkdownIt from 'https://esm.sh/markdown-it@13.0.1';

const md = new MarkdownIt({
  html: true,
  linkify: true,
  typographer: true
});

function renderMarkdown(text) {
  return html`<div dangerouslySetInnerHTML=${{ __html: md.render(text) }} />`;
}

const html = htm.bind(React.createElement)


/**
 * Wrap plot specs so that our node renderer can identify them.
 */
class PlotSpec {    
  /**
   * Create a new plot spec.
   */
  constructor(spec) {
    this.spec = spec;
    const domains = spec.marks?.reduce((acc, mark) => {
      for (const [key, domain] of Object.entries(mark.domains || {})) {
        acc[key] = acc[key] 
          ? [Math.min(acc[key][0], domain[0]), Math.max(acc[key][1], domain[1])]
          : domain;
      }
      return acc;
    }, {}) || {};
    for (const [key, domain] of Object.entries(domains)) {
      this.spec[key] = {...this.spec[key], domain};
    }
  }
}

/**
 * Create a new element.
 */
const el = (tag, props, ...children) => {
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
  children = children.map((child) => html`<${Node} value=${child} />`)
  return html`<${baseTag} ...${props}>${children}</${tag}>`;
};


class MarkSpec {
  constructor(name, data, options) {
    this.fn = Plot[name]
    this.name = name
    this.data = data 
    this.options = options
    // analyze the mark to determine its shape
    if (Array.isArray(data) || 'length' in data) {
      this.format = 'array'
    } else {
      this.format = 'columnar'
      let dimensions = {}
      let domains = {}
      for (const [key, value] of Object.entries(data)) {
        if (value.dimension) {
          let dimension = value.dimension;
          dimension.length = value.value.length;
          dimension.view = dimension.view || 'slider'; // default to slider view
          dimension.fps = dimension.fps || 0;
          domains[key] = [
            Math.min(...value.value.flat()),
            Math.max(...value.value.flat())
          ];
          dimensions[value.dimension.key] = dimension;
        }
      }
      if (Object.keys(dimensions).length > 0) {this.dimensions = dimensions}
      if (Object.keys(domains).length > 0) {this.domains = domains}
    }
  }
}

function readMark(mark, dimensionState) {
  if (!(mark instanceof MarkSpec)) {
    return mark;
  }
  let {fn, data, options, format} = mark
  switch (format) {
    case 'columnar':
      // format columnar data for Observable.Plot;
      // values go into the options map.
      const formattedData = {};
      for (const [key, value] of Object.entries(data)) {
        if (value.dimension) {
          const dimension = value.dimension;
          const i = dimensionState[dimension.key] || 0;
          formattedData[key] = value.value[i];
        } else {
          formattedData[key] = value; 
        }
      }
      return fn({ length: Object.values(formattedData)[0].length }, { ...formattedData, ...options})
    default:
      return fn(data, options)
  }
}

const scope = {
  d3,
  Plot, React, ReactDOM, 
  View: {
    PlotSpec: (x) => new PlotSpec(x),
    MarkSpec: (name, data, options) => new MarkSpec(name, data, options),
    md: (x) => renderMarkdown(x),
    el
  },

}
const { useState, useEffect, useRef, useCallback, useMemo } = React

/**
 * Interpret data recursively, evaluating functions.
 */
export function interpret(data) {
  if (data === null) return null;
  if (Array.isArray(data)) return data.map(interpret);
  if (typeof data === "string" || data instanceof String) return data;
  if (Object.entries(data).length == 0) return data;

  switch (data["pyobsplot-type"]) {
    case "function":
      let fn = data.name ? scope[data.module][data.name] : scope[data.module]
      if (!fn) {
        console.error('f not found', data)
      }
      return fn.call(null, ...interpret(data["args"]));
    case "ref":
      return data.name ? scope[data.module][data.name] : scope[data.module]
    case "js":
      // Use indirect eval to avoid bundling issues
      // See https://esbuild.github.io/content-types/#direct-eval
      let indirect_eval = eval;
      return indirect_eval(data["value"]);
    case "datetime":
      return new Date(data["value"]);
  }
  // recurse into objects
  let ret = {};
  for (const [key, value] of Object.entries(data)) {
    ret[key] = interpret(value);
  }
  return ret;
}

function DimensionSliders({dimensions, dimensionState, setDimensionState}) {
  return html`
    <div>
      ${Object.entries(dimensions).map(([key, dimension]) => html`
          <div class="grid grid-cols-[min-content,1fr] my-2" key=${key}>
            <label>${dimension.label || key}</label>
            <input 
              type="range" 
              min="0" 
              max=${dimension.length - 1}
              value=${dimensionState[key] || 0} 
              onChange=${(e) => setDimensionState({...dimensionState, [key]: Number(e.target.value)})}
            />
                  </div>  
      `)}
    </div>
  `;
}

/**
 * Renders a plot.
 */
function PlotView({ spec }) {
  const [parent, setParent] = useState(null)
  const dimensions = useMemo(() => spec.marks.reduce((acc, mark) => ({...acc, ...mark.dimensions}), {}))
  const [dimensionState, setDimensionState] = useState(
    Object.fromEntries(Object.entries(dimensions).map(([k, d]) => [k, d.initial || 0]))
  );
  const ref = useCallback(setParent) 
  useEffect(() => {
    if (parent) {
      const marks = spec.marks.map((m) => readMark(m, dimensionState))
      const plot = Plot.plot({...spec, marks: marks})
      parent.appendChild(plot)
      return () => parent.removeChild(plot)
    }
  }, [spec, parent, dimensionState])
  return html`
    <div>
      <div ref=${ref}></div>
      <${DimensionSliders} 
        dimensions=${spec.marks.reduce((acc, mark) => ({...acc, ...mark.dimensions}), {})}
        dimensionState=${dimensionState}
        setDimensionState=${setDimensionState}
      />
          </div>
  `
}

/**
 * Renders a node. Arrays are parsed as hiccup.
 */
function Node({ value }) {
  if (Array.isArray(value)) {
    return el.apply(null, value);
  } else if (value instanceof PlotSpec) {
    return html`<${PlotView} spec=${value.spec}/>`;
  } else if (value instanceof MarkSpec) {
    return Node({value: new PlotSpec({marks: [value]})})
  } else {
    return value
  }
}

/**
 * The main app.
 */
function App() {
  const [data, _] = useModelState("data")
  const value = data ? interpret(JSON.parse(data)) : null
  return html`<div class="pa1"><${Node} value=${value}/></div>`
}

const render = createRender(App)

/**
 * Install Tailwind CSS.
 */
const installTailwind = () => {
  const id = "tailwind-cdn"
  const url = "https://cdn.jsdelivr.net/gh/html-first-labs/static-tailwind@759f1d7/dist/tailwind.css"
  if (!document.getElementById(id)) {
    const link = document.createElement("link");
    link.id = id;
    link.rel = "stylesheet";
    link.href = url;
    document.head.appendChild(link);
  }
}

export default { render, initialize: installTailwind }