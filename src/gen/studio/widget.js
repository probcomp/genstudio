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
  return html`<div class='prose' dangerouslySetInnerHTML=${{ __html: md.render(text) }} />`;
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

    // Compute dimension domains for all marks contained in this plog

    const domains = spec.marks?.reduce((acc, mark) => {
      for (const [dimensionKey, domain] of Object.entries(mark.domains || {})) {
        acc[dimensionKey] = acc[dimensionKey]
          ? [Math.min(acc[dimensionKey][0], domain[0]), Math.max(acc[dimensionKey][1], domain[1])]
          : domain;
      }
      return acc;
    }, {}) || {};
    for (const [dimensionKey, domainInfo] of Object.entries(domains)) {
      this.spec[dimensionKey] = { ...this.spec[dimensionKey], domain: domainInfo };
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

const flat = (data, dimensions) => {
  let leaves;
  if (typeof dimensions[dimensions.length - 1] === 'object' && 'leaves' in dimensions[dimensions.length - 1]) {
    leaves = dimensions[dimensions.length - 1]['leaves'];
    dimensions = dimensions.slice(0, -1);
  }

  const _flat = (data, dim, prefix = null) => {
    if (!dim.length) {
      data = leaves ? { [leaves]: data } : data
      return prefix ? [{ ...prefix, ...data }] : [data];
    }

    const results = [];
    const dimName = dim[0].key;
    for (let i = 0; i < data.length; i++) {
      const newPrefix = prefix ? { ...prefix, [dimName]: i } : { [dimName]: i };
      results.push(..._flat(data[i], dim.slice(1), newPrefix));
    }
    return results;
  };
  return _flat(data, dimensions);
}

class MarkSpec {
  constructor(name, data, options) {
    this.fn = Plot[name];
    this.extraMarks = []
    if (!Plot[name]) {
      throw new Error(`Plot function "${name}" not found.`);
    }

    // Below is where we add functionality to Observable.Plot by preprocessing
    // the data & options that are passed in.

    // handle dimensional data passed in the 1st position
    if (data.dimensions) {
      options.dimensions = data.dimensions
      data = data.value
    }

    // flatten dimensional data
    if (options.dimensions) {
      options.dimensions = options.dimensions.map(dim => typeof dim === 'string' ? { 'key': dim } : dim);
      data = flat(data, options.dimensions)
    }

    // handle columnar data in the 1st position
    if (!Array.isArray(data) && !('length' in data)) {
      let length = null
      for (let [key, value] of Object.entries(data)) {
        options[key] = value;
        if (Array.isArray(value)) {
          length = value.length
        }
        if (length === null) {
          throw new Error("Invalid columnar data: at least one column must be an array.");
        }
        data = {length: value.length}
      }

    }

    // handle facetWrap option (grid)
    // see https://github.com/observablehq/plot/pull/892/files
    if (options.facetGrid) {
      const facetGrid = (typeof options.facetGrid === 'string') ? [options.facetGrid, {}] : options.facetGrid
      const [key, gridOpts] = facetGrid
      const keys = Array.from(d3.union(data.map((d) => d[key])))
      const index = new Map(keys.map((key, i) => [key, i]))
      const columns = gridOpts.columns || Math.floor(Math.sqrt(keys.length))

      const fx = (key) => index.get(key) % columns
      const fy = (key) => Math.floor(index.get(key) / columns)
      options.fx = (d) => fx(d[key])
      options.fy = (d) => fy(d[key])
      this.plotOptions = { fx: { axis: null }, fy: { axis: null } }
      this.extraMarks.push(Plot.text(keys, {
        fx, fy,
        frameAnchor: "top",
        dy: 4
      }))
    }

    this.name = name;
    this.data = data;
    this.options = options;
    this.plotOptions = this.plotOptions || {};
  }
}

function readMark(mark, dimensionState) {

  if (!(mark instanceof MarkSpec)) {
    return mark;
  }
  let { fn, data, options} = mark
  return [fn(data, options), ...mark.extraMarks]
}

const repeat = (data) => {
  return (_, i) => data[i % data.length];
}

const scope = {
  d3,
  Plot, React, ReactDOM,
  View: {
    PlotSpec: (x) => new PlotSpec(x),
    MarkSpec: (name, data, options) => new MarkSpec(name, data, options),
    md: (x) => renderMarkdown(x),
    repeat,
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

function SlidersView({ info, splitState, setSplitState }) {
  // `info` should be an object of {key, {label, size}}
  return html`
    <div>
      ${Object.entries(info).map(([key, { label, size }]) => html`
          <div class="flex flex-col my-2 gap-2" key=${key}>
            <label>${label || key}</label>
            ${splitState[key]}
            <input 
              type="range" 
              min="0" 
              max=${size - 1}
              value=${splitState[key] || 0} 
              onChange=${(e) => setSplitState({ ...splitState, [key]: Number(e.target.value) })}
            />
                  </div>  
      `)}
    </div>
  `;
}

function PlotView({ spec, splitState }) {
  const [parent, setParent] = useState(null)
  const ref = useCallback(setParent)
  useEffect(() => {
    if (parent) {
      const marks = spec.marks.flatMap((m) => readMark(m, splitState))
      const plotOptions = spec.marks.reduce((acc, mark) => ({ ...acc, ...mark.plotOptions }), {});
      console.log({ ...spec, ...plotOptions, marks: marks })
      const plot = Plot.plot({ ...spec, ...plotOptions, marks: marks })
      parent.appendChild(plot)
      return () => parent.removeChild(plot)
    }
  }, [spec, parent, splitState])
  return html`
    <div ref=${ref}></div>
  `
}

function PlotWrapper({ spec }) {
  const dimensionInfo = useMemo(() => {
    return spec.marks.flatMap(mark => mark.dimensions).reduce((acc, dimension) => {
      if (!dimension) {
        acc
      } else if (acc[dimension.key]) {
        acc[dimension.key] = { ...acc[dimension.key], ...dimension };
      } else {
        acc[dimension.key] = dimension;
      }
      return acc;
    }, {});
  }, []);

  const [splitState, setSplitState] = useState(
    Object.fromEntries(Object.entries(dimensionInfo).map(([k, d]) => [k, d.initial || 0]))
  );
  return html`
    <div>
      <${PlotView} spec=${spec} splitState=${splitState}></div>
      <${SlidersView} 
        info=${dimensionInfo}
        splitState=${splitState}
        setSplitState=${setSplitState}/>
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
    return html`<${PlotWrapper} spec=${value.spec}/>`;
  } else if (value instanceof MarkSpec) {
    return Node({ value: new PlotSpec({ marks: [value] }) })
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
  const scriptUrl = "https://cdn.tailwindcss.com?plugins=forms,typography,aspect-ratio";
  if (!document.getElementById(id)) {
    const script = document.createElement("script");
    script.id = id;
    script.src = scriptUrl;
    document.head.appendChild(script);
  }
  // const url = "https://cdn.jsdelivr.net/gh/html-first-labs/static-tailwind@759f1d7/dist/tailwind.css"
  // if (!document.getElementById(id)) {
  //   const link = document.createElement("link");
  //   link.id = id;
  //   link.rel = "stylesheet";
  //   link.href = url;
  //   document.head.appendChild(link);
  // }
}

export default { render, initialize: installTailwind }