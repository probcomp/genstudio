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

const flattenDimensions = (data, dimensions, leaf) => {
  const _flat = (data, dimensions, prefix = null) => {
    if (!dimensions.length) {
      data = leaf?.as ? {[leaf.as]: data} : data
      return prefix ? [{ ...prefix, ...data }] : [data];
    }

    const results = [];
    const dKey = dimensions[0].key;
    for (let i = 0; i < data.length; i++) {
      const newPrefix = prefix ? { ...prefix, [dKey]: i } : { [dKey]: i };
      results.push(..._flat(data[i], dimensions.slice(1), newPrefix));
    }
    return results;
  };
  const flattened = _flat(data, dimensions);
  return flattened;
}

const flat = (data, dimensions) => {
  let leaf;
  if (typeof dimensions[dimensions.length - 1] === 'object' && 'as' in dimensions[dimensions.length - 1]) {
    leaf = dimensions[dimensions.length - 1].as;
    dimensions = dimensions.slice(0, -1);
  }
  const _flat = (data, dim, prefix = null) => {
    if (!dim.length) {
      data = leaf ? {[leaf]: data} : data
      return prefix ? [{ ...prefix, ...data }] : [data];
    }

    const results = [];
    const dimName = dim[0];
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
    if (!Plot[name]) {
      throw new Error(`Plot function "${name}" not found.`);
    }
    this.name = name;
    this.data = data;
    this.options = options;
    
    this.format = Array.isArray(data) || 'length' in data ? 'array' : 'columnar';
    
    if (options.dimensions) {
      this.data = flat(this.data, options.dimensions)
    }
    // if (this.format === 'columnar') {
    //   const { dimensions, domains } = this.analyzeColumnarData(data);
    //   console.log("Dimensions", data, dimensions)
    //   // What about the case where we pass "just" the dimensioned data,
    //   // because it is to be expanded? 
    //   if (dimensions?.length > 0) {
    //     this.dimensions = dimensions;
    //     // this.data = flat(data, dimensions)
    //     // this.format = 'array'
    //     // console.log("D", data )
    //     // console.log("F", this.data)
    //   }
    //   if (Object.keys(domains).length > 0) this.domains = domains;
    // }
  }

  // analyzeColumnarData(data) {
  //   let dimensions = [];
  //   let domains = {};
  //   for (const [key, value] of Object.entries(data)) {
  //     if (value.dimensions) {
  //       let dimensionValue = value.value;
  //       value.dimensions.forEach((dimension) => {
  //         dimension.size = dimensionValue?.length || 0;
  //         dimension.view = dimension.view || 'slider'; // default to slider view
  //         dimension.fps = dimension.fps || 0;
  //         domains[key] = dimension.domain;
  //         if (dimensionValue && typeof dimensionValue[0] === 'number') {
  //           let [min, max] = this.calculateDomain(value.value);
  //           dimension.domain = [min, max];
  //           domains[key] = dimension.domain;
  //         }
  //         dimensions.push(dimension);
  //         dimensionValue = dimensionValue && dimensionValue[0];
  //       });
  //     }
  //   }
  //   return { dimensions, domains };
  // }

  calculateDomain(values) {
    let min = Infinity;
    let max = -Infinity;
    for (const subArray of values) {
      for (const num of subArray) {
        if (num < min) min = num;
        if (num > max) max = num;
      }
    }
    return [min, max];
  }
}

function readMark(mark, dimensionState) {
  if (!(mark instanceof MarkSpec)) {
    return mark;
  }
  let { fn, data, options, format } = mark
  switch (format) {
    case 'columnar':
      // format columnar data for Observable.Plot;
      // values go into the options map.
      const formattedData = {};
      for (let [key, value] of Object.entries(data)) {
        if (value.dimensions) {
          const dimensions = value.dimensions;
          formattedData[key] = value.value;
          for (const dimension of dimensions) {
            if (!dimensionState.hasOwnProperty(dimension.key)) {
              throw new Error(`Dimension state for ${dimension.key} is missing.`);
            }
            const i = dimensionState[dimension.key]
            formattedData[key] = formattedData[key][i]
          }
        } else {
          formattedData[key] = value;
        }
      }
      return fn({ length: Object.values(formattedData)[0].length }, { ...formattedData, ...options })
    default:
      return fn(data, options)
  }
}

const repeatedData = (data) => {
  return (_, i) => data[i % data.length];
}

const scope = {
  d3,
  Plot, React, ReactDOM,
  View: {
    PlotSpec: (x) => new PlotSpec(x),
    MarkSpec: (name, data, options) => new MarkSpec(name, data, options),
    md: (x) => renderMarkdown(x),
    flattenDimensions,
    repeatedData,
    flat,
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

function SlidersView({ info, state, setState }) {
  // `info` should be an object of {key, {label, size}}
  return html`
    <div>
      ${Object.entries(info).map(([key, {label, size}]) => html`
          <div class="flex flex-col my-2 gap-2" key=${key}>
            <label>${label || key}</label>
            ${state[key]}
            <input 
              type="range" 
              min="0" 
              max=${size - 1}
              value=${state[key] || 0} 
              onChange=${(e) => setState({ ...state, [key]: Number(e.target.value) })}
            />
                  </div>  
      `)}
    </div>
  `;
}

function PlotView({spec, dimensionState}) {
  const [parent, setParent] = useState(null)
  const ref = useCallback(setParent)
  useEffect(() => {
    if (parent) {
      const marks = spec.marks.map((m) => readMark(m, dimensionState))
      const plot = Plot.plot({ ...spec, marks: marks })
      parent.appendChild(plot)
      return () => parent.removeChild(plot)
    }
  }, [spec, parent, dimensionState])
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
  
  const [dimensionState, setDimensionState] = useState(
    Object.fromEntries(Object.entries(dimensionInfo).map(([k, d]) => [k, d.initial || 0]))
  );
  return html`
    <div>
      <${PlotView} spec=${spec} dimensionState=${dimensionState}></div>
      <${SlidersView} 
        info=${dimensionInfo}
        state=${dimensionState}
        setState=${setDimensionState}/>
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