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

const WidthContext = React.createContext();


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

const flatten = (data, dimensions) => {
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
    const dimName = typeof dim[0] === 'string' ? dim[0] : dim[0].key;
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
    if (!Plot[name]) {
      throw new Error(`Plot function "${name}" not found.`);
    }
    this.fn = Plot[name];
    this.data = data;
    this.options = options;
    this.computed = {}
  }

  compute(containerWidth) {
    if (this.computed.forWidth == containerWidth) {
      return this.computed
    }
    this.extraMarks = [];
    let data = this.data
    let options = { tip: true, ...this.options }
    let computed = {
      forWidth: containerWidth,
      extraMarks: [],
      plotOptions: {},
      fn: this.fn
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
      data = flatten(data, options.dimensions)
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
        data = { length: value.length }
      }

    }
    // handle facetWrap option (grid) with minWidth consideration
    // see https://github.com/observablehq/plot/pull/892/files
    if (options.facetGrid) {
      const facetGrid = (typeof options.facetGrid === 'string') ? [options.facetGrid, {}] : options.facetGrid;
      const [key, gridOpts] = facetGrid;
      const keys = Array.from(d3.union(data.map((d) => d[key])));
      const index = new Map(keys.map((key, i) => [key, i]));

      // Calculate columns based on minWidth and containerWidth
      const minWidth = gridOpts.minWidth || 200;
      const columns = gridOpts.columns || Math.min(Math.floor(containerWidth / minWidth), Math.floor(Math.sqrt(keys.length)));

      const fx = (key) => index.get(key) % columns;
      const fy = (key) => Math.floor(index.get(key) / columns);
      options.fx = (d) => fx(d[key]);
      options.fy = (d) => fy(d[key]);
      computed.plotOptions = { fx: { axis: null }, fy: { axis: null } };
      computed.extraMarks.push(Plot.text(keys, {
        fx, fy,
        frameAnchor: "top",
        dy: 4
      }));
    }

    computed.data = data;
    computed.options = options;
    this.computed = computed;
    return computed;
  }
}

function readMark(mark, width) {

  if (!(mark instanceof MarkSpec)) {
    return mark;
  }
  let { fn, data, options, plotOptions, extraMarks } = mark.compute(width);
  mark = fn(data, options);
  mark.plotOptions = plotOptions;
  return [mark, ...extraMarks]
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
    el,
    AutoGrid,
    flatten
  },

}
const { useState, useEffect, useRef, useCallback, useContext, useMemo } = React

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

function SlidersView({ $info, $state, set$state }) {
  return html`
    <div>
      ${Object.entries($info).map(([key, { label, range, init }]) => html`
          <div class="flex flex-col my-2 gap-2" key=${key}>
            <label>${label || key}</label>
            ${$state[key]}
            <input 
              type="range" 
              min=${range[0]} 
              max=${range[1]}
              value=${$state[key] || init || 0} 
              onChange=${(e) => set$state({ ...$state, [key]: Number(e.target.value) })}
            />
                  </div>  
      `)}
    </div>
  `;
}

function binding(varName, varValue, f) {
  const prevValue = window[varName]
  window[varName] = varValue
  const ret = f()
  window[varName] = prevValue
  return ret
}

function PlotView({ spec, $state, width }) {
  const [parent, setParent] = useState(null)
  const ref = useCallback(setParent)
  useEffect(() => {
    if (parent) {
      const plot = binding("$state", $state, () => Plot.plot(spec))
      parent.appendChild(plot)
      return () => parent.removeChild(plot)
    }
  }, [spec, parent, width])
  return html`
    <div ref=${ref}></div>
  `
}

function PlotWrapper({ spec }) {
  const $info = spec.$state || {} 
  const [$state, set$state] = useState(Object.keys($info).reduce((stateObj, key) => ({
    ...stateObj,
    [key]: $info[key].init || $info[key].range[0]
  }), {}))
  
  const width = useContext(WidthContext)
  const marks = spec.marks.flatMap((m) => readMark(m, width))
  spec = {
    ...spec,
    ...marks.reduce((acc, mark) => ({ ...acc, ...mark.plotOptions }), {}),
    marks: marks
  };
  return html`<div>
                <${PlotView} spec=${spec} $state=${$state} />
                <${SlidersView} $state=${$state} set$state=${set$state} $info=${$info} />  
              </div>`
}

function AutoGrid({ specs: PlotSpecs, plotOptions, layoutOptions }) {
  normalizeDomains(PlotSpecs)
  const containerWidth = useContext(WidthContext);
  const aspectRatio = plotOptions.aspectRatio || 1;
  const minWidth = Math.min(plotOptions.minWidth || 200, containerWidth);
  const defaultPlotOptions = { inset: 10 };

  // Compute the number of columns based on containerWidth and minWidth, no more than the number of specs
  const numColumns = Math.max(Math.min(Math.floor(containerWidth / minWidth), PlotSpecs.length), 1);
  const itemWidth = containerWidth / numColumns;
  const itemHeight = itemWidth / aspectRatio;

  // Merge width and height into defaultPlotOptions
  const mergedPlotOptions = {
    ...defaultPlotOptions,
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
    return el.apply(null, value);
  } else if (value instanceof PlotSpec) {
    return html`<${PlotWrapper} spec=${value.spec}/>`;
  } else if (value instanceof MarkSpec) {
    return Node({ value: new PlotSpec({ marks: [value] }) })
  } else {
    return value
  }
}

function useCellUnmounted(el) {
  // for Python Interactive Output in VS Code, detect when this element 
  // is unmounted & save that state on the element itself.
  // We have to directly read from the ancestor DOM because none of our 
  // cell output is preserved across reload.
  useEffect(() => {
    let observer;
    // .output_container is stable across refresh
    const outputContainer = el?.closest(".output_container")
    // .widgetarea contains all the notebook's cells 
    const widgetarea = outputContainer?.closest(".widgetarea")
    if (el && !el.initialized && widgetarea) {
      el.initialized = true;

      const mutationCallback = (mutationsList, observer) => {
        for (let mutation of mutationsList) {
          if (mutation.type === 'childList' && !widgetarea.contains(outputContainer)) {
            el.unmounted = true
            observer.disconnect();
            break;
          }
        }
      };
      observer = new MutationObserver(mutationCallback);
      observer.observe(widgetarea, { childList: true, subtree: true });
    }
    return () => observer?.disconnect()
  }, [el]);
  return el?.unmounted
}

function useElementWidth(el) {
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const handleResize = () => {
      if (el) {
        setWidth(el.offsetWidth);
      }
    };

    // Set initial width
    handleResize();

    // Add event listener to update width on resize
    window.addEventListener('resize', handleResize);

    // Remove event listener on cleanup
    return () => window.removeEventListener('resize', handleResize);
  }, [el]);
  return width
}

function App() {
  const [el, setEl] = useState();
  const [data, _] = useModelState("data");
  const width = useElementWidth(el)
  const unmounted = useCellUnmounted(el?.parentNode);
  const value = !unmounted && data ? interpret(JSON.parse(data)) : null;
  return html`<${WidthContext.Provider} value=${width}>  
      <div style=${{ color: '#333' }} ref=${setEl}>
        <${Node} value=${el ? value : null}/>
      </div>
    </>`
}

const render = createRender(App)


const installCSS = () => {
  // const id = "tailwind-cdn"
  // const scriptUrl = "https://cdn.tailwindcss.com?plugins=forms,typography,aspect-ratio";
  // if (!document.getElementById(id)) {
  //   const script = document.createElement("script");
  //   script.id = id;
  //   script.src = scriptUrl;
  //   document.head.appendChild(script);
  // }
  // const url = "https://cdn.jsdelivr.net/gh/html-first-labs/static-tailwind@759f1d7/dist/tailwind.css"
  // if (!document.getElementById(id)) {
  //   const link = document.createElement("link");
  //   link.id = id;
  //   link.rel = "stylesheet";
  //   link.href = url;
  //   document.head.appendChild(link);
  // }
}

export default { render }
