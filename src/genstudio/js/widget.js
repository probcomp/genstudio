import { createRender, useModelState } from "@anywidget/react";
import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import MarkdownIt from "markdown-it";
import * as React from "react";
import * as ReactDOM from "react-dom";
import { WidthContext, AUTOGRID_MIN } from "./context";
import { MarkSpec, PlotSpec, PlotWrapper } from "./plot";
import { flatten, html } from "./utils";
const { useState, useEffect, useContext, useMemo } = React

const md = new MarkdownIt({
  html: true,
  linkify: true,
  typographer: true
});

function renderMarkdown(text) {
  return html`<div class='prose' dangerouslySetInnerHTML=${{ __html: md.render(text) }} />`;
}
/**
 * Create a new element.
 */
const hiccup = (tag, props, ...children) => {
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

const scope = {
  d3,
  Plot, React, ReactDOM,
  View: {
    PlotSpec: (x) => new PlotSpec(x),
    MarkSpec: (name, data, options) => new MarkSpec(name, data, options),
    md: (x) => renderMarkdown(x),
    repeat: (data) => (_, i) => data[i % data.length],
    hiccup,
    AutoGrid,
    flatten
  }
}

/**
 * Interpret data recursively, evaluating functions.
 */
export function interpret(data) {

  if (data === null) return null;
  if (Array.isArray(data)) return data.map(interpret);
  if (typeof data === "string" || data instanceof String) return data;
  if (data.constructor !== Object) return data;

  switch (data["pyobsplot-type"]) {
    case "function":
      let fn = data.name ? scope[data.module][data.name] : scope[data.module]
      if (!fn) {
        console.error('f not found', data)
      }
      const interpretedArgs = interpret(data.args)
      return fn.call(null, ...interpretedArgs);
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
    return hiccup.apply(null, value);
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
        setWidth(el.offsetWidth ? el.offsetWidth : document.body.offsetWidth);
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
  const interpretedData = useMemo(() => data ? interpret(JSON.parse(data)) : null, [data])
  const width = useElementWidth(el)
  const unmounted = useCellUnmounted(el?.parentNode);
  const value = unmounted ? null : interpretedData;
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
