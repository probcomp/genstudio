import {React, htm} from "./imports"
const { useState, useEffect } = React

export const html = htm.bind(React.createElement)

export const flatten = (data, dimensions) => {
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

export function binding(varName, varValue, f) {
  const prevValue = window[varName]
  window[varName] = varValue
  const ret = f()
  window[varName] = prevValue
  return ret
}

export function useCellUnmounted(el) {
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

export function useElementWidth(el) {
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
