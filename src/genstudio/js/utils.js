import {React, htm, Twind, presetAutoprefix, presetTailwind, presetTypography} from "./imports"
const { useState, useEffect } = React


const twindConfig = Twind.defineConfig({
  presets: [presetAutoprefix(), presetTailwind(), presetTypography()],
})

export const tw = Twind.twind(twindConfig, Twind.cssom())
export const twInstall = Twind.injectGlobal.bind(tw)
export const twKeyframes = Twind.keyframes.bind(tw)

export const  html = htm.bind(React.createElement)

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
    if (el) {
      const handleResize = () => setWidth(el.offsetWidth ? el.offsetWidth : document.body.offsetWidth);
      handleResize();
      window.addEventListener('resize', handleResize);
      return () => window.removeEventListener('resize', handleResize);
    }
  }, [el]);

  return width
}

export function serializeEvent(e) {

  if (e.constructor === Object) {
    return e;
  }
  e = e.nativeEvent || e;

  const baseEventData = {
    type: e.type,
    altKey: e.altKey,
    ctrlKey: e.ctrlKey,
    shiftKey: e.shiftKey
  };

  switch (e.type) {
    case 'mousedown':
    case 'mouseup':
    case 'mousemove':
    case 'click':
      return {
        ...baseEventData,
        clientX: e.clientX,
        clientY: e.clientY,
        button: e.button
      };
    case 'keydown':
    case 'keyup':
    case 'keypress':
      return {
        ...baseEventData,
        key: e.key,
        code: e.code
      };
    case 'submit':
      e.preventDefault(); // Cancel form submit event by default
      return {
        ...baseEventData,
        formData: Object.fromEntries(new FormData(e.target))
      };
    case 'input':
    case 'change':
      if (e.target.type === 'checkbox' || e.target.type === 'radio') {
        return {
          ...baseEventData,
          checked: e.target.checked,
          value: e.target.value
        };
      } else if (e.target.type === 'select-one' || e.target.type === 'select-multiple') {
        return {
          ...baseEventData,
          value: Array.from(e.target.selectedOptions, option => option.value)
        };
      } else if (e.target.type === 'file') {
        return {
          ...baseEventData,
          files: Array.from(e.target.files, file => ({
            name: file.name,
            type: file.type,
            size: file.size
          }))
        };
      } else {
        return {
          ...baseEventData,
          value: e.target.value
        };
      }
    case 'focus':
    case 'blur':
      return {
        ...baseEventData,
        target: e.target.id || e.target.name || undefined
      };
    default:
      return {
        ...baseEventData,
        target: e.target.id || e.target.name || undefined
      };
  }
}
