import htm from "htm";
import * as React from "react";

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
  