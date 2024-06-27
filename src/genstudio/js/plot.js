
import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import * as React from "react";
import { $StateContext, WidthContext, AUTOGRID_MIN } from "./context";
import { binding, flatten, html } from "./utils";

const { useEffect } = React
export const DEFAULT_PLOT_OPTIONS = { inset: 20 };

/**
 * Wrap plot specs so that our node renderer can identify them.
 */
export class PlotSpec {
    /**
     * Create a new plot spec.
     */
    constructor(spec) {
        this.spec = spec;
    }
}
export class MarkSpec {
    constructor(name, data, options) {
        if (!Plot[name]) {
            throw new Error(`Plot function "${name}" not found.`);
        }
        this.fn = Plot[name];

        options = {...options}

        // handle dimensional data passed in the 1st position
        if (data.dimensions) {
            options.dimensions = data.dimensions
            data = data.value
        }

        // flatten dimensional data
        if (options.dimensions) {
            data = flatten(data, options.dimensions)
        }

        this.data = data;
        this.options = options;
        this.computed = {}
    }

    compute(width) {
        if (width && this.computed.forWidth == width) {
            return this.computed
        }
        this.extraMarks = [];
        let data = this.data
        let options = { ...this.options }
        let computed = {
            forWidth: width,
            extraMarks: [],
            plotOptions: { ...DEFAULT_PLOT_OPTIONS },
            fn: this.fn
        }

        // Below is where we add functionality to Observable.Plot by preprocessing
        // the data & options that are passed in.

        // handle columnar data in the 1st position
        if (data.constructor === Object && !('length' in data)) {
            let length = null
            for (let [key, value] of Object.entries(data)) {
                options[key] = value;
                if (Array.isArray(value)) {
                    length = value.length
                }
                if (length === null) {
                    throw new Error("Invalid columnar data: at least one column must be an array.");
                }
                data = { length: length }
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
            const minWidth = gridOpts.minWidth || AUTOGRID_MIN;
            const columns = gridOpts.columns || Math.floor(width / minWidth);
            const fx = (key) => index.get(key) % columns;
            const fy = (key) => Math.floor(index.get(key) / columns);
            options.fx = (d) => fx(d[key]);
            options.fy = (d) => fy(d[key]);
            computed.plotOptions = { ...computed.plotOptions, fx: { axis: null }, fy: { axis: null } };
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

export function readMark(mark, width) {

    if (!(mark instanceof MarkSpec)) {
        return mark;
    }
    let { fn, data, options, plotOptions, extraMarks } = mark.compute(width);
    mark = fn(data, options);
    mark.plotOptions = plotOptions;
    return [mark, ...extraMarks]
}

export function PlotView({ spec, $state, width }) {
    const [parent, setParent] = React.useState(null)
    const ref = React.useCallback(setParent)
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

export function PlotWrapper({ spec }) {
    const [$state, set$state] = React.useContext($StateContext)
    const width = React.useContext(WidthContext)
    const marks = spec.marks.flatMap((m) => readMark(m, width))
    spec = {
        width: width,
        ...spec,
        ...marks.reduce((acc, mark) => ({ ...acc, ...mark.plotOptions }), {}),
        marks: marks
    };
    return html`<${PlotView} spec=${spec} $state=${$state} />`
}
