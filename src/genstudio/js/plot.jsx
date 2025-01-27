import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import * as mobx from "mobx";
import * as React from "react";
import { $StateContext, AUTOGRID_MIN } from "./context";
import { events } from "./plot/events";
import { ellipse } from "./plot/ellipse";
import { img } from "./plot/img";
import { pixels } from "./plot/pixels"
import { binding, flatten, tw, useContainerWidth, joinClasses } from "./utils";

const Marks = {...Plot, ellipse, events, img, pixels}
const { useEffect } = React
export const DEFAULT_PLOT_OPTIONS = { inset: 10 };

// Add per-mark defaults
const PER_MARK_DEFAULTS = {
    "dot": {"fill": "currentColor"},
    "frame": {"stroke": "#dddddd"}
};

function deepMergeLayers(target, source) {
  if (!source || typeof source !== 'object') return target;

  return Object.keys(source).reduce((result, key) => {
    if (key === 'marks') {
      result[key] = [...result[key], ...source[key]];
    } else if (source[key] && typeof source[key] === 'object' && key in result) {
      result[key] = deepMergeLayers(result[key], source[key]);
    } else {
      result[key] = source[key];
    }
    return result;
  }, { ...target });
}

function mergePlotSpec(layers) {
  return layers.flat().reduce((mergedSpec, layer) => {
    if (!layer) return mergedSpec;
    if (layer instanceof MarkSpec) {
      mergedSpec.marks.push(layer);
      return mergedSpec;
    }
    return deepMergeLayers(mergedSpec, layer.spec || layer);
  }, {"marks": []});
}

export class PlotSpec {
    constructor({layers}) {
        this.spec = mergePlotSpec(layers);
    }

    render() {
        return <PlotWrapper spec={this.spec}/>;
    }
}

export class MarkSpec {
    constructor(name, data = [], options) {
        if (!Marks[name]) {
            throw new Error(`Plot function "${name}" not found.`);
        }
        this.fn = Marks[name];

        options = { ...PER_MARK_DEFAULTS[name], ...options }

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

    render() {
        return <PlotWrapper spec={{ marks: [this] }}/>;
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
        function isArrayLike(value) {
            return Array.isArray(value) || ArrayBuffer.isView(value);
        }

        if (data.constructor === Object && !('length' in data)) {
            let length = null
            data = {...data}
            for (let [key, value] of Object.entries(data)) {
                if (isArrayLike(value)) {
                    options[key] = value;
                    delete data[key]
                    length = value.length
                }
                if (length !== null) {
                    data.length = length
                }
            }
        }
        // handle facetWrap option (grid) with minWidth consideration
        // see https://github.com/observablehq/plot/pull/892/files
        if (options.facetGrid) {
            // Check if data is an array of objects
            if (!Array.isArray(data) || !data.every(item => typeof item === 'object' && item !== null)) {
                throw new Error("Invalid data format: facetGrid expects an array of objects");
            }
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

function someObject(obj) {
    return Object.keys(obj).length === 0 ? null : obj;
}

export function readMark(mark, width) {
    if (!(mark instanceof MarkSpec)) {
        return mark;
    }
    let { fn, data, options, plotOptions, extraMarks } = mark.compute(width);
    mark = someObject(options) ? fn(data, options) : fn(data)
    mark.plotOptions = plotOptions;
    return [mark, ...extraMarks]
}

function prepareSpec(spec, containerWidth) {
    // Calculate width considering maxWidth constraint
    let width = spec.width ?? containerWidth
    if (spec.maxWidth) {
        width = Math.min(width, spec.maxWidth)
    }

    const marks = spec.marks.flatMap((m) => readMark(m, width))
    spec = {...spec,
            width: width,
            ...marks.reduce((acc, mark) => ({ ...acc, ...mark.plotOptions }), {}),
            marks: marks
    }
    // if (!spec.height && containerHeight > 0 && !spec.aspectRatio) {
    //     spec.height = containerHeight
    // }

    if (spec.color_map) {
        const [domain, range] = [Object.keys(spec.color_map), Object.values(spec.color_map)];
        spec.color = spec.color
            ? {
                ...spec.color,
                domain: [...(spec.color.domain || []), ...domain],
                range: [...(spec.color.range || []), ...range]
              }
            : { domain, range };
        delete spec.color_map;
    }
    return spec
}

export function PlotWrapper({spec}) {
    const $state = React.useContext($StateContext)
    return <PlotView spec={spec} $state={$state} />
}
export function PlotView ({ spec, $state }) {
        const [ref, containerWidth] = useContainerWidth()
        useEffect(() => {
            const parent = ref.current
            if (parent && (containerWidth || spec.width)) {
                return mobx.autorun(() => {
                    const preparedSpec = prepareSpec(spec, containerWidth)
                    const startTime = performance.now();

                    const plot = binding("$state", $state, () => Plot.plot(preparedSpec));
                    const endTime = performance.now();
                    // plot.setAttribute('data-render-time-ms', `${endTime - startTime}`);
                    // console.log(`rendered plot in ${endTime - startTime}`)
                    parent.innerHTML = '';
                    if (spec.className) {
                        plot.setAttribute('class', tw(spec.className));
                    }
                    parent.appendChild(plot);
                })
            }
        }, [spec, containerWidth])
        return <div className={tw(`relative`)} style={spec.width && {width: spec.width}} ref={ref}></div>
    }
