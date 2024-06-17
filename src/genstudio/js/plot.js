
import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import * as React from "react";
import { WidthContext, AUTOGRID_MIN } from "./context";
import { binding, flatten, html } from "./utils";

const { useEffect } = React
const DEFAULT_PLOT_OPTIONS = { inset: 20 };

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

function $StateView({ $info, $state, set$state }) {
    const [isPlaying, setIsPlaying] = React.useState(true);

    // Build up the active intervals array up top
    const activeIntervals = Object.entries($info).filter(([key, { kind, fps }]) => kind === 'animate' && fps > 0);

    useEffect(() => {
        const intervals = activeIntervals.map(([key, { fps, loop, range }]) => {
            const intervalId = setInterval(() => {
                if (isPlaying) {
                    set$state((prevState) => {
                        const nextValue = prevState[key] + 1;
                        if (nextValue < range[1]) {
                            return { ...prevState, [key]: nextValue };
                        } else {
                            return loop !== false ? { ...prevState, [key]: range[0] } : prevState;
                        }
                    });
                }
            }, 1000 / fps);
            return intervalId;
        });

        return () => intervals.forEach(clearInterval);
    }, [activeIntervals, set$state, isPlaying]);

    const togglePlayPause = () => {
        setIsPlaying(!isPlaying);
    };

    const handleSliderChange = (key, value) => {
        setIsPlaying(false);
        set$state({ ...$state, [key]: Number(value) });
    };

    const playIcon = `<svg viewBox="0 0 24 24" width="24" height="24"><path fill="currentColor" d="M8 5v14l11-7z"></path></svg>`;
    const pauseIcon = `<svg viewBox="0 0 24 24" width="24" height="24"><path fill="currentColor" d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"></path></svg>`;

    return html`
      <div style=${{ display: 'flex', alignItems: 'center', gap: '10px', width: '100%' }}>
        ${activeIntervals.length ? html`
          <div onClick=${togglePlayPause} style=${{ flexShrink: 0 }} dangerouslySetInnerHTML=${{ __html: isPlaying ? pauseIcon : playIcon }}></div>
        ` : null}
        <div style=${{ display: 'flex', flexGrow: 1, flexDirection: 'column' }}>
        ${Object.entries($info).map(([key, { label, range, init, kind }]) => html`
          <div style=${{ fontSize: "14px", display: 'flex', flexDirection: 'column', marginTop: '0.5rem', marginBottom: '0.5rem', gap: '0.5rem' }} key=${key}>
            <label>${label || key} ${$state[key]}</label>
            ${kind === 'animate' ? null : html`<span>${$state[key]}</span>`}
            <input 
              type="range" 
              min=${range[0]} 
              max=${range[1] - 1}
              value=${$state[key] || init || 0} 
              onChange=${(e) => handleSliderChange(key, e.target.value)}
              style=${{ outline: 'none' }}
            />
          </div>
        `)}
        </div>
      </div>
    `;
}

export function PlotWrapper({ spec }) {
    const $info = spec.$state || {}
    const [$state, set$state] = React.useState(Object.keys($info).reduce((stateObj, key) => ({
        ...stateObj,
        [key]: $info[key].init || $info[key].range[0]
    }), {}))

    const width = React.useContext(WidthContext)
    const marks = spec.marks.flatMap((m) => readMark(m, width))
    spec = {
        width: width,
        ...spec,
        ...marks.reduce((acc, mark) => ({ ...acc, ...mark.plotOptions }), {}),
        marks: marks
    };
    return html`<div>
                  <${PlotView} spec=${spec} $state=${$state} />
                  <${$StateView} $state=${$state} set$state=${set$state} $info=${$info} />  
                </div>`
}