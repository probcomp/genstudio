import { $StateContext, AUTOGRID_MIN as AUTOGRID_MIN_WIDTH } from "./context";
import { MarkSpec, PlotSpec } from "./plot";
import { html } from "./utils";

import * as Plot from "@observablehq/plot";
import bylight from "bylight";
import * as d3 from "d3";
import MarkdownIt from "markdown-it";
import * as mobxReact from "mobx-react-lite";
import * as React from "react";
import * as ReactDOM from "react-dom/client";
import * as render from "./plot/render";
import { tw, useContainerWidth } from "./utils";
const { useState, useEffect, useContext, useMemo, useRef, useCallback } = React
import Katex from "katex";
import markdownItKatex from "./markdown-it-katex";

export { render };
export const CONTAINER_PADDING = 10;
const KATEX_CSS_URL = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css"

let katexCssLoaded = false;
function loadKatexCss() {
    if (katexCssLoaded) return;
    if (!document.querySelector(`link[href="${KATEX_CSS_URL}"]`)) {
        const link = document.createElement('link');
        link.rel = 'stylesheet';
        link.href = KATEX_CSS_URL;
        document.head.appendChild(link);
    }
    katexCssLoaded = true;
}

export function katex(tex) {
    const containerRef = useRef(null);

    loadKatexCss();

    useEffect(() => {
        if (containerRef.current) {
            try {
                Katex.render(tex, containerRef.current, {
                    throwOnError: false
                });
            } catch (error) {
                console.error('Error rendering KaTeX:', error);
            }
        }
    }, [tex]);

    return html`<div ref=${containerRef} />`;
}

const MarkdownItInstance = new MarkdownIt({
    html: true,
    linkify: true,
    typographer: true
});

MarkdownItInstance.use(markdownItKatex)

export function md(text) {
    loadKatexCss();

    return html`<div className=${tw("prose")} dangerouslySetInnerHTML=${{ __html: MarkdownItInstance.render(text) }} />`;
}

export const Slider = mobxReact.observer(
    function (options) {
        let { state_key,
            fps,
            label,
            loop = true,
            init, range, rangeFrom, showValue, showSlider, tail, step } = options;

        if (init === undefined && rangeFrom === undefined && range === undefined) {
            throw new Error("Slider: 'init', 'rangeFrom', or 'range' must be defined");
        }

        let rangeMin, rangeMax;
        if (rangeFrom) {
            // determine range dynamically based on last index of rangeFrom
            rangeMin = 0;
            rangeMax = rangeFrom.length - 1;
        } else if (typeof range === 'number') {
            // range may be specified as the upper end of a [0, n] integer range.
            rangeMin = 0;
            rangeMax = range;
        } else if (range) {
            [rangeMin, rangeMax] = range;
        }
        step = step || 1;

        const $state = useContext($StateContext);
        const isAnimated = typeof fps === 'number' && fps > 0;
        const [isPlaying, setIsPlaying] = useState(isAnimated);

        const sliderValue = clamp($state[state_key] ?? rangeMin, rangeMin, rangeMax);

        useEffect(() => {
            if (isAnimated && isPlaying) {
                const intervalId = setInterval(() => {
                    $state[state_key] = (prevValue) => {
                        const nextValue = (prevValue || 0) + step;
                        if (nextValue > rangeMax) {
                            if (tail) {
                                return rangeMax;
                            } else if (loop) {
                                return rangeMin;
                            } else {
                                setIsPlaying(false);
                                return rangeMax;
                            }
                        }
                        return nextValue;
                    };
                }, 1000 / fps);
                return () => clearInterval(intervalId);
            }
        }, [isPlaying, fps, state_key, rangeMin, rangeMax, step, loop, tail]);

        const handleSliderChange = useCallback((value) => {
            setIsPlaying(false);
            $state[state_key] = Number(value);
        }, [$state, state_key]);

        const togglePlayPause = useCallback(() => setIsPlaying((prev) => !prev), []);
        if (options.visible !== true) return;
        return html`
        <div className=${tw("text-base flex flex-col my-2 gap-2 w-full")}>
          <div className=${tw("flex items-center justify-between")}>
            <span className=${tw("flex gap-2")}>
              <label>${label}</label>
              <span>${showValue && $state[state_key]}</span>
            </span>
            ${isAnimated && html`
              <div onClick=${togglePlayPause} className=${tw("cursor-pointer")}>
                ${isPlaying ? pauseIcon : playIcon}
              </div>
            `}
          </div>
          ${showSlider && html`<input
            type="range"
            min=${rangeMin}
            max=${rangeMax}
            step=${step}
            value=${sliderValue}
            onChange=${(e) => handleSliderChange(e.target.value)}
            className=${tw("w-full outline-none")}
          />`}
        </div>
      `;
    }
)

export function clamp(value, min, max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

export class InitialState {
    render() { }
}

export class OnStateChange {
    // this could be a way of "mounting" a ref callback. eg.
    // a_plot | Plot.onChange({})

    // alternatively, on a widget we could do something like
    // widget.onChange({"foo": cb})

    // alternatively, one might want to sync some state, like
    // widget.sync("foo", "bar")
    // and then read the synced values via widget.foo

    constructor(name, callback) {
        this.name = name
        this.callback = callback
    }
    render() {

        const $state = useContext($StateContext);
        useEffect(() => {
            return mobx.autorun(() => {
                this.callback($state[this.name])
            })
        },
            [this.name, this.callback])
    }
}


const playIcon = html`<svg viewBox="0 0 24 24" width="24" height="24"><path fill="currentColor" d="M8 5v14l11-7z"></path></svg>`;
const pauseIcon = html`<svg viewBox="0 24 24" width="24" height="24"><path fill="currentColor" d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"></path></svg>`;

export const Frames = mobxReact.observer(
    function (props) {
        const { state_key, frames } = props
        const $state = useContext($StateContext);

        if (!Array.isArray(frames)) {
            return html`<div className=${tw("text-red-500")}>Error: 'frames' must be an array.</div>`;
        }

        const index = $state[state_key];
        if (!Number.isInteger(index) || index < 0 || index >= frames.length) {
            return html`<div className=${tw("text-red-500")}>Error: Invalid index. $state[${state_key}] (${index}) must be a valid index of the frames array (length: ${frames.length}).</div>`;
        }

        return html`<${Node} value=${frames[index]} />`;
    }
)
export class Bylight {
    constructor(source, patterns, props = {}) {
        this.patterns = patterns;
        this.source = source;
        this.props = props;
    }

    render() {
        const preRef = React.useRef(null);

        React.useEffect(() => {
            if (preRef.current && this.patterns) {
                bylight.highlight(preRef.current, this.patterns);
            }
        }, [this.source, this.patterns]);

        return React.createElement('pre', {
            ref: preRef,
            className: this.props.className
        }, this.source);
    }
}

export function repeat(data) {
    const length = data.length
    return (_, i) => data[i % length]

}
export { d3, MarkSpec, Plot, PlotSpec, React, ReactDOM };

export function Grid({
    children,
    style,
    minWidth = AUTOGRID_MIN_WIDTH,
    gap = 1,
    rowGap,
    colGap,
    cols,
    minCols = 1,
    maxCols
}) {
    const [containerRef, containerWidth] = useContainerWidth();

    // Handle gap values
    const gapX = colGap ?? gap;
    const gapY = rowGap ?? gap;
    const gapClass = `gap-x-${gapX} gap-y-${gapY}`;
    const gapSize = parseInt(gap); // Keep for width calculations

    // Calculate number of columns
    let numColumns;
    if (cols) {
        numColumns = cols;
    } else {
        const effectiveMinWidth = Math.min(minWidth, containerWidth);
        const autoColumns = Math.floor(containerWidth / effectiveMinWidth);
        numColumns = Math.max(
            minCols,
            maxCols ? Math.min(autoColumns, maxCols) : autoColumns,
            1
        );
        numColumns = Math.min(numColumns, children.length);
    }

    const itemWidth = (containerWidth - (numColumns - 1) * gapSize) / numColumns;

    const containerStyle = {
        display: 'grid',
        gridTemplateColumns: `repeat(${numColumns}, 1fr)`,
        width: '100%',
        ...style
    };

    return html`
    <div ref=${containerRef} class=${tw(gapClass)} style=${containerStyle}>
        ${children.map((value, index) => html`<${Node} key=${index}
                                                       style=${{ width: itemWidth }}
                                                       value=${value}/>`)}
      </div>
  `;
}

export function Row({ children, gap=1, widths, ...props }) {
    const className = `flex flex-row gap-${gap} ${props.className || props.class || ''}`
    delete props["className"]

    let flexClasses = []
    if (widths) {
        flexClasses = widths.map(w => {
            if (typeof w === 'string') {
                return w.includes('/') ? `w-${w}` : `w-[${w}]`
            }
            return `flex-[${w}]`
        })
    } else {
        flexClasses = Array(React.Children.count(children)).fill("flex-1")
    }

    return html`
    <div ...${props} className=${tw(className)}>
      ${React.Children.map(children, (child, index) => html`
        <div className=${tw(flexClasses[index])} key=${index}>
          ${child}
        </div>
      `)}
    </div>
  `;
}

export function Column({ children, gap=1, ...props }) {
    return html`
    <div ...${props} className=${tw(`flex flex-col gap-${gap}`)}>
    ${React.Children.map(children, (child, index) => html`
        <div key=${index}>
          ${child}
        </div>
      `)}
    </div>
  `;
}

export const Node = mobxReact.observer(
    function ({ value }) {
        const $state = useContext($StateContext)
        value = $state.resolveRef(value)
        if (Array.isArray(value)) {
            const [element, ...args] = value
            const maybeElement = element && $state.evaluate(element)
            const elementType = typeof maybeElement

            if (elementType === 'string' || elementType === 'function' || (typeof maybeElement === 'object' && maybeElement !== null && "$$typeof" in maybeElement)) {
                return Hiccup(maybeElement, ...args)
            } else {
                return html`<${React.Fragment} children=${value.map(item =>
                    typeof item !== 'object' || item === null ? item : html`<${Node} value=${item} />`
                )} />`;
            }
        }
        const evaluatedValue = $state.evaluate(value)
        if (typeof evaluatedValue === 'object' && evaluatedValue !== null && 'render' in evaluatedValue) {
            return evaluatedValue.render();
        } else {
            return evaluatedValue;
        }
    }
)

export function Hiccup(tag, props, ...children) {
    const $state = useContext($StateContext)

    if (props?.constructor !== Object || props.__type__ || React.isValidElement(props)) {
        children.unshift(props);
        props = {};
    }
    props = $state.evaluate(props)

    if (props.class) {
        props.className = props.class;
        delete props.class;
    }

    let baseTag = tag;
    if (typeof tag === 'string') {
        let id, classes
        [baseTag, ...classes] = tag.split('.');
        [baseTag, id] = baseTag.split('#');

        if (id) { props.id = id; }

        if (classes.length > 0) {
            if (props.className) {
                classes.push(props.className);
            }
            props.className = classes.join(' ');
        }
    }

    if (props.className) {
        props.className = tw(props.className)
    }

    return children.length > 0
        ? html`<${baseTag} ...${props}>
            ${children.map((child, index) => html`<${Node} key=${index} value=${child}/>`)}
          </>`
        : html`<${baseTag} ...${props} />`;
}
