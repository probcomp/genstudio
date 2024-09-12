import { $StateContext, WidthContext, AUTOGRID_MIN as AUTOGRID_MIN_WIDTH } from "./context";
import { MarkSpec, PlotSpec } from "./plot";
import { html } from "./utils";
import { Plot, d3, MarkdownIt, React, ReactDOM, mobxReact } from "./imports";
const { useState, useEffect, useContext, useMemo, useCallback } = React
import bylight from "bylight";
import { tw } from "./utils";
import * as render from "./render";

export { render };
const DEFAULT_GRID_GAP = "10px"
export const CONTAINER_PADDING = 10;

const MarkdownItInstance = new MarkdownIt({
    html: true,
    linkify: true,
    typographer: true
});

export function md(text) {
    return html`<div className=${tw("prose")} dangerouslySetInnerHTML=${{ __html: MarkdownItInstance.render(text) }} />`;
}

export const Slider = mobxReact.observer(
    function (options) {
        let { state_key,
            fps,
            label,
            loop = true,
            init, range, rangeFrom, tail, step } = options;

        if (init === undefined && rangeFrom === undefined && range === undefined) {
            throw new Error("Reactive: 'init', 'rangeFrom', or 'range' must be defined");
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
        const availableWidth = useContext(WidthContext);
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
        <div className=${tw("text-base flex flex-col my-2 gap-2")} style=${{ width: availableWidth }}>
          <div className=${tw("flex items-center justify-between")}>
            <span className=${tw("flex gap-2")}>
              <label>${label}</label>
              <span>${$state[state_key]}</span>
            </span>
            ${isAnimated && html`
              <div onClick=${togglePlayPause} className=${tw("cursor-pointer")}>
                ${isPlaying ? pauseIcon : playIcon}
              </div>
            `}
          </div>
          <input
            type="range"
            min=${rangeMin}
            max=${rangeMax}
            step=${step}
            value=${sliderValue}
            onChange=${(e) => handleSliderChange(e.target.value)}
            className=${tw("w-full outline-none")}
          />
        </div>
      `;
    }
)

export function clamp(value, min, max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

export class Reactive {
    render() { }
}

const playIcon = html`<svg viewBox="0 0 24 24" width="24" height="24"><path fill="currentColor" d="M8 5v14l11-7z"></path></svg>`;
const pauseIcon = html`<svg viewBox="0 24 24" width="24" height="24"><path fill="currentColor" d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"></path></svg>`;

export const Frames = mobxReact.observer(
    function (props) {
        const { state_key, frames } = props
        const $state = useContext($StateContext);
        console.log("Frames", props)

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
export { d3, Plot, React, ReactDOM, PlotSpec, MarkSpec };

export function Grid({ children, style, minWidth = AUTOGRID_MIN_WIDTH, gap = DEFAULT_GRID_GAP, aspectRatio = 1 }) {
    const availableWidth = useContext(WidthContext);
    const effectiveMinWidth = Math.min(minWidth, availableWidth);
    const gapSize = parseInt(gap);

    const numColumns = Math.max(1, Math.min(Math.floor(availableWidth / effectiveMinWidth), children.length));
    const itemWidth = (availableWidth - (numColumns - 1) * gapSize) / numColumns;
    const itemHeight = itemWidth / aspectRatio;
    const numRows = Math.ceil(children.length / numColumns);
    const layoutHeight = numRows * itemHeight + (numRows - 1) * gapSize;

    const containerStyle = {
        display: 'grid',
        gap,
        gridTemplateColumns: `repeat(${numColumns}, 1fr)`,
        gridAutoRows: `${itemHeight}px`,
        height: `${layoutHeight}px`,
        width: `${availableWidth}px`,
        overflowX: 'auto',
        ...style
    };

    return html`
    <${WidthContext.Provider} value=${itemWidth}>
      <div style=${containerStyle}>
        ${children.map((value, index) => html`<${Node} key=${index}
                                                       style=${{ width: itemWidth }}
                                                       value=${value}/>`)}
      </div>
    </>
  `;
}

export function Row({ children, ...props }) {
    const availableWidth = useContext(WidthContext);
    const childCount = React.Children.count(children);
    const childWidth = availableWidth / childCount;

    return html`
    <div ...${props} className=${tw("flex flex-row")}>
      <${WidthContext.Provider} value=${childWidth}>
        ${React.Children.map(children, (child, index) => html`
          <div className=${tw("flex-1")} key=${index}>
            ${child}
          </div>
        `)}
      </${WidthContext.Provider}>
    </div>
  `;
}

export function Column({ children, ...props }) {
    return html`
    <div ...${props} className=${tw("flex flex-col")}>
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
        value = $state.resolveCached(value)
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
