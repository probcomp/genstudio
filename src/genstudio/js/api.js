import { $StateContext, WidthContext, AUTOGRID_MIN as AUTOGRID_MIN_WIDTH } from "./context";
import { MarkSpec, PlotSpec } from "./plot";
import { html } from "./utils";
import { Plot, d3, MarkdownIt, React, ReactDOM } from "./imports";
const { useState, useEffect, useContext, useMemo, useCallback } = React
import bylight from "bylight";

const DEFAULT_GRID_GAP = "10px"
export const CONTAINER_PADDING = 10;

const MarkdownItInstance = new MarkdownIt({
    html: true,
    linkify: true,
    typographer: true
});

export function md(text) {
    return html`<div className='prose' dangerouslySetInnerHTML=${{ __html: MarkdownItInstance.render(text) }} />`;
}

export function ReactiveSlider(options) {
    let { state_key, fps, label, step = 1, loop = true, tail, rangeMin, rangeMax } = options;
    const [$state, set$state] = useContext($StateContext);
    const availableWidth = useContext(WidthContext);
    const isAnimated = typeof fps === 'number' && fps > 0;
    const [isPlaying, setIsPlaying] = useState(isAnimated);

    const sliderValue = clamp($state[state_key] ?? rangeMin, rangeMin, rangeMax);

    useEffect(() => {
        if (isAnimated && isPlaying) {
            const intervalId = setInterval(() => {
                $state[state_key] = (prevValue) => {
                    const nextValue = prevValue + step;
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
    }, [set$state, state_key]);

    const togglePlayPause = useCallback(() => setIsPlaying((prev) => !prev), []);
    if (options.kind !== 'Slider') return;
    return html`
    <div className="f1 flex flex-column mv2 gap2" style=${{ width: availableWidth }}>
      <div className="flex items-center justify-between">
        <span className="flex g2">
          <label>${label}</label>
          <span>${$state[state_key]}</span>
        </span>
        ${isAnimated && html`
          <div onClick=${togglePlayPause} className="pointer">
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
        className="w-100 outline-0"
      />
    </div>
  `;
}

export function clamp(value, min, max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

export class Reactive {
    constructor(data) {
        let { init, range, rangeFrom, tail, step } = data;

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
        init = init ?? rangeMin;
        step = step || 1;


        this.options = {
            ...data,
            rangeMin,
            rangeMax,
            tail,
            init,
            step
        };
    }

    render() {
        return ReactiveSlider(this.options);
    }
}

const playIcon = html`<svg viewBox="0 0 24 24" width="24" height="24"><path fill="currentColor" d="M8 5v14l11-7z"></path></svg>`;
const pauseIcon = html`<svg viewBox="0 24 24" width="24" height="24"><path fill="currentColor" d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"></path></svg>`;

export function Frames(props) {
    const { state_key, frames } = props
    const [$state] = useContext($StateContext);
    if (!Array.isArray(frames)) {
        return html`<div className="red">Error: 'frames' must be an array.</div>`;
    }

    const index = $state[state_key];
    if (!Number.isInteger(index) || index < 0 || index >= frames.length) {
        return html`<div className="red">Error: Invalid index. $state[${state_key}] (${index}) must be a valid index of the frames array (length: ${frames.length}).</div>`;
    }

    return html`<${Node} value=${frames[index]} />`;
}

export class Bylight {
    constructor(source, patterns, props={}) {
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
    <div ...${props} className="layout-row">
      <${WidthContext.Provider} value=${childWidth}>
        ${React.Children.map(children, (child, index) => html`
          <div className="row-item" key=${index}>
            ${child}
          </div>
        `)}
      </${WidthContext.Provider}>
    </div>
  `;
}

export function Column({ children, ...props }) {
    return html`
    <div ...${props} className="layout-column">
      ${children}
    </div>
  `;
}

export function Node({ value }) {
    if (Array.isArray(value)) {
        return (['string', 'function'].includes(typeof value[0])) ? Hiccup(...value) : Hiccup("div", ...value);
    } else if (typeof value === 'object' && value !== null && 'render' in value) {
        return value.render();
    } else {
        return value;
    }
}

export function Hiccup(tag, props, ...children) {
    if (props?.constructor !== Object) {
        children.unshift(props);
        props = {};
    }

    let baseTag = tag;
    if (typeof tag === 'string') {
        let id, classes
        [baseTag, ...classes] = tag.split('.');
        [baseTag, id] = baseTag.split('#');

        if (id) { props.id = id; }

        if (classes.length > 0) {
            props.className = `${props.className || ''} ${classes.join(' ')}`.trim();
        }
    }

    return baseTag instanceof PlotSpec
        ? baseTag.render()
        : children.length > 0
            ? html`<${baseTag} ...${props}>
          ${children.map((child, index) => html`<${Node} key=${index} value=${child}/>`)}
        </>`
            : html`<${baseTag} ...${props} />`;
}
