import { $StateContext, AUTOGRID_MIN as AUTOGRID_MIN_WIDTH } from "./context";
import { MarkSpec, PlotSpec } from "./plot";
import * as Plot from "@observablehq/plot";
import bylight from "bylight";
import * as d3 from "d3";
import MarkdownIt from "markdown-it";
import * as mobxReact from "mobx-react-lite";
import * as React from "react";
import * as ReactDOM from "react-dom/client";
import * as render from "./plot/render";
import {Grid, Row, Column} from "./layout"
import { tw } from "./utils";
const { useState, useEffect, useContext, useRef, useCallback } = React
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

    return <div ref={containerRef} />;
}

const MarkdownItInstance = new MarkdownIt({
    html: true,
    linkify: true,
    typographer: true
});

MarkdownItInstance.use(markdownItKatex)

export function md(text) {
    loadKatexCss();

    return <div className={tw("prose")} dangerouslySetInnerHTML={{ __html: MarkdownItInstance.render(text) }} />;
}

export const Slider = mobxReact.observer(
    function (options) {
        let { state_key,
            fps,
            label,
            loop = true,
            showSlider = true,
            init, range, rangeFrom, showValue, showFps, tail, step } = options;

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
        const lastFrameTimeRef = useRef(performance.now());
        const frameCountRef = useRef(0);
        const lastLogTimeRef = useRef(performance.now());
        const [currentFps, setCurrentFps] = useState(0);

        const sliderValue = clamp($state[state_key] ?? rangeMin, rangeMin, rangeMax);

        useEffect(() => {
            if (isAnimated && isPlaying) {
                const intervalId = setInterval(() => {
                    const now = performance.now();
                    frameCountRef.current++;

                    // Log FPS once per second
                    if (now - lastLogTimeRef.current >= 500) {
                        const fps = frameCountRef.current * 1000 / (now - lastLogTimeRef.current);
                        setCurrentFps(Math.round(fps));
                        frameCountRef.current = 0;
                        lastLogTimeRef.current = now;
                    }

                    lastFrameTimeRef.current = now;

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
        if (showSlider !== true) return;
        return (
            <div className={tw("text-xs flex flex-col my-2 gap-2 w-full")}>

                    <span className={tw("flex gap-1")}>
                        {label && <label className={tw("font-semibold")}>{label}</label>}
                        {showValue && <span>{$state[state_key]}</span>}
                    </span>


                {showSlider && (
                    <div className={tw("flex gap-1 items-center")}>
                        <input
                        type="range"
                        min={rangeMin}
                        max={rangeMax}
                        step={step}
                        value={sliderValue}
                        onChange={(e) => handleSliderChange(e.target.value)}
                        className={tw("w-full outline-none")}
                    />
                    {showFps && isPlaying && <span className={tw("whitespace-nowrap -mr-1")}>{currentFps} FPS</span>}
                    {isAnimated && (
                        <div onClick={togglePlayPause} className={tw("cursor-pointer")}>
                            {isPlaying ? pauseIcon : playIcon}
                        </div>
                    )}
                    </div>
                )}
            </div>
        );
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

const playIcon = <svg viewBox="0 0 24 24" width="24" height="24"><path fill="currentColor" d="M8 5v14l11-7z"></path></svg>;
const pauseIcon = <svg viewBox="0 24 24" width="24" height="24"><path fill="currentColor" d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"></path></svg>;

export const Frames = mobxReact.observer(
    function (props) {
        const { state_key, frames } = props
        const $state = useContext($StateContext);

        if (!Array.isArray(frames)) {
            return <div className={tw("text-red-500")}>Error: 'frames' must be an array.</div>;
        }

        const index = $state[state_key];
        if (!Number.isInteger(index) || index < 0 || index >= frames.length) {
            return <div className={tw("text-red-500")}>Error: Invalid index. $state[{state_key}] ({index}) must be a valid index of the frames array (length: {frames.length}).</div>;
        }

        return <Node value={frames[index]} />;
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
export { d3, MarkSpec, Plot, PlotSpec, React, ReactDOM, Row, Column, Grid };

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
                return <React.Fragment>
                    {value.map(item =>
                        typeof item !== 'object' || item === null ? item : <Node value={item} />
                    )}
                </React.Fragment>;
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

function isProps(props) {
    return props?.constructor === Object && !props.__type__ && !React.isValidElement(props);
}

export function Hiccup(tag, props, ...children) {
    const $state = useContext($StateContext)

    if (!isProps(props)) {
        children.unshift(props);
        props = {};
    }

    props = $state.evaluate(props)



    if (props.class) {
        props.className = props.class;
        delete props.class;
    }

    let baseTag = tag;
    if (tag === "<>") {
        baseTag = React.Fragment;
    } else if (typeof tag === 'string') {
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
        ? React.createElement(baseTag, props,
            ...children.map((child, index) => <Node key={index} value={child}/>)
          )
        : React.createElement(baseTag, props);
}
