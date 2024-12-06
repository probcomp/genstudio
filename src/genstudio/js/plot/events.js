import * as Plot from "@observablehq/plot";
import * as d3 from "d3";

import {
  applyIndirectStyles,
  applyTransform,
  calculateScaleFactors,
  invertPoint
} from "./style";

/**
 * A custom mark for mouse interaction on plots.
 * @extends Plot.Mark
 */
export class EventHandler extends Plot.Mark {
  /**
   * Creates a new event handler mark.
   * @param {Object} options - Configuration options for the event handler mark.
   * @param {Function} [options.onDrawStart] - Callback function called when drawing starts. Receives an event object with {type: "drawstart", x, y, startTime, key}.
   * @param {Function} [options.onDraw] - Callback function called during drawing. Receives an event object with {type: "draw", x, y, startTime, key}.
   * @param {Function} [options.onDrawEnd] - Callback function called when drawing ends. Receives an event object with {type: "drawend", x, y, startTime, key}.
   * @param {Function} [options.onMouseMove] - Callback function called when the mouse moves over the drawing area. Receives an event object with {type: "mousemove", x, y, key}.
   * @param {Function} [options.onClick] - Callback function called when the drawing area is clicked. Receives an event object with {type: "click", x, y, key}.
   * @param {Function} [options.onMouseDown] - Callback function called when the mouse button is pressed down. Receives an event object with {type: "mousedown", x, y, startTime, key}.
   */
  constructor(options = {}) {
    super([null], {}, options, {
      ariaLabel: "draw area",
      fill: "none",
      stroke: "none",
      strokeWidth: 1,
      pointerEvents: "all"
    });

    this.onDrawStart = options.onDrawStart;
    this.onDraw = options.onDraw;
    this.onDrawEnd = options.onDrawEnd;
    this.onMouseMove = options.onMouseMove;
    this.onClick = options.onClick;
    this.onMouseDown = options.onMouseDown;
  }

  /**
   * Renders the event handler mark.
   * @param {number} index - The index of the mark.
   * @param {Object} scales - The scales for the plot.
   * @param {Object} channels - The channels for the plot.
   * @param {Object} dimensions - The dimensions of the plot.
   * @param {Object} context - The rendering context.
   * @returns {SVGGElement} The rendered SVG group element.
   */
  render(index, scales, channels, dimensions, context) {
    const { width, height } = dimensions;
    let currentDrawingRect = null;
    let drawingArea = null;
    let scaleFactors;
    let lineStartTime;
    let lineKey;

    const eventData = (eventType, point) => ({
      type: eventType,
      x: point[0],
      y: point[1],
      startTime: lineStartTime,
      key: lineKey
    });

    const isWithinDrawingArea = (rect, x, y) =>
      x >= rect.left && x <= rect.right && y >= rect.top && y <= rect.bottom;

    const handleMouseDown = (event) => {
      const rect = drawingArea.getBoundingClientRect();
      if (!isWithinDrawingArea(rect, event.clientX, event.clientY)) return;

      scaleFactors = calculateScaleFactors(drawingArea.ownerSVGElement);
      const offsetX = event.clientX - rect.left;
      const offsetY = event.clientY - rect.top;
      const point = invertPoint(offsetX, offsetY, scales, scaleFactors);
      this.onMouseDown?.(eventData("mousedown", point));

      if (this.onDrawStart || this.onDraw || this.onDrawEnd) {
        event.preventDefault()
        currentDrawingRect = rect;
        lineStartTime = Date.now();
        lineKey = (Math.round(offsetX * offsetY * lineStartTime) % (2**31));
        this.onDrawStart?.(eventData("drawstart", point));
      }

    };

    const handleDrawEnd = (event) => {
      if (currentDrawingRect) {
        const offsetX = event.clientX - currentDrawingRect.left;
        const offsetY = event.clientY - currentDrawingRect.top;
        const point = invertPoint(offsetX, offsetY, scales, scaleFactors);
        this.onDrawEnd?.(eventData("drawend", point));
        currentDrawingRect = null;
        lineStartTime = null;
        lineKey = null;
      }

    };

    const handleMouseMove = (event) => {

      if (this.onDraw && currentDrawingRect) {
        const offsetX = event.clientX - currentDrawingRect.left;
        const offsetY = event.clientY - currentDrawingRect.top;
        const point = invertPoint(offsetX, offsetY, scales, scaleFactors);
        this.onDraw?.(eventData("draw", point));
      }

      if (this.onMouseMove) {
        const rect = currentDrawingRect || drawingArea.getBoundingClientRect();
        if (!isWithinDrawingArea(rect, event.clientX, event.clientY)) return;
        const offsetX = event.clientX - rect.left;
        const offsetY = event.clientY - rect.top;
        const point = invertPoint(offsetX, offsetY, scales, calculateScaleFactors(drawingArea.ownerSVGElement));
        this.onMouseMove(eventData("mousemove", point));
      }
    };

    const handleClick = (event) => {
      if (this.onClick) {
        const rect = drawingArea.getBoundingClientRect();
        if (!isWithinDrawingArea(rect, event.clientX, event.clientY)) return;
        const offsetX = event.clientX - rect.left;
        const offsetY = event.clientY - rect.top;
        const point = invertPoint(offsetX, offsetY, scales, calculateScaleFactors(drawingArea.ownerSVGElement));
        this.onClick(eventData("click", point));
      }
    };

    const g = d3.create("svg:g")
      .call(applyIndirectStyles, this, dimensions, context)
      .call(applyTransform, this, scales, 0, 0);

    drawingArea = g.append("rect")
      .attr("width", width)
      .attr("height", height)
      .attr("fill", "none")
      .attr("pointer-events", "none")
      .node();

    // Attach all event listeners to document
    document.addEventListener('click', handleClick);
    document.addEventListener('mousedown', handleMouseDown);
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleDrawEnd);

    return g.node();
  }
}

/**
 * Returns a new event handler mark for the given options.
 * @param {Object} _data - Unused parameter (maintained for consistency with other mark functions).
 * @param {EventHandlerOptions} options - Options for the event handler mark.
 * @returns {EventHandler} A new event handler mark.
 */
export function events(_data, options = {}) {
  return new EventHandler(options);
}

/**
 * @typedef {Object} EventHandlerOptions
 * @property {Function} [onDrawStart] - Callback function called when drawing starts. Receives an event object with {type: "drawstart", x, y, startTime}.
 * @property {Function} [onDraw] - Callback function called during drawing. Receives an event object with {type: "draw", x, y, startTime}.
 * @property {Function} [onDrawEnd] - Callback function called when drawing ends. Receives an event object with {type: "drawend", x, y, startTime}.
 * @property {Function} [onMouseMove] - Callback function called when the mouse moves over the drawing area. Receives an event object with {type: "mousemove", x, y}.
 * @property {Function} [onClick] - Callback function called when the drawing area is clicked. Receives an event object with {type: "click", x, y}.
 * @property {Function} [onMouseDown] - Callback function called when the mouse button is pressed down. Receives an event object with {type: "mousedown", x, y, startTime}.
 */
