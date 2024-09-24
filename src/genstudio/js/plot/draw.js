import * as Plot from "@observablehq/plot";
import * as d3 from "d3";

import {
  applyIndirectStyles,
  applyTransform,
  calculateScaleFactors,
  invertPoint
} from "./style";

/**
 * A custom mark for interactive drawing on plots.
 * @extends Plot.Mark
 */
export class Draw extends Plot.Mark {
  /**
   * Creates a new Draw mark.
   * @param {Object} options - Configuration options for the Draw mark.
   * @param {Function} [options.onDrawStart] - Callback function called when drawing starts.
   * @param {Function} [options.onDraw] - Callback function called during drawing.
   * @param {Function} [options.onDrawEnd] - Callback function called when drawing ends.
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
  }

  /**
   * Renders the Draw mark.
   * @param {number} index - The index of the mark.
   * @param {Object} scales - The scales for the plot.
   * @param {Object} channels - The channels for the plot.
   * @param {Object} dimensions - The dimensions of the plot.
   * @param {Object} context - The rendering context.
   * @returns {SVGGElement} The rendered SVG group element.
   */
  render(index, scales, channels, dimensions, context) {
    const { width, height } = dimensions;
    let path = [];
    let currentDrawingRect = null;
    let drawingArea = null;
    let scaleFactors;

    const eventData = (eventType, path) => ({ type: eventType, path: [...path] });

    const isWithinDrawingArea = (rect, x, y) =>
      x >= rect.left && x <= rect.right && y >= rect.top && y <= rect.bottom;

    const handleDrawStart = (event) => {
      currentDrawingRect = drawingArea.getBoundingClientRect();
      if (!isWithinDrawingArea(currentDrawingRect, event.clientX, event.clientY)) return;

      scaleFactors = calculateScaleFactors(drawingArea.ownerSVGElement);
      const offsetX = event.clientX - currentDrawingRect.left;
      const offsetY = event.clientY - currentDrawingRect.top;
      path = [invertPoint(offsetX, offsetY, scales, scaleFactors)];
      this.onDrawStart?.(eventData("drawstart", path));

      document.addEventListener('mousemove', handleDraw);
      document.addEventListener('mouseup', handleDrawEnd);
    };

    const handleDraw = (event) => {
      if (!currentDrawingRect) return;
      event.preventDefault();
      const offsetX = event.clientX - currentDrawingRect.left;
      const offsetY = event.clientY - currentDrawingRect.top;
      path.push(invertPoint(offsetX, offsetY, scales, scaleFactors));
      this.onDraw?.(eventData("draw", path));
    };

    const handleDrawEnd = (event) => {
      if (!currentDrawingRect) return;
      const offsetX = event.clientX - currentDrawingRect.left;
      const offsetY = event.clientY - currentDrawingRect.top;
      path.push(invertPoint(offsetX, offsetY, scales, scaleFactors));
      this.onDrawEnd?.(eventData("drawend", path));

      document.removeEventListener('mousemove', handleDraw);
      document.removeEventListener('mouseup', handleDrawEnd);
      currentDrawingRect = null;
    };

    const g = d3.create("svg:g")
      .call(applyIndirectStyles, this, dimensions, context)
      .call(applyTransform, this, scales, 0, 0);

    drawingArea = g.append("rect")
      .attr("width", width)
      .attr("height", height)
      .attr("fill", "none")
      .attr("pointer-events", "all")
      .node();

    document.addEventListener('mousedown', handleDrawStart);

    return g.node();
  }
}

/**
 * Returns a new draw mark for the given options.
 * @param {Object} _data - Unused parameter (maintained for consistency with other mark functions).
 * @param {DrawOptions} options - Options for the draw mark.
 * @returns {Draw} A new Draw mark.
 */
export function draw(_data, options = {}) {
  return new Draw(options);
}

/**
 * @typedef {Object} DrawOptions
 * @property {Function} [onDrawStart] - Callback function called when drawing starts.
 * @property {Function} [onDraw] - Callback function called during drawing.
 * @property {Function} [onDrawEnd] - Callback function called when drawing ends.
 */
