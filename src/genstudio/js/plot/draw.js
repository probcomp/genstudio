import { Plot, d3 } from "../imports"
import {
  applyIndirectStyles,
  applyTransform
} from "./style";

export class Draw extends Plot.Mark {
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

  render(index, scales, channels, dimensions, context) {
    const { width, height } = dimensions;
    let path = [];
    let currentDrawingRect = null;
    let drawingArea = null;
    let scaleX, scaleY;

    // Calculate scale factors to account for differences between
    // SVG logical dimensions and actual rendered size
    const calculateScaleFactors = (rect) => {
      scaleX = rect.width / width;
      scaleY = rect.height / height;
    };

    // Convert pixel coordinates to data coordinates
    const invertPoint = (x, y) => [
      scales.x.invert(x / scaleX),
      scales.y.invert(y / scaleY)
    ];

    const eventData = (eventType, path) => ({ type: eventType, path: [...path] });

    const isWithinDrawingArea = (rect, x, y) =>
      x >= rect.left && x <= rect.right && y >= rect.top && y <= rect.bottom;

    const handleDrawStart = (event) => {
      currentDrawingRect = drawingArea.getBoundingClientRect();
      if (!isWithinDrawingArea(currentDrawingRect, event.clientX, event.clientY)) return;

      calculateScaleFactors(currentDrawingRect);
      const offsetX = event.clientX - currentDrawingRect.left;
      const offsetY = event.clientY - currentDrawingRect.top;
      path = [invertPoint(offsetX, offsetY)];
      this.onDrawStart?.(eventData("drawstart", path));

      document.addEventListener('mousemove', handleDraw);
      document.addEventListener('mouseup', handleDrawEnd);
    };

    const handleDraw = (event) => {
      if (!currentDrawingRect) return;
      event.preventDefault();
      const offsetX = event.clientX - currentDrawingRect.left;
      const offsetY = event.clientY - currentDrawingRect.top;
      path.push(invertPoint(offsetX, offsetY));
      this.onDraw?.(eventData("draw", path));
    };

    const handleDrawEnd = (event) => {
      if (!currentDrawingRect) return;
      const offsetX = event.clientX - currentDrawingRect.left;
      const offsetY = event.clientY - currentDrawingRect.top;
      path.push(invertPoint(offsetX, offsetY));
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
 * @param {Object} options - Options for the draw mark
 * @param {Function} [options.onDrawStart] - Callback function called when drawing starts
 * @param {Function} [options.onDraw] - Callback function called during drawing
 * @param {Function} [options.onDrawEnd] - Callback function called when drawing ends
 * @returns {Draw} A new Draw mark
 */
export function draw(_data, options = {}) {
  return new Draw(options);
}

/**
 * @typedef {Object} DrawOptions
 * @property {Function} [onDrawStart] - Callback function called when drawing starts
 * @property {Function} [onDraw] - Callback function called during drawing
 * @property {Function} [onDrawEnd] - Callback function called when drawing ends
 */
