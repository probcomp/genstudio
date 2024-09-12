import { Plot, d3 } from "./imports"
import {
  applyIndirectStyles,
  applyTransform
} from "./plot_style";

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
    let isDrawing = false;
    let path = [];

    // Helper function to invert point with logging
    const invertPoint = (x, y) => {
      return [scales.x.invert(x), scales.y.invert(y)];
    };

    // Helper function to create a payload for callbacks
    const createPayload = (x, y, type) => {
      const [invertedX, invertedY] = invertPoint(x, y);
      return {
        x: invertedX,
        y: invertedY,
        pixels: { x, y },
        type
      };
    };

    // Helper function to create path payload
    const createPathPayload = () => path.map(([x, y]) => invertPoint(x, y));

    const handleDrawStart = (event) => {
      isDrawing = true;
      const { clientX, clientY } = event;
      const rect = event.target.getBoundingClientRect();
      const offsetX = clientX - rect.left;
      const offsetY = clientY - rect.top;
      path = [[offsetX, offsetY]];
      if (this.onDrawStart) {
        this.onDrawStart(createPayload(offsetX, offsetY, "drawstart"));
      }

      // Set up global listeners
      document.addEventListener('mousemove', handleDraw);
      document.addEventListener('mouseup', handleDrawEnd);
    };

    const handleDraw = (event) => {
      if (!isDrawing) return;
      event.preventDefault()
      const rect = event.target.getBoundingClientRect();
      const offsetX = event.clientX - rect.left;
      const offsetY = event.clientY - rect.top;
      path.push([offsetX, offsetY]);
      if (this.onDraw) {
        return this.onDraw({
          ...createPayload(offsetX, offsetY, "draw"),
          path: createPathPayload()
        });
      }
    };

    const handleDrawEnd = (event) => {
      if (!isDrawing) return;
      isDrawing = false;
      const rect = event.target.getBoundingClientRect();
      const offsetX = event.clientX - rect.left;
      const offsetY = event.clientY - rect.top;
      path.push([offsetX, offsetY]);
      if (this.onDrawEnd) {
        this.onDrawEnd({
          ...createPayload(offsetX, offsetY, "drawend"),
          path: createPathPayload()
        });
      }

      // Remove global listeners
      document.removeEventListener('mousemove', handleDraw);
      document.removeEventListener('mouseup', handleDrawEnd);
    };

    return d3.create("svg:g")
      .call(applyIndirectStyles, this, dimensions, context)
      .call(applyTransform, this, scales, 0, 0)
      .call(g => g.append("rect")
        .attr("width", width)
        .attr("height", height)
        .attr("fill", "none")
        .attr("pointer-events", "all")
        .on("mousedown", handleDrawStart)
      )
      .node();
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
