import * as Plot from "@observablehq/plot";
import * as d3 from "d3";

import {
  applyChannelStyles,
  applyDirectStyles,
  applyIndirectStyles,
  applyTransform
} from "./style";

export class Ellipse extends Plot.Mark {
  /**
   * Creates an Ellipse mark.
   * @param {Object|Array[]} data - The data for the ellipses. Can be:
   *   - An object with columnar data for channels (x, y, rx, ry, rotate)
   *   - An array of arrays, each inner array in one of these formats:
   *     [x, y, r]
   *     [x, y, rx, ry]
   *     [x, y, rx, ry, rotation]
   *   Where:
   *     - x, y: center coordinates
   *     - r: radius (used for both rx and ry if specified)
   *     - rx, ry: x and y radii respectively
   *     - rotation: rotation in degrees (optional)
   * @param {Object} options - Additional options for customizing the ellipses
   */
  constructor(data, options = {}) {
    let { x, y, rx, ry, r, rotate } = options;
    x = x || (d => d?.[0]);
    y = y || (d => d?.[1]);
    ry = ry || rx || r || (d => d ? (d[3] || d[2]) : undefined);
    rx = rx || r || (d => d?.[2]);
    rotate = rotate || (d => d?.[4] || 0);

    super(data, {
      x: { value: x, scale: "x" },
      y: { value: y, scale: "y" },
      rx: { value: rx },
      ry: { value: ry },
      rotate: {value: rotate, optional: true}
    },
      options,
      {
        ariaLabel: "ellipse",
        fill: "currentColor",
        stroke: "none"
      });
  }

  render(index, scales, channels, dimensions, context) {
    let { x: X, y: Y, rx: RX, ry: RY, rotate: ROTATE } = channels;

    return d3.create("svg:g")
      .call(applyIndirectStyles, this, dimensions, context)
      .call(applyTransform, this, scales, 0, 0)
      .call(g => g.selectAll()
        .data(index)
        .join("ellipse")
        .call(applyDirectStyles, this)
        .attr("cx", i => X[i])
        .attr("cy", i => Y[i])
        .attr("rx", i => Math.abs(scales.x(RX[i]) - scales.x(0)))
        .attr("ry", i => Math.abs(scales.y(RY[i]) - scales.y(0)))
        .attr("transform", i => ROTATE ? `rotate(${ROTATE[i]}, ${X[i]}, ${Y[i]})` : null)
        .call(applyChannelStyles, this, channels)
      )
      .node();
  }
}

/**
 * Returns a new ellipse mark for the given *data* and *options*.
 *
 * If neither **x** nor **y** are specified, *data* is assumed to be an array of
 * pairs [[*x₀*, *y₀*], [*x₁*, *y₁*], [*x₂*, *y₂*], …] such that **x** = [*x₀*,
 * *x₁*, *x₂*, …] and **y** = [*y₀*, *y₁*, *y₂*, …].
 */
export function ellipse(data, options = {}) {
  return new Ellipse(data, options);
}

/**
 * @typedef {Object} EllipseOptions
 * @property {ChannelValue} [x] - The x-coordinate of the center of the ellipse.
 * @property {ChannelValue} [y] - The y-coordinate of the center of the ellipse.
 * @property {ChannelValue} [rx] - The x-radius of the ellipse.
 * @property {ChannelValue} [ry] - The y-radius of the ellipse.
 * @property {ChannelValue} [r] - The radius of the ellipse (used for both rx and ry if specified).
 * @property {string} [stroke] - The stroke color of the ellipse.
 * @property {number} [strokeWidth] - The stroke width of the ellipse.
 * @property {string} [fill] - The fill color of the ellipse.
 * @property {number} [fillOpacity] - The fill opacity of the ellipse.
 * @property {number} [strokeOpacity] - The stroke opacity of the ellipse.
 * @property {string} [title] - The title of the ellipse (tooltip).
 */
