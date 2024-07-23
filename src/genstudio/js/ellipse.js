import { Plot, d3 } from "./imports"

import {
  applyChannelStyles,
  applyDirectStyles,
  applyIndirectStyles,
  applyTransform
} from "./plot_style";

export const first = (x) => (x ? x[0] : undefined);
export const second = (x) => (x ? x[1] : undefined);
export function maybeTuple(x, y) {
  return x === undefined && y === undefined ? [first, second] : [x, y];
}
export function maybeNumberChannel(value, defaultValue) {
  if (value === undefined) value = defaultValue;
  return value === null || typeof value === "number" ? [undefined, value] : [value, undefined];
}

const defaults = {
  ariaLabel: "ellipse",
  fill: "currentColor",
  stroke: "none"
};

export class Ellipse extends Plot.Mark {
  constructor(data, options = {}) {
    let { x, y, rx, ry, r, rotate } = options;

    [x, y] = maybeTuple(x, y)
    rx = r ?? rx
    ry = ry ?? rx

    super(data, {
      x: { value: x, scale: "x" },
      y: { value: y, scale: "y" },
      rx: { value: rx },
      ry: { value: ry },
      rotate: {value: rotate, optional: true}
    },
      options,
      defaults);
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
