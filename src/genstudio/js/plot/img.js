import * as Plot from "@observablehq/plot";
import * as d3 from "d3";

import {
  applyChannelStyles,
  applyDirectStyles,
  applyIndirectStyles,
  applyTransform
} from "./style";

const resolveChannel = (channel, data, index) => {
  if (typeof channel === 'number') return channel;
  if (typeof channel === 'string') return data[channel];
  if (typeof channel === 'function') return channel(data, index);
  throw new Error(`Invalid channel type: ${typeof channel}`);
};

const addChannels = (c1, c2) => {
  if (typeof c1 === 'number' && typeof c2 === 'number') {
    return c1 + c2;
  }
  return (data, index) =>
    resolveChannel(c1, data, index) + resolveChannel(c2, data, index);
};

/**
 * A custom mark for rendering images on plots.
 * @extends Plot.Mark
 */
export class Img extends Plot.Mark {
  /**
   * Creates a new Img mark.
   * @param {Object|Array} data - The data for the images.
   * @param {Object} options - Configuration options for the Img mark.
   * @param {ChannelValue} [options.src] - The source path of the image.
   * @param {ChannelValue} [options.x=0] - The x-coordinate of the top-left corner.
   * @param {ChannelValue} [options.y=0] - The y-coordinate of the top-left corner.
   * @param {ChannelValue} [options.width] - The width of the image in x-scale units.
   * @param {ChannelValue} [options.height] - The height of the image in y-scale units.
   * @param {ChannelValue} [options.ariaLabel='image'] - Custom aria-label for accessibility.
   */
  constructor(data, options = {}) {
    const { src, x = 0, y = 0, width, height, ariaLabel = 'image' } = options;

    if (width === undefined || height === undefined) {
      throw new Error("Both width and height must be specified for the Img mark.");
    }

    const channels = {
      src: { value: src },
      x1: { value: x, scale: "x" },
      y1: { value: y, scale: "y" },
      x2: { value: addChannels(x, width), scale: "x" },
      y2: { value: addChannels(y, height), scale: "y" }
    }
    super(data, channels, options, {ariaLabel});
  }

  render(index, scales, channels, dimensions, context) {
    const { src: SRC, x1: X1, y1: Y1, x2: X2, y2: Y2 } = channels;
    return d3.create("svg:g")
      .call(applyIndirectStyles, this, dimensions, context)
      .call(applyTransform, this, scales, 0, 0)
      .call(g => g.selectAll()
        .data(index)
        .join("image")
        .call(applyDirectStyles, this)
        .attr("xlink:href", i => SRC[i])
        .attr("x", i => X1[i])
        .attr("y", i => Y2[i])
        .attr("width", i => Math.abs(X2[i] - X1[i]))
        .attr("height", i => Math.abs(Y2[i] - Y1[i]))
        .call(applyChannelStyles, this, channels)
      )
      .node();
  }
}

/**
 * Returns a new image mark for the given data and options.
 *
 * This image mark differs from the built-in Observable Plot image mark in that
 * it accepts width and height in x/y scale units rather than pixels.
 *
 * @param {Object|Array} data - The data for the images.
 * @param {Object} options - Options for customizing the images.
 * @param {ChannelValue} [options.src] - The source path of the image.
 * @param {ChannelValue} [options.x] - The x-coordinate of the top-left corner.
 * @param {ChannelValue} [options.y] - The y-coordinate of the top-left corner.
 * @param {ChannelValue} [options.width] - The width of the image in x-scale units.
 * @param {ChannelValue} [options.height] - The height of the image in y-scale units.
 * @returns {Img} A new Image mark.
 */
export function img(data, options = {}) {
  return new Img(data, options);
}

/**
 * @typedef {(number|string|function)} ChannelValue
 */
