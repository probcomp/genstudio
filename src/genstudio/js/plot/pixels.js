import * as Plot from "@observablehq/plot";
import * as d3 from "d3";

import {
  applyChannelStyles,
  applyDirectStyles,
  applyIndirectStyles,
  applyTransform
} from "./style";

/**
 * A custom mark for rendering RGB pixel data efficiently using canvas.
 * @extends Plot.Mark
 */
export class Pixels extends Plot.Mark {
  /**
   * A custom mark for efficiently rendering a single image from raw RGB(A) pixel data.
   * Unlike most Observable Plot marks which render multiple elements from data arrays,
   * this mark renders a single image from a flat array of pixel values.
   *
   * @param {Object|Array} pixelData - Raw pixel data as a flat array in either RGB format [r,g,b,r,g,b,...]
   *                                   or RGBA format [r,g,b,a,r,g,b,a,...]. Each value should be 0-255.
   * @param {Object} options - Configuration options
   * @param {number} options.imageWidth - Width of the source image in pixels
   * @param {number} options.imageHeight - Height of the source image in pixels
   * @param {ChannelValue} [options.x=0] - X coordinate of top-left corner in plot coordinates
   * @param {ChannelValue} [options.y=0] - Y coordinate of top-left corner in plot coordinates
   * @param {ChannelValue} [options.width] - Displayed width in plot coordinates (defaults to imageWidth)
   * @param {ChannelValue} [options.height] - Displayed height in plot coordinates (defaults to imageHeight)
   */
  constructor(pixelData, {x = 0, y = 0, width, height, imageWidth, imageHeight, ...options }) {
    if (typeof imageWidth !== 'number' || typeof imageHeight !== 'number') {
      throw new Error("imageWidth and imageHeight must be specified as numbers");
    }

    width = width || imageWidth;
    height = height || imageHeight;

    const channels = {
      x: { value: "0", scale: "x" },
      y: { value: "1", scale: "y" }
    };

    const data = [[x, y], [x + width, y + height]];
    super(data, channels, options, { ariaLabel: "pixel image" });

    this.imageWidth = imageWidth;
    this.imageHeight = imageHeight;
    this.pixelData = pixelData;
  }

  getChannelValue(channel, dataPoint, index) {
    if (typeof channel === 'function') {
      return channel(dataPoint, index);
    }
    if (typeof channel === 'string') {
      return dataPoint[channel];
    }
    return channel[index];
  }

  render(index, scales, channels, dimensions, context) {
    const { pixelData, imageWidth, imageHeight } = this;

    const [x1, x2] = channels.x
    const [y1, y2] = channels.y
    const width = Math.abs(x1 - x2)
    const height = Math.abs(y1 - y2)

    // Return empty SVG if width or height is 0
    if (width === 0 || height === 0) {
      return d3.create("svg:g").node();
    }

    const canvas = document.createElement("canvas");
    canvas.width = imageWidth;
    canvas.height = imageHeight;
    const ctx = canvas.getContext("2d");
    const imageData = ctx.createImageData(imageWidth, imageHeight);
    const data = imageData.data;

    // Check if alpha channel is provided (4 values per pixel instead of 3)
    const hasAlpha = pixelData.length === imageWidth * imageHeight * 4;
    const stride = hasAlpha ? 4 : 3;

    // Fill from flat array of [r,g,b,r,g,b,...] or [r,g,b,a,r,g,b,a,...]
    for (let i = 0; i < pixelData.length; i += stride) {
      const idx = (i / stride) * 4;

      // Get RGB values from flat array
      data[idx] = pixelData[i];       // Red
      data[idx + 1] = pixelData[i+1]; // Green
      data[idx + 2] = pixelData[i+2]; // Blue
      data[idx + 3] = hasAlpha ? pixelData[i+3] : 255; // Alpha
    }

    ctx.putImageData(imageData, 0, 0);

    // Check if browser is Safari
    const isSafari = /^((?!chrome|android).)*safari/i.test(navigator.userAgent);

    const transform = `translate(${x1},${y2}) scale(${width/imageWidth},${height/imageHeight})`

    const g = d3.create("svg:g")
      .call(applyIndirectStyles, this, dimensions, context)
      .call(applyTransform, this, scales, 0, 0);

    if (isSafari) {
      // For Safari, use image element with data URL
      const dataUrl = canvas.toDataURL();
      g.selectAll()
        .data([0])
        .join("image")
        .style("transform-origin", "0 0")
        .call(applyDirectStyles, this)
        .attr("transform", transform)
        .attr("width", imageWidth)
        .attr("height", imageHeight)
        .attr("href", dataUrl)
        .call(applyChannelStyles, this, channels);
    } else {
      // For other browsers, use foreignObject with canvas for better performance
      g.selectAll()
        .data([0])
        .join("foreignObject")
        .style("transform-origin", "0 0")
        .call(applyDirectStyles, this)
        .attr("transform", transform)
        .attr("width", imageWidth)
        .attr("height", imageHeight)
        .call(applyChannelStyles, this, channels)
        .append(() => canvas);
    }

    return g.node();
  }
}

/**
 * Returns a new pixel image mark for the given data and options.
 * @param {Object|Array} data - The data containing RGB values as a flat array of (r,g,b) tuples
 * @param {Object} options - Options for customizing the pixel image
 * @returns {Pixels} A new Pixels mark
 */
export function pixels(data, options = {}) {
  return new Pixels(data, options);
}

/**
 * @typedef {(number|string|function)} ChannelValue
 */
