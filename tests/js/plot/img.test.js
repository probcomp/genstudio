import { describe, it, expect, vi } from 'vitest'
import { Img, img } from '../../../src/genstudio/js/plot/img'
import * as Plot from "@observablehq/plot";

describe('Img', () => {
  it('should create an Img instance', () => {
    const data = [{ src: 'image.jpg', x: 0, y: 0, width: 100, height: 100 }]
    const imgMark = new Img(data, { src: 'src', x: 'x', y: 'y', width: 'width', height: 'height' })

    expect(imgMark).toBeInstanceOf(Img)
    expect(imgMark).toBeInstanceOf(Plot.Mark)
  })

  it('should throw an error if width or height is not specified', () => {
    const data = [{ src: 'image.jpg', x: 0, y: 0 }]
    expect(() => new Img(data, { src: 'src', x: 'x', y: 'y' })).toThrow('Both width and height must be specified for the Img mark.')
  })

  it('should render images correctly', () => {
    const data = [{ src: 'image1.jpg', x: 0, y: 0, width: 100, height: 100 }]
    const imgMark = new Img(data, { src: 'src', x: 'x', y: 'y', width: 'width', height: 'height' })

    const mockScales = { x: vi.fn(x => x), y: vi.fn(y => y) }
    const mockChannels = {
      src: ['image1.jpg'],
      x1: [0], y1: [0], x2: [100], y2: [100],
      ariaLabel: [undefined]
    }

    const result = imgMark.render([0], mockScales, mockChannels, { width: 500, height: 300 }, {})

    expect(result.tagName).toBe('g')
    const image = result.querySelector('image')
    expect(image).not.toBeNull()
    expect(image.getAttribute('href')).toBe('image1.jpg')
    expect(image.getAttribute('x')).toBe('0')
    expect(image.getAttribute('y')).toBe('100')
    expect(image.getAttribute('width')).toBe('100')
    expect(image.getAttribute('height')).toBe('100')
  })
})

describe('img function', () => {
  it('should return an Img instance', () => {
    const data = [{ src: 'image.jpg', x: 0, y: 0, width: 100, height: 100 }]
    const result = img(data, { src: 'src', x: 'x', y: 'y', width: 'width', height: 'height' })

    expect(result).toBeInstanceOf(Img)
  })
})
