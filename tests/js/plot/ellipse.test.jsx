import { describe, it, expect, vi } from 'vitest'
import { Ellipse, ellipse } from '../../../src/genstudio/js/plot/ellipse'
import * as Plot from "@observablehq/plot";

describe('Ellipse', () => {
  it('should create an Ellipse instance', () => {
    const data = [{ x: 0, y: 0, rx: 50, ry: 30 }]
    const ellipseMark = new Ellipse(data, { x: 'x', y: 'y', rx: 'rx', ry: 'ry' })

    expect(ellipseMark).toBeInstanceOf(Ellipse)
    expect(ellipseMark).toBeInstanceOf(Plot.Mark)
  })

  it('should render ellipses correctly', () => {
    const data = [{ x: 100, y: 100, rx: 50, ry: 30, rotate: 45 }]
    const ellipseMark = new Ellipse(data, { x: 'x', y: 'y', rx: 'rx', ry: 'ry', rotate: 'rotate' })

    const mockScales = { x: vi.fn(x => x), y: vi.fn(y => y) }
    const mockChannels = {
      x: [100], y: [100], rx: [50], ry: [30], rotate: [45]
    }

    const result = ellipseMark.render([0], mockScales, mockChannels, { width: 500, height: 300 }, {})

    expect(result.tagName).toBe('g')
    const ellipse = result.querySelector('ellipse')
    expect(ellipse).not.toBeNull()
    expect(ellipse.getAttribute('cx')).toBe('100')
    expect(ellipse.getAttribute('cy')).toBe('100')
    expect(ellipse.getAttribute('rx')).toBe('50')
    expect(ellipse.getAttribute('ry')).toBe('30')
    expect(ellipse.getAttribute('transform')).toBe('rotate(45, 100, 100)')
  })

  it('should handle array input correctly', () => {
    const data = [[100, 100, 50, 30, 45]]
    const ellipseMark = new Ellipse(data)

    const mockScales = { x: vi.fn(x => x), y: vi.fn(y => y) }
    const mockChannels = {
      x: [100], y: [100], rx: [50], ry: [30], rotate: [45]
    }

    const result = ellipseMark.render([0], mockScales, mockChannels, { width: 500, height: 300 }, {})

    const ellipse = result.querySelector('ellipse')
    expect(ellipse).not.toBeNull()
    expect(ellipse.getAttribute('cx')).toBe('100')
    expect(ellipse.getAttribute('cy')).toBe('100')
    expect(ellipse.getAttribute('rx')).toBe('50')
    expect(ellipse.getAttribute('ry')).toBe('30')
    expect(ellipse.getAttribute('transform')).toBe('rotate(45, 100, 100)')
  })
})

describe('ellipse function', () => {
  it('should return an Ellipse instance', () => {
    const data = [{ x: 0, y: 0, rx: 50, ry: 30 }]
    const result = ellipse(data, { x: 'x', y: 'y', rx: 'rx', ry: 'ry' })

    expect(result).toBeInstanceOf(Ellipse)
  })
})
