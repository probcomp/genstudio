import { describe, it, expect, vi } from 'vitest'
import { Draw, draw } from '../../../src/genstudio/js/plot/draw'
import { Plot, d3 } from '../../../src/genstudio/js/imports.npm'
import { JSDOM } from 'jsdom'

describe('Draw', () => {
  let document;
  let window;

  beforeEach(() => {
    const dom = new JSDOM('<!DOCTYPE html><html><body></body></html>');
    window = dom.window;
    document = window.document;
    global.document = document;
    global.window = window;
  });

  it('should create a Draw instance', () => {
    const drawMark = new Draw()

    expect(drawMark).toBeInstanceOf(Draw)
    expect(drawMark).toBeInstanceOf(Plot.Mark)
  })

  it('should render a drawing area', () => {
    const drawMark = new Draw()
    const mockScales = { x: vi.fn(x => x), y: vi.fn(y => y) }

    const result = drawMark.render([0], mockScales, {}, { width: 500, height: 300 }, {})

    expect(result.tagName).toBe('g')
    const rect = result.querySelector('rect')
    expect(rect).not.toBeNull()
    expect(rect.getAttribute('width')).toBe('500')
    expect(rect.getAttribute('height')).toBe('300')
    expect(rect.getAttribute('fill')).toBe('none')
    expect(rect.getAttribute('pointer-events')).toBe('all')
  })

  it('should have callback properties', () => {
    const onDrawStart = vi.fn()
    const onDraw = vi.fn()
    const onDrawEnd = vi.fn()

    const drawMark = new Draw({ onDrawStart, onDraw, onDrawEnd })

    expect(drawMark.onDrawStart).toBe(onDrawStart)
    expect(drawMark.onDraw).toBe(onDraw)
    expect(drawMark.onDrawEnd).toBe(onDrawEnd)
  })
})

describe('draw function', () => {
  it('should return a Draw instance', () => {
    const result = draw({})

    expect(result).toBeInstanceOf(Draw)
  })
})
