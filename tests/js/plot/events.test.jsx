import { describe, it, expect, vi } from 'vitest'
import { EventHandler, events } from '../../../src/genstudio/js/plot/events'
import * as Plot from "@observablehq/plot";
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
    const drawMark = new EventHandler()

    expect(drawMark).toBeInstanceOf(EventHandler)
    expect(drawMark).toBeInstanceOf(Plot.Mark)
  })

  it('should render a drawing area', () => {
    const drawMark = new EventHandler()
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

    const drawMark = new EventHandler({ onDrawStart, onDraw, onDrawEnd })

    expect(drawMark.onDrawStart).toBe(onDrawStart)
    expect(drawMark.onDraw).toBe(onDraw)
    expect(drawMark.onDrawEnd).toBe(onDrawEnd)
  })
})

describe('draw function', () => {
  it('should return a Draw instance', () => {
    const result = events({})

    expect(result).toBeInstanceOf(EventHandler)
  })
})
