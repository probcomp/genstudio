import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { evaluate, evaluateCache, collectReactiveInitialState, useReactiveState, StateProvider, renderData } from '../../src/genstudio/js/widget'
import { React, Plot, ReactDOM } from '../../src/genstudio/js/imports.npm'
import { render, act } from '@testing-library/react'

// Add this at the top of the file
beforeEach(() => {
  vi.useFakeTimers()
})

afterEach(() => {
  vi.useRealTimers()
})

describe('Widget', () => {
  describe('collectReactiveInitialState', () => {
    it('should collect initial state from Reactive components', () => {
      const ast = {
        __type__: 'function',
        path: 'Reactive',
        args: [{ state_key: 'testKey', init: 5 }]
      }

      const initialState = collectReactiveInitialState(ast)
      expect(initialState).toEqual({ testKey: 5 })
    })
  })

  describe('evaluate', () => {
    it('should evaluate a simple ast', () => {
      const ast = {
        __type__: 'function',
        path: 'md',
        args: ['# Hello, World!']
      }

      const result = evaluate(ast, {}, {}, null)
      expect(result).toBeDefined()
      expect(React.isValidElement(result)).toBe(true)
    })

    it('should handle references correctly', () => {
      const ast = {
        __type__: 'ref',
        path: 'Plot.dot'
      }
      const result = evaluate(ast, {}, {}, null)
      expect(result).toBe(Plot.dot)
    })

    it('should evaluate JavaScript expressions', () => {
      const ast = {
        __type__: 'js',
        value: '2 + 2'
      }
      const result = evaluate(ast, {}, {}, null)
      expect(result).toBe(4)
    })

    it('should handle datetime objects', () => {
      const ast = {
        __type__: 'datetime',
        value: '2023-04-01T12:00:00Z'
      }
      const result = evaluate(ast, {}, {}, null)
      expect(result).toBeInstanceOf(Date)
      expect(result.toISOString()).toBe('2023-04-01T12:00:00.000Z')
    })
  })

  describe('evaluateCache', () => {
    it('should evaluate cache entries', () => {
      const cache = {
        key1: { __type__: 'js', value: '1 + 1' },
        key2: { __type__: 'js', value: '$state.value * 2' }
      }
      const $state = { value: 3 }

      const evaluatedCache = evaluateCache(cache, $state, null)
      expect(evaluatedCache).toEqual({
        key1: 2,
        key2: 6
      })
    })

    it('should handle circular references', () => {
      const cache = {
        key1: { __type__: 'js', value: '$state.key2' },
        key2: { __type__: 'js', value: '$state.key1' }
      }
      const $state = { key1: 1, key2: 2 }

      const evaluatedCache = evaluateCache(cache, $state, null)
      expect(evaluatedCache).toEqual({
        key1: 2,
        key2: 1
      })
    })
  })

  describe('useReactiveState', () => {
    it('should initialize state correctly', () => {
      const ast = {
        __type__: 'function',
        path: 'Reactive',
        args: [{ state_key: 'count', init: 0 }]
      }
      let result;
      function TestHook() {
        result = useReactiveState(ast);
        return null;
      }
      render(<TestHook />);
      expect(result).toBeDefined();
      const $state = result;
      expect($state.count).toEqual(0);
    })

    it('should update state correctly', async () => {
      const ast = {
        __type__: 'function',
        path: 'Reactive',
        args: [{ state_key: 'count', init: 0 }]
      }
      let result;
      function TestHook() {
        result = useReactiveState(ast);
        return null;
      }
      render(<TestHook />);
      const $state = result;
      await act(async () => {
        $state.count = 1;
      });

      expect($state.count).toEqual(1)
    })
  })

  describe('StateProvider', () => {
    it('should render a reactive variable in markdown', () => {
      const ast = {
        __type__: 'function',
        path: 'Hiccup',
        args: [
          "div",
          {
            __type__: 'function',
            path: 'Reactive',
            args: [{ state_key: 'count', init: 0 }]
          },
          {
            __type__: 'function',
            path: 'md',
            args: [{
              __type__: 'js',
              value: '`Count: ${$state.count}`'
            }]
          }
        ]
      };
      const cache = {};
      const experimental = null;
      const model = { on: vi.fn(), off: vi.fn() };

      const { container, rerender } = render(
        <StateProvider ast={ast} cache={cache} experimental={experimental} model={model} />
      );

      expect(container.innerHTML).toContain('Count: 0');
    });
  });

  describe('renderData', () => {
    it('should render data correctly', async () => {
      const container = document.createElement('div')
      const data = { ast: { __type__: 'function', path: 'md', args: ['# Test'] } }

      await act(async () => {
        renderData(container, data)
      })

      expect(container.innerHTML).toContain('Test')
    })
  })
})
