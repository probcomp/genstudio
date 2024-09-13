import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { evaluate, createStateStore, StateProvider, renderData } from '../../src/genstudio/js/widget'
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

  describe('evaluate', () => {
    it('should evaluate a simple ast', () => {
      const ast = {
        __type__: 'function',
        path: 'md',
        args: ['# Hello, World!']
      }

      const result = evaluate(ast, {}, null)
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

  describe('useStateStore', () => {
    it('should initialize state correctly', () => {
      const init = {"$state.count": 0}
      let result;
      function TestHook() {
        result = createStateStore(init);
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
        result = createStateStore(ast);
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
            path: 'md',
            args: [{
              __type__: 'js',
              value: '`Count: ${$state.count}`'
            }]
          }
        ]
      };
      const cache = {"$state.count": 0};
      const experimental = null;
      const model = { on: vi.fn(), off: vi.fn() };

      const { container, rerender } = render(
        <StateProvider ast={ast} cache={cache} experimental={experimental} model={model} />
      );

      expect(container.innerHTML).toContain('Count: 0');
    });

    it('should update cache and $state simultaneously', async () => {
      const ast = {
        __type__: 'function',
        path: 'Hiccup',
        args: [
          "div",
          {
            __type__: 'function',
            path: 'md',
            args: [{
              __type__: 'js',
              value: '`Count: ${$state.count}, Cached: ${$state.cached("testKey")}`'
            }]
          }
        ]
      };
      const cache = { testKey: 'initial', '$state.count': 0 };
      const experimental = null;
      const model = {
        on: vi.fn(),
        off: vi.fn(),
        trigger: vi.fn()
      };

      const { container } = render(
        <StateProvider ast={ast} cache={cache} experimental={experimental} model={model} />
      );

      expect(container.innerHTML).toContain('Count: 0, Cached: initial');

      // Simulate updating both cache and $state
      await act(async () => {
        const updateMsg = {
          type: 'update_cache',
          updates: JSON.stringify([
            ['testKey', 'reset', 'updated'],
            ['$state.count', 'reset', 1]
          ])
        };
        model.on.mock.calls[0][1](updateMsg);
      });

      expect(container.innerHTML).toContain('Count: 1, Cached: updated');
    });
  });

  describe('renderData', () => {
    it('should render data correctly', async () => {
      const container = document.createElement('div')
      const data = { ast: { __type__: 'function', path: 'md', args: ['# Test'] }, cache: {} }

      await act(async () => {
        renderData(container, data)
      })

      expect(container.innerHTML).toContain('Test')
    })
  })

  describe('Plot.Reactive and Plot.js combination', () => {
    it('should handle Plot.Reactive and Plot.js combination correctly', async () => {
      const consoleSpy = vi.spyOn(console, 'log');

      // Simulate the AST created by Python's `&` operator
      const ast = {
        __type__: 'function',
        path: 'Row',
        args: [
          {}, // options object for Row
          {
            __type__: 'function',
            path: 'Reactive',
            args: ['foo', {__type__: 'cached', id: '$state.foo'}]
          },
          {__type__: "js", value: 'console.log($state.foo) || $state.foo'}
        ]
      };

      const cache = {
        '$state.foo': 123
      };


      render(
        <StateProvider ast={ast} cache={cache} />
      );
      // Check that console.log was called with the correct value
      expect(consoleSpy).toHaveBeenCalledWith(123);
      expect(consoleSpy).toHaveBeenCalledTimes(1);

      consoleSpy.mockRestore();
    });
  });
})
