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
        __type__: "js_ref",
        path: 'Plot.dot'
      }
      const result = evaluate(ast, {}, {}, null)
      expect(result).toBe(Plot.dot)
    })

    it('should evaluate JavaScript expressions', () => {
      const ast = {
        __type__: "js_source",
        expression: true,
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
      const init = {"count": 0}
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
        path: 'InitialState',
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
              __type__: "js_source",
              expression: true,
              value: '`Count: ${$state.count}`'
            }]
          }
        ]
      };
      const cache = {"count": 0};
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
      const data = { ast: { __type__: 'function', path: 'md', args: ['# Test'] }, cache: {} }

      await act(async () => {
        renderData(container, data)
      })

      expect(container.innerHTML).toContain('Test')
    })
  })

  describe('Plot.InitialState and Plot.js combination', () => {
    it('should handle Plot.InitialState and Plot.js combination correctly', async () => {
      const consoleSpy = vi.spyOn(console, 'log');

      // Simulate the AST created by Python's `&` operator
      const ast = {
        __type__: 'function',
        path: 'Row',
        args: [
          {}, // options object for Row
          {
            __type__: 'function',
            path: 'InitialState',
            args: ['foo', {__type__: 'ref', id: 'foo'}]
          },
          {__type__: "js_source", value: 'console.log($state.foo) || $state.foo'}
        ]
      };

      const cache = {
        'foo': 123
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

  describe('createStateStore', () => {
    it('should initialize with basic values', () => {
      const store = createStateStore({ count: 0, name: 'Test' });
      expect(store.count).toBe(0);
      expect(store.name).toBe('Test');
    });

    it('should update values', () => {
      const store = createStateStore({ count: 0 });
      store.count = 1;
      expect(store.count).toBe(1);
    });

    it('should handle computed values', () => {
      const store = createStateStore({
        count: 0,
        doubleCount: { __type__: 'js_source', expression: true, value: '$state.count * 2' }
      });
      expect(store.doubleCount).toBe(0);
      store.count = 2;
      expect(store.doubleCount).toBe(4);
    });

    it('should handle references', () => {
      const store = createStateStore({
        original: 10,
        reference: { __type__: 'ref', id: 'original' }
      });
      expect(store.reference).toBe(10);
      store.original = 20;
      expect(store.reference).toBe(20);
    });

    it('should handle complex updates', () => {
      const store = createStateStore({
        list: [1, 2, 3],
        sum: { __type__: 'js_source', expression: true, value: '$state.list.reduce((a, b) => a + b, 0)' }
      });
      expect(store.sum).toBe(6);
      store.update([['list', 'append', 4]]);
      expect(store.list).toEqual([1, 2, 3, 4]);
      expect(store.sum).toBe(10);
    });

    it('should handle circular references without infinite loops', () => {
      const store = createStateStore({
        a: { __type__: 'ref', id: 'b' },
        b: { __type__: 'ref', id: 'a' },
        c: 10
      });
      expect(() => store.a).toThrow(/Cycle detected in computation/);
      expect(store.c).toBe(10);
    });

    it('should demonstrate that update order matters for dependent variables', () => {
      const store = createStateStore({
        a: 1,
        b: { __type__: 'js_source', expression: true, value: '$state.a + 1' }
      });

      // Initial state
      expect(store.a).toBe(1);
      expect(store.b).toBe(2);

      // Update 'a' first, then 'b'
      store.update([
        ['a', 'reset', 10],
        ['b', 'reset', { __type__: 'js_source', expression: true, value: '$state.a + 1' }]
      ]);
      expect(store.a).toBe(10);
      expect(store.b).toBe(11);

      // Reset the store
      store.a = 1;

      // Update 'b' first, then 'a'
      store.update([
        ['b', 'reset', { __type__: 'js_source', expression: true, value: '$state.a + 1' }],
        ['a', 'reset', 10]
      ]);
      expect(store.a).toBe(10);
      expect(store.b).toBe(2);  // 'b' is still based on the old value of 'a'
    });

  });
})
