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

    it('should resolve a js reference', () => {
      const ast = {
        __type__: "js_ref",
        path: 'Plot.dot'
      }
      const result = evaluate(ast, {}, {}, null)
      expect(result).toBe(Plot.dot)
    })

    it('should evaluate a js expression', () => {
      const ast = {
        __type__: "js_source",
        expression: true,
        value: '2 + 2'
      }
      const result = evaluate(ast, {}, {}, null)
      expect(result).toBe(4)
    })

    it('should evaluate a multi-line js source (requires explicit return)', () => {
      const ast = {
        __type__: "js_source",
        value: 'let x = 0\n x = 1\n return x'
      }
      const result = evaluate(ast, {}, {}, null)
      expect(result).toBe(1)
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
    it('should initialize state', () => {
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

  describe('state and Plot.js combination', () => {
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

  const js_expr = (expr) => {return {__type__: 'js_source', expression: true, value: expr}}

  describe('createStateStore', () => {
    it('should initialize with basic values', () => {
      const $state = createStateStore({ count: 0, name: 'Test' });
      expect($state.count).toBe(0);
      expect($state.name).toBe('Test');
    });

    it('should update a value', () => {
      const $state = createStateStore({ count: 0 });
      $state.count = 1;
      expect($state.count).toBe(1);
    });

    it('should handle computed values', () => {
      const $state = createStateStore({
        count: 0,
        doubleCount: js_expr('$state.count * 2')
      });
      expect($state.doubleCount).toBe(0);
      $state.count = 2;
      expect($state.doubleCount).toBe(4);
    });

    it('should handle references', () => {
      const $state = createStateStore({
        original: 10,
        reference: { __type__: 'ref', id: 'original' }
      });
      expect($state.reference).toBe(10);
      $state.original = 20;
      expect($state.reference).toBe(20);
    });

    it('should apply "append" operation', () => {
      const $state = createStateStore({
        firstValue: js_expr("1"),
        list: [js_expr('$state.firstValue'), 2, 3],
        listSum: js_expr('$state.list.reduce((a, b) => a + b, 0)'),
        listWithX: js_expr('[...$state.list, "X"]'),
      });
      expect($state.listSum).toBe(6);
      $state.update([['list', 'append', 4]]);
      expect($state.list).toEqual([1, 2, 3, 4]);
      expect($state.listWithX).toEqual([1, 2, 3, 4, "X"])
      expect($state.listSum).toBe(10);
      $state.firstValue = 0
      expect($state.listWithX).toEqual([0, 2, 3, 4, "X"])

    });

    it('should throw if circular reference is detected', () => {
      const $state = createStateStore({
        a: { __type__: 'ref', id: 'b' },
        b: { __type__: 'ref', id: 'a' },
        c: 10
      });
      expect(() => $state.a).toThrow(/Cycle detected in computation/);
      expect($state.c).toBe(10);
    });

    it('should demonstrate that during "update", ASTs are evaluated in order and not re-evaluated in a second pass', () => {
      const $state = createStateStore({
        a: 1,
        b: js_expr('$state.a + 1')
      });

      // Initial state
      expect($state.a).toBe(1);
      expect($state.b).toBe(2);

      // Update 'a' first, then 'b'
      $state.update([
        ['a', 'reset', 10],
        ['b', 'reset', js_expr('$state.a + 2')]
      ]);
      expect($state.a).toBe(10);
      expect($state.b).toBe(12);

      // Reset the store
      $state.a = 1;

      // Update 'b' first, then 'a'
      $state.update([
        ['b', 'reset', js_expr('$state.a + 1')],
        ['a', 'reset', 10]
      ]);
      expect($state.a).toBe(10);
      expect($state.b).toBe(2);  // 'b' is still based on the old value of 'a'
    });

  });
})
