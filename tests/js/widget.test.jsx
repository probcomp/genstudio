import * as Plot from "@observablehq/plot";
import { render } from '@testing-library/react';
import * as React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { createStateStore, evaluate, StateProvider } from '../../src/genstudio/js/widget';

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

    it('should evaluate a js expression with params', () => {
      const ast = {
        __type__: "js_source",
        expression: true,
        value: '%1 + %2',
        params: [2, 3]
      }
      const result = evaluate(ast, {}, null)
      expect(result).toBe(5)
    })

    it('should evaluate a js expression with complex params', () => {
      const ast = {
        __type__: "js_source",
        expression: true,
        value: '%1.map(x => x * %2)',
        params: [[1, 2, 3], 2]
      }
      const result = evaluate(ast, {}, null)
      expect(result).toEqual([2, 4, 6])
    })
    it('should preserve object identity in params', () => {
      const obj = new Date('2024-01-01')
      const ast = {
        __type__: "js_source",
        expression: true,
        value: '%1 === %2',
        params: [obj, obj]
      }
      const result = evaluate(ast, {}, null)
      expect(result).toBe(true)
    })

    it('should evaluate a multi-line js source (requires explicit return)', () => {
      const ast = {
        __type__: "js_source",
        value: 'let x = 0\n x = 1\n return x'
      }
      const result = evaluate(ast, {}, {}, null)
      expect(result).toBe(1)
    })

    it('should evaluate a multi-line js source with params', () => {
      const ast = {
        __type__: "js_source",
        value: 'let x = %1\n x += %2\n return x',
        params: [5, 3]
      }
      const result = evaluate(ast, {}, null)
      expect(result).toBe(8)
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
      const init = {
        initialState: {"count": 0},
        syncedKeys: new Set(["count"])
      }
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

      const model = { on: vi.fn(), off: vi.fn() };

      const data = {
        ast,
        model,
        initialState: {"count": 0},
        syncedKeys: new Set(["count"])
      };


      const { container, rerender } = render(
        <StateProvider {...data}  />
      );

      expect(container.innerHTML).toContain('Count: 0');
    });
  });

  describe('state and Plot.js combination', () => {
    it('should handle Plot.InitialState and Plot.js combination correctly', async () => {
      const consoleSpy = vi.spyOn(console, 'log');

      // Simulate the AST created by Python's `&` operator
      const ast = [
        {__type__: "js_ref", path: "Row"},
        {},
        {
          __type__: 'function',
          path: 'InitialState',
          args: ['foo', {__type__: 'ref', state_key: 'foo'}]
        },
        {__type__: "js_source", value: 'console.log($state.foo) || $state.foo'}
      ];

      const data = {
        ast,
        initialState: {'foo': 123},
        syncedKeys: new Set(['foo'])
      };

      render(
        <StateProvider {...data} />
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
      const $state = createStateStore({
        initialState: {
          count: 0,
          name: 'Test'
        },
        syncedKeys: new Set(['count', 'name'])
      });
      expect($state.count).toBe(0);
      expect($state.name).toBe('Test');
    });

    it('should update a value', () => {
      const $state = createStateStore({
        initialState: { count: 0 },
        syncedKeys: new Set(['count'])
      });
      $state.count = 1;
      expect($state.count).toBe(1);
    });

    it('should handle computed values', () => {
      const $state = createStateStore({
        initialState: {
          count: 0,
          doubleCount: js_expr('$state.count * 2'),
          countArray: js_expr('[$state.count, $state.count]')
        },
        syncedKeys: new Set(['count'])
      });
      expect($state.doubleCount).toBe(0);
      expect($state.countArray).toEqual([0, 0])
      $state.count = 2;
      expect($state.doubleCount).toBe(4);
      expect($state.countArray).toEqual([2, 2])
    });

    it('should handle references', () => {
      const $state = createStateStore({
        initialState: {
          original: 10,
          reference: { __type__: 'ref', state_key: 'original' },
          c: 10
        },
        syncedKeys: new Set(['original', 'reference'])
      });
      expect($state.reference).toBe(10);
      $state.original = 20;
      expect($state.reference).toBe(20);
    });

    it('should take computed properties as the initial value for updates', () => {
      const $state = createStateStore({
        initialState: {
          firstValue: js_expr("1"),
          list: js_expr('[$state.firstValue, 2, 3]'),
        }
      });
      expect([$state.firstValue, $state.list]).toEqual([1, [1, 2, 3]])
      $state.firstValue = 10
      expect($state.list).toEqual([10, 2, 3])
      $state.update(['list', 'append', 10])
      expect($state.list).toEqual([10, 2, 3, 10])
      $state.firstValue = 0
      expect($state.list).toEqual([10, 2, 3, 10])
    });

    it('should apply "append" operation', () => {
      const $state = createStateStore({
        initialState: {
          list: js_expr('[1, 2, 3]')
        }
      });
      expect($state.list).toEqual([1, 2, 3]);
      $state.update(['list', 'append', 4]);
      expect($state.list).toEqual([1, 2, 3, 4]);
    });

    it('should throw if circular reference is detected', () => {
      const $state = createStateStore({
        initialState: {
          a: { __type__: 'ref', state_key: 'b' },
          b: { __type__: 'ref', state_key: 'a' },
          c: 10
        },
        syncedKeys: new Set(['a', 'b'])
      });
      expect(() => $state.a).toThrow(/Cycle detected in computation/);
      expect($state.c).toBe(10);
    });

    it('should demonstrate that during "update", ASTs are evaluated in order and not re-evaluated in a second pass', () => {
      const $state = createStateStore({
        initialState: {
          a: 1,
          b: js_expr('$state.a + 1')
        },
        syncedKeys: new Set(['a', 'b'])
      });

      // Initial state
      expect($state.a).toBe(1);
      expect($state.b).toBe(2);

      // Update 'a' first, then 'b'
      $state.update(
        ['a', 'reset', 10],
        ['b', 'reset', js_expr('$state.a + 2')]
      );
      expect($state.a).toBe(10);
      expect($state.b).toBe(12);

      // Reset the store
      $state.a = 1;

      // Update 'b' first, then 'a'
      $state.update(
        ['b', 'reset', js_expr('$state.a + 1')],
        ['a', 'reset', 10]
      );
      expect($state.a).toBe(10);
      expect($state.b).toBe(2);  // 'b' is still based on the old value of 'a'
    });

    describe('deep property access', () => {
      it('should get deeply nested values', () => {
        const $state = createStateStore({
          initialState: {
            nested: { a: { b: { c: 42 } } },
            array: [{ x: 1 }, { x: 2 }],
            typedArray: new Float32Array([1, 2, 3])
          }
        });

        expect($state['nested.a.b.c']).toBe(42);
        expect($state['array.0.x']).toBe(1);
        expect($state['array.1.x']).toBe(2);
        expect($state['typedArray.1']).toBe(2);
      });

      it('should set deeply nested values', () => {
        const $state = createStateStore({
          initialState: {
            nested: { a: { b: { c: 42 } } },
            array: [{ x: 1 }, { x: 2 }]
          }
        });

        $state['nested.a.b.c'] = 100;
        expect($state['nested.a.b.c']).toBe(100);

        $state['array.0.x'] = 10;
        expect($state['array.0.x']).toBe(10);
        expect($state.array[0].x).toBe(10);
      });

      it('should create intermediate objects when setting deep paths', () => {
        const $state = createStateStore({
          initialState: {
            data: {}
          }
        });

        $state['data.deeply.nested.value'] = 42;
        expect($state['data.deeply.nested.value']).toBe(42);
        expect($state.data.deeply.nested.value).toBe(42);
      });

      it('should handle array paths with automatic array creation', () => {
        const $state = createStateStore({
          initialState: {
            points: []
          }
        });

        $state['points.0.x'] = 10;
        $state['points.0.y'] = 20;

        expect($state.points[0]).toEqual({ x: 10, y: 20 });
        expect($state['points.0.x']).toBe(10);
      });

      it('should maintain reactivity with deep updates', () => {
        const $state = createStateStore({
          initialState: {
            data: { value: 1 },
            computed: js_expr('$state.data.value * 2')
          }
        });

        expect($state.computed).toBe(2);

        $state['data.value'] = 5;
        expect($state.computed).toBe(10);
      });
    });

  });
})
