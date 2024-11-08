# %%

import unittest
from genstudio.layout import JSRef, JSCall

d3 = JSRef("d3")
Math = JSRef("Math")

JSRef("TestModule.test_method")


class TestJSRef(unittest.TestCase):
    def test_jsref_init(self):
        wrapper = JSRef("TestModule.test_method")
        self.assertEqual(
            wrapper.path,
            "TestModule.test_method",
            f"Expected path 'TestModule.test_method', got '{wrapper.path}'",
        )
        self.assertEqual(
            wrapper.__name__,
            "test_method",
            f"Expected __name__ 'test_method', got '{wrapper.__name__}'",
        )
        self.assertIsNone(
            wrapper.__doc__, f"Expected __doc__ None, got '{wrapper.__doc__}'"
        )

    def test_jsref_call(self):
        wrapper = JSRef("TestModule.test_method")
        result = wrapper(1, 2, 3)
        self.assertIsInstance(
            result, JSCall, f"Expected JSCall instance, got {type(result)}"
        )
        expected = {
            "__type__": "function",
            "path": "TestModule.test_method",
            "args": (1, 2, 3),
        }
        actual = result.for_json()
        self.assertEqual(actual, expected, f"Expected {expected}, got {actual}")

    def test_jsref_getattr(self):
        result = d3.test_method
        self.assertIsInstance(
            result, JSRef, f"Expected JSRef instance, got {type(result)}"
        )
        expected = {
            "__type__": "js_ref",
            "path": "d3.test_method",
        }
        actual = result.for_json()
        self.assertEqual(actual, expected, f"Expected {expected}, got {actual}")

    def test_math_getattr(self):
        result = Math.test_method
        self.assertIsInstance(
            result, JSRef, f"Expected JSRef instance, got {type(result)}"
        )
        expected = {
            "__type__": "js_ref",
            "path": "Math.test_method",
        }
        actual = result.for_json()
        self.assertEqual(actual, expected, f"Expected {expected}, got {actual}")

    def test_d3_method_call(self):
        result = d3.test_method(1, 2, 3)
        self.assertIsInstance(
            result, JSCall, f"Expected JSCall instance, got {type(result)}"
        )
        expected = {
            "__type__": "function",
            "path": "d3.test_method",
            "args": (1, 2, 3),
        }
        actual = result.for_json()
        self.assertEqual(actual, expected, f"Expected {expected}, got {actual}")

    def test_math_method_call(self):
        result = Math.test_method(4, 5, 6)
        self.assertIsInstance(
            result, JSCall, f"Expected JSCall instance, got {type(result)}"
        )
        expected = {
            "__type__": "function",
            "path": "Math.test_method",
            "args": (4, 5, 6),
        }
        actual = result.for_json()
        self.assertEqual(actual, expected, f"Expected {expected}, got {actual}")


if __name__ == "__main__":
    unittest.main()

# %%
