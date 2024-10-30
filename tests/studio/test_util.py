# %%

from genstudio.layout import JSRef, JSCall

d3 = JSRef("d3")
Math = JSRef("Math")

JSRef("TestModule.test_method")


def assert_with_message(condition, message):
    try:
        assert condition, message
    except AssertionError as e:
        print(f"Assertion failed: {str(e)}")
        raise


def test_jsref_init():
    wrapper = JSRef("TestModule.test_method")
    assert_with_message(
        wrapper.path == "TestModule.test_method",
        f"Expected path 'TestModule.test_method', got '{wrapper.path}'",
    )
    assert_with_message(
        wrapper.__name__ == "test_method",
        f"Expected __name__ 'test_method', got '{wrapper.__name__}'",
    )
    assert_with_message(
        wrapper.__doc__ is None, f"Expected __doc__ None, got '{wrapper.__doc__}'"
    )


def test_jsref_call():
    wrapper = JSRef("TestModule.test_method")
    result = wrapper(1, 2, 3)
    assert_with_message(
        isinstance(result, JSCall), f"Expected JSCall instance, got {type(result)}"
    )
    expected = {
        "__type__": "function",
        "path": "TestModule.test_method",
        "args": (1, 2, 3),
    }
    actual = result.for_json()
    assert_with_message(actual == expected, f"Expected {expected}, got {actual}")


def test_jsref_getattr():
    result = d3.test_method
    assert_with_message(
        isinstance(result, JSRef), f"Expected JSRef instance, got {type(result)}"
    )
    expected = {
        "__type__": "js_ref",
        "path": "d3.test_method",
    }
    actual = result.for_json()
    assert_with_message(actual == expected, f"Expected {expected}, got {actual}")


def test_math_getattr():
    result = Math.test_method
    assert_with_message(
        isinstance(result, JSRef), f"Expected JSRef instance, got {type(result)}"
    )
    expected = {
        "__type__": "js_ref",
        "path": "Math.test_method",
    }
    actual = result.for_json()
    assert_with_message(actual == expected, f"Expected {expected}, got {actual}")


def test_d3_method_call():
    result = d3.test_method(1, 2, 3)
    assert_with_message(
        isinstance(result, JSCall), f"Expected JSCall instance, got {type(result)}"
    )
    expected = {
        "__type__": "function",
        "path": "d3.test_method",
        "args": (1, 2, 3),
    }
    actual = result.for_json()
    assert_with_message(actual == expected, f"Expected {expected}, got {actual}")


def test_math_method_call():
    result = Math.test_method(4, 5, 6)
    assert_with_message(
        isinstance(result, JSCall), f"Expected JSCall instance, got {type(result)}"
    )
    expected = {
        "__type__": "function",
        "path": "Math.test_method",
        "args": (4, 5, 6),
    }
    actual = result.for_json()
    assert_with_message(actual == expected, f"Expected {expected}, got {actual}")


def run_tests():
    test_jsref_getattr()
    test_math_getattr()
    test_jsref_call()
    test_jsref_init()
    test_d3_method_call()
    test_math_method_call()
    print("all tests pass")


run_tests()

# %%
