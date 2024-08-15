# %%

from genstudio.plot import JSRef, d3, Math


def test_jswrapper_init():
    wrapper = JSRef("TestModule", "test_method")
    assert wrapper == {
        "__type__": "ref",
        "module": "TestModule",
        "name": "test_method",
    }
    assert wrapper.__name__ == "test_method"
    assert wrapper.__doc__ is None


def test_jswrapper_call():
    wrapper = JSRef("TestModule", "test_method")
    result = wrapper(1, 2, 3)

    assert result == {
        "__type__": "function",
        "module": "TestModule",
        "name": "test_method",
        "args": (1, 2, 3),
    }


def test_jsmodule_getattr():
    result = d3.test_method
    assert isinstance(result, JSRef)
    assert result == {
        "__type__": "ref",
        "module": "d3",
        "name": "test_method",
    }


def test_math_getattr():
    result = Math.test_method
    assert isinstance(result, JSRef)
    assert result == {
        "__type__": "ref",
        "module": "Math",
        "name": "test_method",
    }


def test_d3_method_call():
    result = d3.test_method(1, 2, 3)
    assert result == {
        "__type__": "function",
        "module": "d3",
        "name": "test_method",
        "args": (1, 2, 3),
    }


def test_math_method_call():
    result = Math.test_method(4, 5, 6)
    assert result == {
        "__type__": "function",
        "module": "Math",
        "name": "test_method",
        "args": (4, 5, 6),
    }


def run_tests():
    test_jsmodule_getattr()
    test_math_getattr()
    test_jswrapper_call()
    test_jswrapper_init()
    test_d3_method_call()
    test_math_method_call()
    print("all tests pass")


run_tests()

# %%
