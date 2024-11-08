import unittest
import datetime
import numpy as np

try:
    import jax.numpy as jnp
except ImportError:
    jnp = None

from genstudio.widget import (
    to_json,
    deserialize_buffer_entry,
    replace_buffers,
    CollectedState,
    serialize_binary_data,
)


class TestWidgetArrayHandling(unittest.TestCase):
    def test_numpy_scalar(self):
        # Test handling of 0-d numpy arrays (scalars)
        scalar = np.array(42)
        result = to_json(scalar)
        self.assertEqual(result, 42)

    def test_numpy_array(self):
        # Test serialization of numpy arrays
        arr = np.array([[1, 2], [3, 4]], dtype=np.float32)
        buffers = []
        result = to_json(arr, buffers=buffers)

        # Add check to ensure result is a dictionary
        assert isinstance(result, dict)
        self.assertEqual(len(buffers), 1)
        self.assertEqual(result["__type__"], "ndarray")
        self.assertEqual(result["dtype"], "float32")
        self.assertEqual(result["shape"], (2, 2))

        # Test deserialization
        deserialized = deserialize_buffer_entry(result, buffers)
        np.testing.assert_array_equal(deserialized, arr)

    def test_nested_arrays(self):
        # Test handling of nested structures with arrays
        data = {
            "arr1": np.array([1, 2, 3]),
            "nested": {"arr2": np.array([[4, 5], [6, 7]])},
        }
        buffers = []
        result = to_json(data, buffers=buffers)

        self.assertEqual(len(buffers), 2)  # Should have 2 buffers

        # Test full round-trip
        reconstructed = replace_buffers(result, buffers)
        np.testing.assert_array_equal(reconstructed["arr1"], data["arr1"])
        np.testing.assert_array_equal(
            reconstructed["nested"]["arr2"], data["nested"]["arr2"]
        )

    @unittest.skipIf(jnp is None, "JAX not installed")
    def test_jax_array(self):
        if jnp is None:  # Add extra guard
            return
        # Test handling of JAX arrays
        arr = jnp.array([[1, 2], [3, 4]])
        buffers = []
        result = to_json(arr, buffers=buffers)

        assert isinstance(result, dict)
        self.assertEqual(len(buffers), 1)
        self.assertEqual(result["__type__"], "ndarray")

        # Test deserialization
        deserialized = deserialize_buffer_entry(result, buffers)
        np.testing.assert_array_equal(deserialized, np.array([[1, 2], [3, 4]]))

    def test_array_dtypes(self):
        # Test various numpy dtypes
        dtypes = [np.int8, np.int16, np.int32, np.int64, np.float32, np.float64]

        for dtype in dtypes:
            arr = np.array([1, 2, 3], dtype=dtype)
            buffers = []
            result = to_json(arr, buffers=buffers)

            assert isinstance(result, dict)
            self.assertEqual(result["dtype"], dtype.__name__)
            deserialized = deserialize_buffer_entry(result, buffers)
            np.testing.assert_array_equal(deserialized, arr)
            self.assertEqual(deserialized.dtype, dtype)

    def test_nan_handling(self):
        # Test handling of NaN values
        nan_float = float("nan")
        self.assertIsNone(to_json(nan_float))

        # Test NaN in numpy array
        arr = np.array([1.0, np.nan, 3.0])
        buffers = []
        result = to_json(arr, buffers=buffers)

        assert isinstance(result, dict)
        self.assertEqual(result["__type__"], "ndarray")

        # Test deserialization
        deserialized = deserialize_buffer_entry(result, buffers)
        np.testing.assert_array_equal(np.isnan(deserialized), np.isnan(arr))
        np.testing.assert_array_equal(
            deserialized[~np.isnan(deserialized)], arr[~np.isnan(arr)]
        )

    def test_nan_in_nested_structure(self):
        # Test NaN handling in nested structures
        data = {
            "regular": 1.0,
            "nan_value": float("nan"),
            "list": [1.0, float("nan"), 3.0],
            "nested": {"nan": float("nan")},
        }
        result = to_json(data)
        assert isinstance(result, dict)
        assert isinstance(result["nested"], dict)
        self.assertEqual(result["regular"], 1.0)
        self.assertIsNone(result["nan_value"])
        self.assertEqual(result["list"], [1.0, None, 3.0])
        self.assertIsNone(result["nested"]["nan"])


class TestCollectedState(unittest.TestCase):
    def test_state_entry_collection(self):
        cs = CollectedState()

        # Test basic state entry
        entry = cs.state_entry("key1", "value1")
        self.assertEqual(entry, {"__type__": "ref", "state_key": "key1"})
        self.assertEqual(cs.initialState["key1"], "value1")

        # Test synced state entry
        entry = cs.state_entry("key2", "value2", sync=True)
        self.assertTrue("key2" in cs.syncedKeys)

    def test_listener_registration(self):
        cs = CollectedState()

        def dummy_listener():
            pass

        listeners = {"key1": dummy_listener}
        cs.add_listeners(listeners)

        self.assertTrue("key1" in cs.syncedKeys)
        self.assertEqual(cs.stateListeners["key1"], [dummy_listener])


class TestSerialization(unittest.TestCase):
    def test_datetime_serialization(self):
        # Test date
        date = datetime.date(2024, 1, 1)
        result = to_json(date)
        self.assertEqual(result, {"__type__": "datetime", "value": "2024-01-01"})

        # Test datetime
        dt = datetime.datetime(2024, 1, 1, 12, 30)
        result = to_json(dt)
        self.assertEqual(
            result, {"__type__": "datetime", "value": "2024-01-01T12:30:00"}
        )

    def test_binary_data_serialization(self):
        data = bytes([1, 2, 3, 4])
        buffers = []
        result = serialize_binary_data(buffers, {"__type__": "buffer", "data": data})

        self.assertEqual(len(buffers), 1)
        self.assertEqual(buffers[0], data)
        self.assertEqual(result["__buffer_index__"], 0)
        self.assertIsNone(result["data"])

    def test_unsupported_type_error(self):
        class UnsupportedType:
            pass

        with self.assertRaises(TypeError):
            to_json(UnsupportedType())

    def test_exhaustible_iterator_warning(self):
        def gen():
            yield from range(3)

        iterator = gen()
        with self.assertWarnsRegex(
            UserWarning, "Potentially exhaustible iterator encountered"
        ):
            result = to_json(iterator)

        self.assertEqual(result, [0, 1, 2])
