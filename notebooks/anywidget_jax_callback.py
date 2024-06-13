# %%
# Example of using JAX's io_callback to send messages to an AnyWidget instance.
# The widget responds, but python won't receive the message until after the
# current cell has evaluated.

from jax.experimental import io_callback
import jax
import jax.numpy as jnp
import anywidget

# %%


class Widget(anywidget.AnyWidget):
    _esm = "anywidget_jax_callback.js"

    def __init__(self):
        super().__init__()

    @anywidget.experimental.command
    def _receive_message(self, msg, buffers):
        print(f"Received message from js: {msg}")
        return ["ok", []]


w = Widget()
w
# %%


def effectful_fn(x):
    print(f"Performing an effect with {x}")
    w.send({"kind": "genstudio", "content": x.tolist()})
    return x


@jax.jit
def numpy_random_like(x):
    return io_callback(effectful_fn, x, x)


jax.vmap(numpy_random_like)(jnp.zeros(5))
