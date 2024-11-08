# test numppy and jax arrays in widget and html modes

import genstudio.plot as Plot
import jax.random as random
import numpy as np

np_data = np.random.uniform(size=(10, 2))
jax_data = random.uniform(random.PRNGKey(0), shape=(10, 2))

(
    Plot.html(["div.text-lg.font-bold", "Numpy arrays"])
    | Plot.dot(np_data).display_as("html") & Plot.dot(np_data).display_as("widget")
    | Plot.html(["div.text-lg.font-bold", "JAX arrays"])
    | Plot.dot(jax_data).display_as("html") & Plot.dot(jax_data).display_as("widget")
)
