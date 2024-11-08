# test numppy and jax arrays

import genstudio.plot as Plot
import jax.random as random
import numpy as np

Plot.dot(random.uniform(random.PRNGKey(0), shape=(10, 2)))

Plot.dot(np.random.uniform(size=(10, 2)))
