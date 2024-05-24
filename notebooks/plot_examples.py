# %% 

import gen.studio.plot as Plot
from gen.studio.js_modules import Hiccup
import numpy as np
import genjax as genjax
from genjax import gen
import jax 
import jax.numpy as jnp 
import jax.random as jrand
import numpy as np

# %% [markdown]
# ## Approach 
#
# - The [pyobsplot](https://github.com/juba/pyobsplot) library creates "stubs" in python which directly mirror the Observable Plot API. An AST-like "spec" is created in python and then interpreted in javascript.
# - The [Observable Plot](https://observablehq.com/plot/) library does not have "chart types" but rather "marks", which are layered to produce a chart. These are composable via `+` in Python.
#
# ## Instructions 
#
# The starting point for seeing what's possible is the [Observable Plot](https://observablehq.com/plot/what-is-plot) website.
# Plots are composed of **marks**, and you'll want to familiarize yourself with the available marks and how they're created.
#
#
# Generate random data from a normal distribution
def normal_100():
    return np.random.normal(loc=0, scale=1, size=1000)


# %% [markdown]
# ### Histogram

Plot.histogram(normal_100())

# %% [markdown]
# ### Scatter and Line plots
# Unlike other mark types which expect a single values argument, `dot` and `line`
# also accept separate `xs` and `ys` for passing in columnar data (usually the case
# when working with jax.)

Plot.dot({'x': normal_100(), 'y': normal_100()}) + Plot.frame()

# %% [markdown]
# ### One-dimensional heatmap

(
    Plot.rect(normal_100(), Plot.binX({"fill": "count"}))
    + Plot.color_scheme("YlGnBu")
    + {"height": 75}
)

 # %% [markdown]
 # ### Plot.doc
 # Plot.doc(Plot.foo) will render a markdown-formatted docstring when available:
 
 Plot.doc(Plot.line)
 
 # %% [markdown]
 # ### Plot composition 
 # 
 # Marks and options can be composed by including them as arguments to `Plot.new(...)`,
 # or by adding them to a plot. Adding marks or options does not change the underlying plot,
 # so you can re-use plots in different combinations.
 
circle = Plot.dot([[0, 0]], r=100)
circle

#%%
circle + Plot.frame() + {'inset': 50}

#%%

#%% [markdown]

# A GenJAX example


key = jrand.PRNGKey(314159)

# A regression distribution.
@gen
def regression(x, coefficients, sigma):
    basis_value = jnp.array([1.0, x, x**2])
    polynomial_value = jnp.sum(basis_value * coefficients)
    y = genjax.normal(polynomial_value, sigma) @ "v"
    return y


# Regression, with an outlier random variable.
@gen
def regression_with_outlier(x, coefficients):
    is_outlier = genjax.flip(0.1) @ "is_outlier"
    sigma = jnp.where(is_outlier, 30.0, 0.3)
    is_outlier = jnp.array(is_outlier, dtype=int)
    return regression(x, coefficients, sigma) @ "y"


# The full model, sample coefficients for a curve, and then use
# them in independent draws from the regression submodel.
@gen
def full_model(xs):
    coefficients = (
        genjax.mv_normal(
            jnp.zeros(3, dtype=float),
            2.0 * jnp.identity(3),
        )
        @ "alpha"
    )
    ys = regression_with_outlier.vmap(in_axes=(0, None))(xs, coefficients) @ "ys"
    return ys

data = jnp.arange(0, 10, 0.5)
key, sub_key = jrand.split(key)
tr = jax.jit(full_model.simulate)(sub_key, (data,))

key, *sub_keys = jrand.split(key, 10)
traces = jax.vmap(lambda k: full_model.simulate(k, (data,)))(jnp.array(sub_keys))

Plot.dot({'x': data, 
          'y': Plot.get_choice(traces, ["ys", Plot.Dimension('samples', view='grid'), "y", "v"])})


# %% [markdown]

### Things in progress

time_data = [[1, 2, 1, 2, 1, 2, 1],
             [1.5, 2.5, 1.5, 2.5, 1.5, 2.5, 1.5],
             [3, 4, 3, 4, 3, 4, 3],
             [3.5, 4.5, 3.5, 4.5, 3.5, 4.5, 3.5]]

# Plot.get_in is like get_address but for ordinary Python data structures (dicts/lists)
Plot.dot({'x': [0, 1, 2, 3, 4, 5, 6],
          'y': Plot.get_in(time_data, [Plot.Dimension('time')])})

#%%
Plot.dot({'x': [0, 1, 2, 3, 4, 5], 
          'y': Plot.Dimension('time', initial=1, fps=1, value=[[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3,]])}) + \
    Plot.dot({'x': [0, 1, 2, 3, 4, 5], 
              'y': Plot.Dimension('particle', view='grid', value=[[10, 10, 10, 10, 10, 10], [12, 12, 12, 12, 12, 12], [14, 14, 14, 14, 14, 14,]])}, fill='green')
    
    
# %%