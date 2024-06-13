# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

import genstudio.plot as Plot
import numpy as np
import genjax as genjax
from genjax import gen
import jax
import jax.numpy as jnp
import jax.random as jrand


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
# Generate random data from a normal distribution
# %%
def normal_100():
    return np.random.normal(loc=0, scale=1, size=1000)


# %% [markdown]
# #### Histogram
# %%
Plot.histogram(normal_100())

# %% [markdown]
# #### Scatter and Line plots
# Unlike other mark types which expect a single values argument, `dot` and `line`
# also accept separate `xs` and `ys` for passing in columnar data (usually the case
# when working with jax.)
# %%
Plot.dot({"x": normal_100(), "y": normal_100()}) + Plot.frame()

# %% [markdown]
# #### One-dimensional heatmap
# %%
(
    Plot.rect(normal_100(), Plot.binX({"fill": "count"}))
    + Plot.color_scheme("YlGnBu")
    + {"height": 75}
)

# %% [markdown]
# #### Plot.doc
#
# Plot.doc(Plot.foo) will render a markdown-formatted docstring when available:
# %%
Plot.doc(Plot.line)

# %% [markdown]
# #### Plot composition
#
# Marks and options can be composed by including them as arguments to `Plot.new(...)`,
# or by adding them to a plot. Adding marks or options does not change the underlying plot,
# so you can re-use plots in different combinations.

# %%
circle = Plot.dot([[0, 0]], r=100)
circle

# %%
circle + Plot.frame() + {"inset": 50}

# %% [markdown]
#
# A GenJAX example

# A regression distribution.
# %%
key = jrand.PRNGKey(314159)


@gen
def regression(x, coefficients, sigma):
    basis_value = jnp.array([1.0, x, x**2])
    polynomial_value = jnp.sum(basis_value * coefficients)
    y = genjax.normal(polynomial_value, sigma) @ "v"
    return y


# %% [markdown]
# Regression, with an outlier random variable.
# %%
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

traces

# %% [markdown]
# #### Multi-dimensional (nested) data
#
# Data from GenJAX often comes in the form of multi-dimensional (nested) lists.
# To prepare data for plotting, we can describe these dimensions using `Plot.dimensions`.
# %%
ys = traces.get_choices()["ys", ..., "y", "v"]

# %% [markdown]
# When passed to a plotting function, this annotated dimensional data will be flattened into
# a single list of objects, with entries for each dimension and leaf name. Here, we'll call
# .flatten() directly in python, but in practice the arrays will be flattened after (de)serialization
# to our JavaScript rendering environment.
# %%
Plot.dimensions(ys, ["sample", "ys"], leaves="y").flatten()[:10]
# => <Dimensioned shape=(9, 20), names=['sample', 'ys', 'y']>

# %% [markdown]
# #### Small Multiples
#
# The `facetGrid` option splits a dataset by key, and shows each split in its own chart
# with consistent scales.
# %%
(
    Plot.dot(
        Plot.dimensions(ys, ["sample", "ys"], leaves="y"),
        facetGrid="sample",
        x=Plot.repeat(data),
        y="y",
    )
    + {"height": 600}
    + Plot.frame()
)

# %% [markdown]
# `Plot.get_in` reads data from a nested structure, giving names to dimensions and leaves
# along the way in a single step. It works with Python lists/dicts as well as GenJAX
# traces and choicemaps. Here we'll construct a synthetic dataset and plot using `get_in`.
# %%
import random

# Initialize the bean growth with stem_length and soil_quality as constants for each bean
bean_growth = [
    [
        {
            "stem_length": 0,
            "soil_quality": random.uniform(0.7, 1.3),
            "genetic_disposition": random.uniform(0.8, 1.2),
        }
        for _ in range(10)
    ]
]

for day in range(1, 21):
    weather_event = random.choice(["rain", "no_rain"])
    rainfall = random.uniform(0.1, 3) if weather_event == "rain" else 0
    growth_factor = (
        random.uniform(0.05, 0.15) + (rainfall * random.uniform(0.1, 0.3))
        if rainfall
        else random.uniform(0.05, 0.15)
    )
    today = []
    for plant in bean_growth[-1]:
        stem_length = plant["stem_length"]
        disease_event = (
            0 if random.random() > 0.02 else -stem_length * random.uniform(0.2, 0.5)
        )
        growth = growth_factor * plant["soil_quality"] * plant["genetic_disposition"]
        noise = random.uniform(-0.3, 0.3)
        today.append(
            {**plant, "stem_length": stem_length + disease_event + growth + noise}
        )
    bean_growth.append(today)

bean_growth

# %% [markdown]
# Using `get_in` we've given names to each level of nesting (and leaf values), which we can see in the metadata
# of the Dimensioned object:
# %%
bean_data_dims = Plot.dimensions(bean_growth, ["day", "bean"])
# %%
bean_data_dims.flatten()

# %%[markdown]
# Now that our dimensions and leaf have names, we can pass them as options to `Plot.dot`.
# Here we'll use the `facetGrid` option to render a separate plot for each bean.
# %%

(
    Plot.dot(
        Plot.dimensions(bean_growth, ["day", "bean"]),
        {"x": "day", "y": "stem_length", "fill": "bean", "facetGrid": "bean"},
    )
    + Plot.frame()
)

# %% [markdown]
# Let's draw a line for each bean to plot its growth over time. The `z` channel splits the data into
# separate lines.
# %%
(
    Plot.line(
        bean_data_dims, {"x": "day", "y": "stem_length", "z": "bean", "stroke": "bean"}
    )
    + Plot.frame()
)

# %%
