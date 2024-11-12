# %% [markdown]

# GenStudio provides a complete Python interface to the Observable Plot API,
# designed to be as consistent as possible with the original.

# %%

import genstudio.plot as Plot

data = [{"x": 1, "y": 2}, {"x": 2, "y": 3}, {"x": 3, "y": 4}]

# %% [markdown]
# ## Basic Plot Creation

# Let's start with a simple scatter plot to compare the syntax:

# **Observable Plot**
# ```javascript
# Plot.plot({
#   marks: [
#     Plot.dot(data, {x: "x", y: "y"})
#   ]
# })
# ```

# **GenStudio**

# %%
Plot.dot(data, {"x": "x", "y": "y"})

# %% [markdown]
# As you can see, the basic structure is very similar. The main difference is that in GenStudio, we don't wrap marks in a `Plot.plot()` call - the `Plot.dot()` function returns a `PlotSpec` object that can be displayed directly.

# %% [markdown]
# ## Combining Marks

# In Observable Plot, you typically combine marks by including them in the `marks` array. In GenStudio, you use the `+` operator to combine marks and options:

# **Observable Plot**
# ```javascript
# Plot.plot({
#   marks: [
#     Plot.dot(data, {x: "x", y: "y"}),
#     Plot.line(data, {x: "x", y: "y"})
#   ]
# })
# ```

# **GenStudio**

# %%
Plot.dot(data, {"x": "x", "y": "y"}) + Plot.line(data, {"x": "x", "y": "y"})

# %% [markdown]
# ## Adding Plot Options

# In Observable Plot, you typically add options to the `Plot.plot()` call. In GenStudio, you can add options using the `+` operator:

# **Observable Plot**
# ```javascript
# Plot.plot({
#   marks: [Plot.dot(data, {x: "x", y: "y"})],
#   x: {domain: [0, 4]},
#   y: {domain: [0, 5]}
# })
# ```

# **GenStudio**

# %%
Plot.dot(data, {"x": "x", "y": "y"}) + Plot.domain([0, 4], [0, 5])

# %% [markdown]
# ## JavaScript Functions in Options

# Observable Plot allows you to use JavaScript functions directly in your options. GenStudio provides a similar capability using `Plot.js()`:

# **Observable Plot**
# ```javascript
# Plot.plot({
#   marks: [
#     Plot.dot(data, {
#       x: "x",
#       y: "y",
#       fill: d => d.x > 2 ? "red" : "blue"
#     })
#   ]
# })
# ```

# **GenStudio**

# %%
Plot.dot(data, {"x": "x", "y": "y", "fill": Plot.js("d => d.x > 2 ? 'red' : 'blue'")})

# %% [markdown]
# ## Interactivity

# Both Observable Plot and GenStudio support interactivity, but the approaches differ slightly due to the different environments.

# **Observable Plot**
# In Observable Plot, you typically use Observable's reactive programming model:
# ```javascript
# viewof frequency = Inputs.range([0.1, 10], {step: 0.1, label: "Frequency"})

# Plot.plot({
#   marks: [
#     Plot.line(d3.range(100), {
#       x: (d, i) => i,
#       y: (d, i) => Math.sin(i * frequency * Math.PI / 50)
#     })
#   ]
# })
# ```

# **GenStudio**

# %%
(
    Plot.line(
        {"x": range(100)},
        {
            "y": Plot.js("""(d, i) => {
                return Math.sin(i * $state.frequency * Math.PI / 50)
            }"""),
            "curve": "natural",
        },
    )
    + Plot.domain([0, 99], [-1, 1])
) | Plot.Slider(
    key="frequency",
    label="Frequency:",
    showValue=True,
    range=[0.1, 10],
    step=0.1,
    init=1,
)

# %% [markdown]
# In GenStudio, we use the `Plot.Slider` function to create interactive elements, and `$state` to access their values in our JavaScript functions.

# %% [markdown]
# ## Additional Features in GenStudio

# GenStudio includes some [additional marks](additions-to-observable) that aren't part of the core Observable Plot library.
