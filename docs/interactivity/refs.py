# %% [markdown]
#
# We can refer to the same value more than once in a plot by wrapping it in `Plot.ref`. The value will only be serialized once. Plot marks are automatically referenced in this way.
#
# `Plot.ref` gives us a "target" inside the widget that we can send updates to, using the `.state.update` method. We can send any number of operations, in the form `[ref_object, operation, value]`. The referenced value will change, and affected nodes will re-render.


# %%
import genstudio.plot as Plot

numbers = Plot.ref([1, 2, 3])
view = Plot.html(numbers).display_as("widget")
view

# %%

view.state.update([numbers, "append", 4])

# %% [markdown]

# There are three supported operations:
# - `"reset"` for replacing the entire value,
# - `"append"` for adding a single value to a list,
# - `"concat"` for adding multiple values to a list.

# If multiple updates are provided, they are applied synchronously.

# One can update `$state` values by specifying the name of the variable in the first position, eg. `view.state.update(["foo", "reset", "bar"])`.
