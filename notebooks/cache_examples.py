# %%

import genstudio.plot as Plot

# Only one console.log, Plot marks are automatically cached.
data = [[1, 2], [3, 4], [5, Plot.js("console.log('evaluating cached data') || 6")]]
d = Plot.dot(data)
d & d

# %% Frames
# Only one console.log, Plot marks are automatically cached.
Plot.Frames([d for _ in range(6)], fps=2)

# %% Multiple marks wrapped in Plot.cache
Plot.new(Plot.cache(Plot.dot([[2, 2]]) + Plot.dot([[1, 1]])))

# %% Add a mark to a cached mark

Plot.dot([[1, 1]]) + Plot.cache(Plot.dot([[2, 2]]))

# %%

import genstudio.plot as Plot

data1 = Plot.cache(["div", 1, 2, 3])
data2 = Plot.cache(["div", 9, 9, 9])
widget = (Plot.html(data1) & Plot.html(data2)).display_as("widget")
widget

# %% Updating Cached Data
widget.update_cache([data1, "append", 4])

widget.update_cache([data2, "concat", [5, 6]])

# %% Tailed Widget Example

tailedData = Plot.cache([1, 2, 3])
tailedWidget = Plot.Frames(tailedData, tail=True, fps=2).widget()
tailedWidget


# %% Updating Tailed Widget
tailedWidget.update_cache([tailedData, "concat", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])

# %% Granular state propagation
import genstudio.plot as Plot

Plot.configure(display_as="widget")

render_2 = Plot.cache(Plot.js("console.log('foo') || 'foo: '+$state.foo")) | Plot.cache(
    Plot.js("console.log('bar') || 'bar: '+$state.bar")
)

(
    Plot.js("'top'")
    | Plot.Slider("foo", label="foo", init=1)
    | Plot.Slider("bar", label="bar", init=1)
    | Plot.js("console.log('foo') || 'foo: '+$state.foo")
    | Plot.js("console.log('bar') || 'bar: '+$state.bar")
    | render_2
)


# %% Plot.Reactive initializes a variable
# We should see '123' logged once.
Plot.initial_state("foo", 123) & Plot.js("console.log($state.foo) || $state.foo")

import genstudio.plot as Plot
from IPython.display import display

p = Plot.new()
display(p)

p.reset(Plot.initial_state("foo", "foo") | Plot.js("$state.foo"))

p.reset(Plot.initial_state("blah", "blah") | Plot.js("$state.blah"))

# %%

one = Plot.cache(Plot.js("$state.foo"))
two = Plot.cache(Plot.js("$state.bar"))
plot = Plot.new() | Plot.initial_state("foo", "FOO") | Plot.initial_state("bar", "BAR")
plot

plot.reset(one)

plot.reset(two)
