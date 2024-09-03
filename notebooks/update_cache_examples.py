import genstudio.plot as Plot

#
Plot.configure(display_as="widget")
#
data = Plot.cache(["div", 1, 2, 3])
#
widget = Plot.html(data).widget()
widget


widget.update_cache([data, "append", 4])

widget.update_cache([data, "concat", [5, 6]])

# issues / questions / todo's ...
# - every widget keeps its own copy of the data
# - the data is only updated in javascript, not python
# - what about sliders and $state? scenarios:
#   - add frames. the "Slider" or "Reactive" instance should not have a
#     pre-computed range, but rather a "frames" argument which is used
#     directly to check for the end condition (for looping/stopping)
# - "tail" option will pause at end, if more data is added it will then continue

import genstudio.plot as Plot

tailedData = Plot.cache([1, 2, 3])
tailedWidget = Plot.Frames(tailedData, tail=True, fps=2).widget()
tailedWidget


tailedWidget.update_cache([tailedData, "concat", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
