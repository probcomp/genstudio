import genstudio.plot as Plot
import numpy as np


points = np.random.rand(100, 3)

Plot.plot(
    {
        "marks": [Plot.dot(points, fill="2")],
        "color": {"legend": True, "label": "My Title"},
    }
)

# equivalent:
(
    Plot.dot(points, x="0", y="1", fill="2")
    + {"color": {"legend": True, "label": "My Title"}}
)
