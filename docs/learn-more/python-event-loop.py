# %% [markdown]
# This guide demonstrates how to create Python-controlled animations using GenStudio plots, the `.reset` method, and interactive sliders. We'll cover:
# 1. Setting up a basic animated plot
# 2. Creating interactive animations with ipywidgets
#
# We must use the `"widget"` [rendering modes](getting-started/#rendering-modes) for bidirectional python/javascript communication:

# %%
import genstudio.plot as Plot

Plot.configure({"display_as": "widget"})

# %% [markdown]
# First, a simple sine wave plot:

# %%
import numpy as np

x = np.linspace(0, 10, 100)
basic_plot = (
    Plot.line(list(zip(x, np.sin(x))))
    + Plot.domain([0, 10], [-1, 1])
    + Plot.height(200)
)
basic_plot

# %% [markdown]
# Now, let's animate it:

# %% tags=["hide_source"]
Plot.html(
    [
        "div.bg-black.text-white.p-3",
        """The following examples depend on communication with a python backend, and will not be interactive on the docs website.""",
    ],
)


# %%
import asyncio
import time


async def animate(duration=5):
    start_time = time.time()
    while time.time() - start_time < duration:
        t = time.time() - start_time
        y = np.sin(x + t)
        basic_plot.reset(
            Plot.line(list(zip(x, y)))
            + Plot.domain([0, 10], [-1, 1])
            + Plot.height(200)
        )
        await asyncio.sleep(1 / 30)  # 30 FPS


future = asyncio.ensure_future(animate())

# %% [markdown]
# We use the [reset method](bylight?dir=up&match=basic_plot.reset) of a plot to update its content in-place, inside an [async function](bylight?dir=up&match=async+def) containing a `while` loop, using [sleep](bylight?dir=up&match=asyncio.sleep(...\)) to control the frame rate. To avoid interference with Jupyter comms, we use [ensure_future](bylight?dir=up&match=asyncio.ensure_future(...\)) to run the function in a new thread.
#
# Let's make it interactive, using [ipywidgets](bylight?match=import...as+widgets,/widgets.FloatSlider/) sliders to control frequency and amplitude:

# %%
import ipywidgets as widgets

interactive_plot = (
    Plot.line(list(zip(x, np.sin(x))))
    + Plot.domain([0, 10], [-2, 2])
    + Plot.height(200)
)
frequency_slider = widgets.FloatSlider(
    value=1.0, min=0.1, max=5.0, step=0.1, description="Frequency:"
)
amplitude_slider = widgets.FloatSlider(
    value=1.0, min=0.1, max=2.0, step=0.1, description="Amplitude:"
)

# %% [markdown]
# Now, in our animation loop we [use the slider values](bylight?match=/\w%2B_slider\.value/) to compute the y value:


# %%
from IPython.display import display


async def interactive_animate(duration=10):
    start_time = time.time()
    while time.time() - start_time < duration:
        t = time.time() - start_time
        y = amplitude_slider.value * np.sin(frequency_slider.value * (x + t))
        interactive_plot.reset(
            Plot.line(list(zip(x, y)))
            + Plot.domain([0, 10], [-2, 2])
            + Plot.height(200)
        )
        await asyncio.sleep(1 / 30)


display(interactive_plot)
display(frequency_slider, amplitude_slider)
future = asyncio.ensure_future(interactive_animate())
