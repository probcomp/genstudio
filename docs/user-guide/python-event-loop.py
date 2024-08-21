# %% [markdown]
# > **NOTE**: This technique will only work in a "live" jupyter session because it depends on communication with a python backend.

# %%

import genstudio.plot as Plot
from IPython.display import display
import asyncio
import time
import numpy as np

# %% [markdown]

# # Python-Controlled Animations in GenStudio

# This guide demonstrates how to create Python-controlled animations using GenStudio plots, the `.reset` method, and interactive sliders. We'll cover:
# 1. Setting up a basic animated plot
# 2. Creating interactive animations with ipywidgets

# GenStudio plots can be displayed as "widgets" or "html". In order for python<>javascript communication to work, we must use its "widget" mode.

# %%

Plot.configure({"display_as": "widget"})

# %% [markdown]

# First, a simple sine wave plot:

# %%
x = np.linspace(0, 10, 100)
basic_plot = (
    Plot.line(list(zip(x, np.sin(x))))
    + Plot.domain([0, 10], [-1, 1])
    + Plot.height(200)
)
basic_plot
# %% [markdown]

# Now, let's animate it:

# %%


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

# We use the [reset method](uplight?match=basic_plot.reset) of a plot to update its content in-place, inside an [async function](uplight?match=async+def) containing a `while` loop, using [sleep](uplight?match=asyncio.sleep(...\)) to control the frame rate. To avoid interference with Jupyter comms, we use [ensure_future](uplight?match=asyncio.ensure_future(...\)) to run the function in a new thread.


# Let's make it interactive, using [ipywidgets](uplight?dir=down&match=import...as+widgets,/widgets.FloatSlider/) sliders to control frequency and amplitude:

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

# Now, in our animation loop we [use the slider values](uplight?dir=down&match=/\w%2B_slider\.value/) to compute the y value:

# %%


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
