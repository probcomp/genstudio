import genstudio.plot as Plot
from IPython.display import display
import asyncio
import time
import numpy as np
import ipywidgets as widgets

# This guide shows how to create interactive, animated plots using Python's asyncio with GenStudio.
# Note: Interactive elements only work in a live Python/Jupyter environment.

# Basic Animation Setup
# Let's animate dots moving in a sine wave:

ANIMATION_DURATION = 10  # seconds
NUM_DOTS = 75

# Initial plot with domain
p1 = (Plot.dot([[0, 0]] * NUM_DOTS) + Plot.domain([0, 10], [-1, 1])).display_as(
    "widget"
)
display(p1)

# Frequency control slider
frequency_slider = widgets.FloatSlider(
    value=1.0,
    min=0.1,
    max=5.0,
    step=0.1,
    description="Frequency:",
    continuous_update=True,
)
display(frequency_slider)


# Animation Function
async def animate_dots(duration):
    start_time = time.time()
    x_values = np.linspace(0, 10, NUM_DOTS)
    while time.time() - start_time < duration:
        y_values = np.sin(frequency_slider.value * (x_values + time.time()))
        dot_positions = list(zip(x_values, y_values))
        new_plot = Plot.dot(dot_positions) + Plot.domain([0, 10], [-1, 1])
        p1.reset(new_plot)
        await asyncio.sleep(1 / 60)  # 60 FPS


# Run the Animation
future = asyncio.ensure_future(animate_dots(ANIMATION_DURATION))

print(f"Animation running for {ANIMATION_DURATION} seconds")
print("To stop early: future.cancel()")
print("Adjust frequency with the slider")

# Key Concepts:
# 1. Asynchronous Programming: Using async/await for non-blocking animations.
# 2. Widget Integration: Real-time interaction via frequency_slider.
# 3. Plot Updates: Updating plot each frame with p1.reset().
# 4. Frame Rate Control: asyncio.sleep(1/60) maintains 60 FPS.
# 5. Asyncio Scheduling: asyncio.ensure_future() manages the animation task.
