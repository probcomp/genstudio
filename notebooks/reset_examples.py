import genstudio.plot as Plot
from IPython.display import display
import asyncio
import time
import numpy as np
import ipywidgets as widgets

# Configuration
ANIMATION_DURATION = 10  # seconds
NUM_DOTS = 75

# Create initial plot with domain
p = (Plot.dot([[0, 0]] * NUM_DOTS) + Plot.domain([0, 10], [-1, 1])).display_as("widget")
display(p)

# Create a slider widget for frequency control
frequency_slider = widgets.FloatSlider(
    value=1.0,
    min=0.1,
    max=5.0,
    step=0.1,
    description="Frequency:",
    continuous_update=True,
)
display(frequency_slider)


async def animate_dots(duration):
    start_time = time.time()
    x_values = np.linspace(0, 10, NUM_DOTS)
    while time.time() - start_time < duration:
        # Calculate y positions using sin wave with adjustable frequency
        y_values = np.sin(frequency_slider.value * (x_values + time.time()))

        # Update dot positions
        dot_positions = list(zip(x_values, y_values))
        new_plot = Plot.dot(dot_positions) + Plot.domain([0, 10], [-1, 1])
        p.reset(new_plot)

        # Wait for next frame (60 FPS)
        await asyncio.sleep(1 / 60)


# Run the animation
# Schedule the animation using asyncio.ensure_future()
future = asyncio.ensure_future(animate_dots(ANIMATION_DURATION))

# To stop the animation early, you can use: future.cancel()
print(f"Animation scheduled for {ANIMATION_DURATION} seconds")
print("To stop the animation early, use: future.cancel()")
print("Use the slider to adjust the frequency of the wave")

# Reset a blank plot with a Plot.Frames

(p := Plot.new().display_as("widget"))

p.reset(
    Plot.Frames(
        [Plot.dot([[i, 10]]) + Plot.domain([0, 20], [10, 10]) for i in range(20)], fps=5
    )
)
