import genstudio.plot as Plot
from IPython.display import display
import asyncio
import time
import numpy as np

# Configuration
ANIMATION_DURATION = 10  # seconds
NUM_DOTS = 50

# Create initial plot with domain
p = (Plot.dot([[0, 0]] * NUM_DOTS) + Plot.domain([0, 10], [-1, 1])).display_as("widget")
display(p)


async def animate_dots(duration):
    start_time = time.time()
    x_values = np.linspace(0, 10, NUM_DOTS)
    while time.time() - start_time < duration:
        # Calculate y positions using sin wave
        y_values = np.sin(x_values + time.time())

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
