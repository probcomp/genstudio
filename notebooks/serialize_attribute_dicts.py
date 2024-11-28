# %% [markdown]
# # Serializing Structured Data
#
# This example demonstrates two key features of GenStudio's state management:
#
# 1. Support for serializing structured data using `@Pytree.dataclass`. Any class with an `attributes_dict`
#    method can be automatically serialized and synchronized between Python and JavaScript.
#
# 2. Support for nested state updates using dot notation (e.g., `state.colors.0`). The state manager
#    automatically creates intermediate objects/arrays as needed, maintaining proper reactivity.

# %%
import genstudio.plot as Plot
from genjax import Pytree
from genstudio.plot import js
import jax.numpy as jnp


@Pytree.dataclass
class ColoredPoints(Pytree):
    """A simple dataclass containing points and their colors."""

    points: jnp.ndarray  # Shape (N, 2) for N 2D points
    colors: jnp.ndarray  # Shape (N, 3) for N RGB colors


def colored_dots(state_path):
    """Creates a reusable colored dots component that displays and allows modification of dots.

    Args:
        state_path: String path to a ColoredPoints instance in state
    """
    return Plot.dot(
        js(f"$state.{state_path}.points"),
        fill=js(
            f"""
            (_, i) => {{
                const [r, g, b] = $state.{state_path}.colors[i]
                return `rgb(${{r * 255}}, ${{g * 255}}, ${{b * 255}})`
            }}
            """
        ),
        r=20,
        render=Plot.renderChildEvents(
            {
                "onClick": js(f"""(e) => {{
            $state[`{state_path}.colors.${{e.index}}`] = [Math.random(), Math.random(), Math.random()]
        }}""")
            }
        ),
    )


# Example instantiation - clicking dots will randomly change their colors
points = jnp.array([[0.0, 0.0], [1.0, 1.0], [-1.0, 2.0]])
colors = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # RGB colors
colored_points = ColoredPoints(points=points, colors=colors)

(Plot.initialState({"ColoredPoints": colored_points}) | colored_dots("ColoredPoints"))
