import genstudio.plot as Plot
from genjax import Pytree
from genstudio.plot import js
import jax.numpy as jnp


@Pytree.dataclass
class ColoredPoints(Pytree):
    """A simple dataclass containing points and their colors."""

    points: jnp.ndarray  # Shape (N, 2) for N 2D points
    colors: jnp.ndarray  # Shape (N, 3) for N RGB colors


# Example instantiation
points = jnp.array([[0.0, 0.0], [1.0, 1.0], [-1.0, 2.0]])
colors = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # RGB colors
colored_points = ColoredPoints(points=points, colors=colors)

(
    Plot.initialState({"ColoredPoints": colored_points})
    | Plot.dot(
        js("$state.ColoredPoints.points"),
        {
            "fill": js(
                """
                (_, i) => {
                    const [r, g, b] = $state.ColoredPoints.colors[i]
                    return `rgb(${r * 255}, ${g * 255}, ${b * 255})`
                }
                """
            )
        },
    )
)
