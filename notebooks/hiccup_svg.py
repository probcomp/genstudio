# %%
import genstudio.plot as Plot

# %% [markdown]
# Example showing direct use of d3 to create an SVG visualization.
# `html` children can be svg or html elements.

# %%
Plot.js(
    """
// Create an SVG using d3
const svg = d3.create("svg")
  .attr("width", 200)
  .attr("height", 200);

// Add a circle
svg.append("circle")
  .attr("cx", 100)
  .attr("cy", 100)
  .attr("r", 50)
  .attr("fill", "steelblue");

// Add some text
svg.append("text")
  .attr("x", 100)
  .attr("y", 100)
  .attr("text-anchor", "middle")
  .attr("fill", "white")
  .text("D3 SVG");

return html(["div", svg.node()]);
""",
    expression=False,
)
