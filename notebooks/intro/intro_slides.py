# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: slide,slideshow,-all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
# ---

# %% slideshow={"slide_type": "none"}
import random

plant_growth = [
    [
        {
            "stem_length": 0,
            "soil_quality": random.uniform(0.7, 1.3),
            "climate_adaption": random.uniform(0.8, 1.2),
        }
        for _ in range(10)
    ]
]

for day in range(1, 21):
    weather_event = random.choice(["rain", "no_rain"])
    rainfall = random.uniform(0.1, 3) if weather_event == "rain" else 0
    growth_factor = (
        random.uniform(0.05, 0.15) + (rainfall * random.uniform(0.1, 0.3))
        if rainfall
        else random.uniform(0.05, 0.15)
    )
    today = []
    for plant in plant_growth[-1]:
        stem_length = plant["stem_length"]
        disease_event = (
            0 if random.random() > 0.02 else -stem_length * random.uniform(0.2, 0.5)
        )
        growth = growth_factor * plant["soil_quality"] * plant["climate_adaption"]
        noise = random.uniform(-0.3, 0.3)
        today.append(
            {**plant, "stem_length": stem_length + disease_event + growth + noise}
        )
    plant_growth.append(today)


# %% [markdown] slideshow={"slide_type": "slide"}
# <h3 style="transform: rotate(-6deg) translateY(45px); display: inline-block;">GenStudio</h3>
# <h1 style="font-size: 11em;">Plot</h1>


# %% [markdown] slideshow={"slide_type": "slide"}
# ### Plotting in Python for GenJAX


# %% [markdown] slideshow={"slide_type": "slide"}
# ### If plotting is faster / easier / nicer, we'll do more of it


# %% [markdown] slideshow={"slide_type": "slide"}
# ### Composable
# ### Functional / Stateless
# ### Interactive
# ### Works in common environments

# %% slideshow={"slide_type": "slide"}
import genstudio.plot as Plot

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Describe & use dimensional data

# %% slideshow={"slide_type": "slide"}
# Multidimensional array representing _ days of growth of _ plants.
plant_growth

# %% slideshow={"slide_type": "slide"}
plant_growth_described = Plot.dimensions(plant_growth, ["day", "plant"])
plant_growth_described

# %% slideshow={"slide_type": "slide"}
plant_growth_described.flatten()

# %% slideshow={"slide_type": "slide"}

Plot.line(
    plant_growth_described,
    {"x": "day", "y": "stem_length", "z": "plant", "stroke": "plant", "tip": True},
)

# %% slideshow={"slide_type": "slide"}

Plot.get_in(
    [[1, 5, 3, 4, 1], [4, 7, 2, 4, 8], [1, 9, 6, 8, 3]],
    [{...: "day"}, {...: "plant"}, {"leaves": "height"}],
).flatten()


# %% [markdown] slideshow={"slide_type": "slide"}
# ## Composition

# %% slideshow={"slide_type": "slide"}

# Calculate average, min, and max stem length per day
daily_stats = [
    {
        "day": day,
        "average_stem_length": sum(plant["stem_length"] for plant in plants)
        / len(plants),
        "min_stem_length": min(plant["stem_length"] for plant in plants),
        "max_stem_length": max(plant["stem_length"] for plant in plants),
    }
    for day, plants in enumerate(plant_growth)
]
daily_stats

# %% slideshow={"slide_type": "slide"}

average_line = Plot.line(daily_stats, {"x": "day", "y": "average_stem_length"})
average_line

# %% slideshow={"slide_type": "slide"}

average_line + Plot.grid() + Plot.frame()

# %% slideshow={"slide_type": "slide"}
spread_rects = Plot.barY(
    daily_stats,
    {"x": "day", "y1": "min_stem_length", "y2": "max_stem_length", "fillOpacity": 0.2},
)

average_line + spread_rects

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Small multiples

# %% slideshow={"slide_type": "slide"}

(
    Plot.dot(
        plant_growth_described,
        {"x": "day", "y": "stem_length", "facetGrid": "plant", "tip": True},
    )
    + Plot.frame()
)

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Animate & scrub

# %% slideshow={"slide_type": "slide"}
(
    Plot.dot(
        plant_growth_described,
        x="day",
        y="stem_length",
        facetGrid="plant",
        filter=Plot.js("(plant) => plant.day <= $state.currentDay"),
    )
    + Plot.frame()
    + Plot.animate("currentDay", plant_growth_described.size("day"), fps=10)
)


# %% slideshow={"slide_type": "slide"}
import numpy as np

values = np.random.normal(loc=0, scale=1, size=1000)

Plot.histogram(values)

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Where:
# - Jupyter / IPython
# - Colab
# - Marimo
# - Quarto
# - Jupytext / VS Code Python Interactive Window

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Learn more:
# - probcomp/genstudio
# - Observable Plot (observablehq.com/plot)
# <br/><br/>
# ### Credits
# - PyObsplot (github.com/juba/pyobsplot)
# - AnyWidget (github.com/manzt/anywidget)

# %% [markdown] slideshow={"slide_type": "none"}
# <style>
#   .jp-InputArea-prompt, .jp-OutputArea-prompt {
#       display: none;
#    }
# .reveal {
#   h1, h2, h3, h4, h5, h6, p, bq, ul, ol {
#       font-family: Georgia;
#   }
#   .doc-header {
#       font-size: 30px;
#   }
#   .doc-content::after {
#       content: "";
#       display: block;
#       position: absolute;
#       top: 0;
#       left: 0;
#       right: 0;
#       height: 150px; /* Set the height to limit the fading effect to 150px */
#       background: linear-gradient(to bottom, rgba(255,255,255,0) 0%, rgba(255,255,255,1) 100%);
#       pointer-events: none;
#   }
#   .doc-content {
#       position: relative;
#       max-width: none;
#       font-size: 18px;
#       height: 149px;
#       overflow: hidden;
#   }
#   div.highlight > pre, div.jp-OutputArea-output > pre {
#       font-size: 18px;
#   }
#   .widget-subarea {
#    margin: 20px 50px;
#    }
# }
#
# </style>
