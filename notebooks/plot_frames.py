# %%
# %load_ext autoreload
# %autoreload 2

# Notebook for sketching / designing a reactive variable feature to enable
# animation and sliders.

import genstudio.plot as Plot
from plot_examples import bean_data_dims

# Plot.slider and Plot.animate can be added to a plot.
# They will 

(
    Plot.dot(
        bean_data_dims,
        {
            "x": "day",
            "y": "height",
            "facetGrid": "bean",
            "filter": Plot.js("(d) => d.day <= $state.currentDay"),
        },
    )
    + Plot.frame()
    
    # add a slider for a $state variable
    + Plot.slider("currentDay", range=bean_data_dims.size("day"))
    # (OR)
    # animate a $state variable
    + Plot.animate("currentDay", range=bean_data_dims.size("day"), fps=1)
)



# %%

# animation
# - a reactive variable's options can include 'interval'

(
    Plot.dot(
        bean_data_dims,
        {
            "x": "day",
            "y": "height",
            "facetGrid": "bean"
        },
    )
    + Plot.frame()
)

# %%

# shorthand for "animate over frames split by $key with options $fps, $loop
# - adds filter for $key where (d) => d[$key] === reactive[$key]
# - adds reactive variable $key with options {$fps=1, $loop=true}

{'animate': {'fps': 1, 
             'loop': True, 
             'filterKey': 'day', 
             'comparator': '=='}}