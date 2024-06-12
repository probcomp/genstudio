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
            "filter": Plot.js("(d) => d.day <= $state.day"),
        },
    )
    + Plot.frame()
    
    # add a slider for a $state variable
    # + Plot.slider("day", range=bean_data_dims.size("day"))
    # (OR)
    # animate a $state variable
    + Plot.animate("day", range=bean_data_dims.size("day"))
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