import genstudio.plot as Plot

# %% [markdown]

# ### Timeslines with barX

# For a richer example, we'll use some sample data from ancient civilizations.

# %%

civilizations = [
    {
        "name": "Mesopotamia",
        "start": -3500,
        "end": -539,
        "continent": "Asia",
        "peak_population": 10000000,
    },
    {
        "name": "Indus Valley Civilization",
        "start": -3300,
        "end": -1300,
        "continent": "Asia",
        "peak_population": 5000000,
    },
    {
        "name": "Ancient Egypt",
        "start": -3100,
        "end": -30,
        "continent": "Africa",
        "peak_population": 5000000,
    },
    {
        "name": "Ancient China",
        "start": -2070,
        "end": 1912,
        "continent": "Asia",
        "peak_population": 60000000,
    },
    {
        "name": "Maya Civilization",
        "start": -2000,
        "end": 1500,
        "continent": "North America",
        "peak_population": 2000000,
    },
    {
        "name": "Ancient Greece",
        "start": -800,
        "end": -146,
        "continent": "Europe",
        "peak_population": 8000000,
    },
    {
        "name": "Persian Empire",
        "start": -550,
        "end": 651,
        "continent": "Asia",
        "peak_population": 50000000,
    },
    {
        "name": "Roman Empire",
        "start": -27,
        "end": 476,
        "continent": "Europe",
        "peak_population": 70000000,
    },
    {
        "name": "Byzantine Empire",
        "start": 330,
        "end": 1453,
        "continent": "Europe",
        "peak_population": 30000000,
    },
    {
        "name": "Inca Empire",
        "start": 1438,
        "end": 1533,
        "continent": "South America",
        "peak_population": 12000000,
    },
    {
        "name": "Aztec Empire",
        "start": 1428,
        "end": 1521,
        "continent": "North America",
        "peak_population": 5000000,
    },
]

# %% [markdown]

# Below: a [barX](uplight?dir=down&match=Plot.barX) mark specifies [x1 and x2](uplight?dir=down&match="x1","x2") channels to show civilization timespans, with a [text mark](uplight?dir=down&match=Plot.text) providing labels that align with the bars. Both marks use the civilization name for the [y channel](uplight?dir=down&match="y":+"name"). [Color is used](uplight?dir=down&match=Plot.colorLegend(\),"fill":+"continent") to indicate the civilization name, and a legend is provided.

# %%

(
    Plot.barX(
        civilizations,
        {
            "x1": "start",
            "x2": "end",
            "y": "name",
            "fill": "continent",
            "sort": {"y": "x1"},
        },
    )
    + Plot.text(
        civilizations,
        {"text": "name", "x": "start", "y": "name", "textAnchor": "end", "dx": -3},
    )
    + {"axis": None, "marginLeft": 100}
    + Plot.colorLegend()
)
