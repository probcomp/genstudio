# Create an isotype chart using text marks with emoji

import genstudio.plot as Plot

data = [
    {"animal": "pigs", "country": "Great Britain", "count": 1354979},
    {"animal": "cattle", "country": "Great Britain", "count": 3962921},
    {"animal": "sheep", "country": "Great Britain", "count": 10931215},
    {"animal": "pigs", "country": "United States", "count": 6281935},
    {"animal": "cattle", "country": "United States", "count": 9917873},
    {"animal": "sheep", "country": "United States", "count": 7084151},
]

(
    Plot.initialState({"data": data})
    | Plot.text(
        Plot.js("$state.data"),
        {
            "text": Plot.js(
                """d => {
                const emoji = {pigs: "🐷", cattle: "🐮", sheep: "🐑"};
                return emoji[d.animal].repeat(Math.round(d.count / 1e6))
            }"""
            ),
            "y": "animal",
            "fy": "country",  # facet vertically by country
            "dx": 10,  # space between axis label and emoji
            "fontSize": 30,
            "frameAnchor": "left",
        },
    )
    + Plot.axisFy({"fontSize": 14, "frameAnchor": "top", "dy": -5})
    + {  # compute a height to fit the data
        "height": Plot.js(
            """const rowHeight = 50;
               const animals = new Set($state.data.map(d => d.animal)).size
               const countries = new Set($state.data.map(d => d.country)).size
               const titleHeight = 20
               return animals * countries * rowHeight + titleHeight""",
            expression=False,
        ),
        "y": {"label": None},
        "fy": {"label": None},
    }
    + Plot.title("Live stock (millions)")
)
