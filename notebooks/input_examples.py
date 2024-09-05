import genstudio.plot as Plot

p = Plot.new().display_as("widget")
params = {"q": 1}


#
def render():
    return Plot.Hiccup(
        [
            "div.flex.gap-4.items-center.text-sm",
            [
                "input",
                {
                    "type": "range",
                    "min": 0,
                    "max": 9,
                    "defaultValue": params["q"],
                    "onChange": lambda e: params.update({"q": e["value"]})
                    or p.reset(render()),
                },
            ],
            params["q"],
        ]
    )


p.reset(render())
p
