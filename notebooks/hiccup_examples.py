import genstudio.plot as Plot

html = Plot.Hiccup


#
def on_click(event):
    print(f"Clicked on: {event['id']} at x position {event["clientX"]}")


#
# Create a Hiccup structure with interactive divs
hiccup = html(
    "div.container",
    ["h1", "Interactive Div Example"],
    ["p", "Click on a colored div to see its ID in the console."],
    [
        "div.interactive-area",
        [
            "div#red.box",
            {
                "style": {"backgroundColor": "red"},
                "onClick": lambda e: on_click({"id": "red", **e}),
                "onMouseMove": Plot.js("(e) => e.target.innerHTML = e.clientX"),
            },
            "Red",
        ],
        [
            "div#blue.box",
            {
                "style": {"backgroundColor": "lightblue"},
                "onClick": lambda e: on_click({"id": "blue", **e}),
                "onMouseMove": Plot.js("(e) => e.target.innerHTML = e.clientX"),
            },
            "Blue",
        ],
        [
            "div#green.box",
            {
                "style": {"backgroundColor": "green"},
                "onClick": lambda e: on_click({"id": "green", **e}),
                "onMouseMove": Plot.js("(e) => e.target.innerHTML = e.clientX"),
            },
            "Green",
        ],
    ],
    [
        "div.footer",
        [
            "p",
            "This example demonstrates html with onClick callbacks on simple div elements.",
        ],
    ],
).display_as("widget")
#
hiccup
