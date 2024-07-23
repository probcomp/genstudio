import genstudio.plot as Plot

html = Plot.Hiccup


#
def on_click(event):
    print(f"Clicked on: {event['id']}")


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
                "onClick": lambda _: on_click({"id": "red"}),
            },
            "Red",
        ],
        [
            "div#blue.box",
            {
                "style": {"backgroundColor": "blue"},
                "onClick": lambda _: on_click({"id": "blue"}),
                "onMouseEnter": lambda _: print("mouseenter blue"),
            },
            "Blue",
        ],
        [
            "div#green.box",
            {
                "style": {"backgroundColor": "green"},
                "onClick": lambda _: on_click({"id": "green"}),
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
