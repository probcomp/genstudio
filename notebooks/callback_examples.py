import genstudio.plot as Plot

html = Plot.Hiccup


def on_click(event):
    print(f"Clicked on: {event['id']} at x position {event['clientX']}")


# Create a Hiccup structure with interactive divs
hiccup = html(
    "div.container",
    ["h1", "Interactive Div Example with Click and Mouse Move Events"],
    [
        "p",
        "Click on a colored div to log its ID. Move the mouse over a div to display its X coordinate.",
    ],
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
            "This example demonstrates Hiccup HTML with onClick and onMouseMove callbacks on div elements.",
        ],
    ],
)

hiccup


# Create a list of colors
colors = ["Red", "Blue", "Green", "Yellow", "Purple"]

# Create a Hiccup structure with interactive divs using list comprehension
color_hiccup = html(
    "div.container",
    ["h1", "Interactive Color List Example with Lambda Callbacks"],
    ["p", "Click on a color box to print its name to the console."],
    [
        "div",
        *[
            [
                "div.color-box",
                {
                    "style": {
                        "backgroundColor": color.lower(),
                        "margin": "5px",
                        "padding": "10px",
                        "cursor": "pointer",
                    },
                    "onClick": lambda e, c=color: print(f"Clicked on {c}"),
                },
                color,
            ]
            for color in colors
        ],
    ],
    [
        "div.footer",
        [
            "p",
            "This example demonstrates using list comprehension to create interactive elements with lambda callbacks.",
        ],
    ],
)

color_hiccup
