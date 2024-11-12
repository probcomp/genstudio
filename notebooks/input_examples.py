import genstudio.plot as Plot

(
    Plot.initialState({"q": 1})
    | Plot.html(
        [
            "div.flex.gap-4.items-center.text-sm",
            [
                "input",
                {
                    "type": "range",
                    "min": 0,
                    "max": 9,
                    "value": Plot.js("$state.q"),
                    "onChange": Plot.js("(e) => $state.q = e.target.value"),
                },
            ],
            Plot.js("$state.q"),
        ]
    )
)
