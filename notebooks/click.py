import genstudio.plot as Plot

(
    Plot.initialState({"count": 0})
    | [
        "div.flex.flex-col.items-center.gap-4.p-8",
        [
            "div.text-4xl.font-bold",
            Plot.js("$state.count"),
        ],
        [
            "button.px-4.py-2.bg-blue-500.text-white.rounded-md.hover:bg-blue-600",
            {
                "onClick": Plot.js("""
                                   () => $state.count += 1
                                   """)
            },
            "Increment",
        ],
    ]
)
