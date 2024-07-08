import genstudio.plot as Plot

#
x = Plot.Hiccup(
    "div",
    {
        "style": {
            "backgroundColor": "pink",
            "fontSize": "20px",
            "width": "100px",
            "height": "100px",
        }
    },
    "Hello, world.",
)
x

x.save_image("scratch/images_test.png")
