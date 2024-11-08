import unittest

import genstudio.plot as Plot
from genstudio.layout import JSCall
from genstudio.plot_spec import MarkSpec, PlotSpec
from genstudio.widget import Widget


class TestPlot(unittest.TestCase):
    def setUp(self):
        self.xs = [1, 2, 3, 4, 5]
        self.ys = [2, 3, 2, 1, 8]

    def test_PlotSpec_init(self):
        ps = Plot.new()
        self.assertIsInstance(ps, PlotSpec)
        self.assertEqual(len(ps.layers), 0)

        ps = Plot.dot({"x": self.xs, "y": self.ys})
        self.assertEqual(len(ps.layers), 1)
        self.assertIsInstance(ps.layers[0], MarkSpec)

        ps = Plot.new(width=100)
        self.assertEqual(len(ps.layers), 1)
        self.assertEqual(ps.layers[0], {"width": 100})

        # Test multiple arguments
        ps = Plot.new(
            Plot.dot({"x": self.xs, "y": self.ys}),
            Plot.line({"x": self.xs, "y": self.ys}),
            width=100,
        )
        self.assertEqual(len(ps.layers), 3)
        self.assertIsInstance(ps.layers[0], MarkSpec)
        self.assertIsInstance(ps.layers[1], MarkSpec)
        self.assertEqual(ps.layers[2], {"width": 100})

    def test_PlotSpec_add(self):
        ps1 = Plot.new(Plot.dot({"x": self.xs, "y": self.ys}), width=100)
        ps2 = Plot.new(Plot.line({"x": self.xs, "y": self.ys}), height=200)

        ps3 = ps1 + ps2
        self.assertEqual(len(ps3.layers), 4)  # dot, width, line, height
        self.assertIn({"width": 100}, ps3.layers)
        self.assertIn({"height": 200}, ps3.layers)

        ps4 = ps1 + Plot.text("foo")
        self.assertEqual(len(ps4.layers), 3)  # dot, width, text

        ps5 = ps1 + {"color": "red"}
        self.assertIn({"color": "red"}, ps5.layers)

        # Test right addition
        ps6 = {"color": "blue"} + ps1
        self.assertIn({"color": "blue"}, ps6.layers)
        self.assertEqual(ps6.layers[0], {"color": "blue"})

    def test_PlotSpec_widget(self):
        ps = Plot.new(Plot.dot({"x": self.xs, "y": self.ys}), width=100)
        plot = ps.widget()
        self.assertIsInstance(plot, Widget)

    def test_sugar(self):
        ps = Plot.new() + Plot.grid()
        self.assertIn({"grid": True}, ps.layers)

        ps = Plot.new() + Plot.colorLegend()
        self.assertIn({"color": {"legend": True}}, ps.layers)

        ps = Plot.new() + Plot.clip()
        self.assertIn({"clip": True}, ps.layers)

        ps = Plot.new() + Plot.title("My Plot")
        self.assertIn({"title": "My Plot"}, ps.layers)

        ps = Plot.new() + Plot.subtitle("Subtitle")
        self.assertIn({"subtitle": "Subtitle"}, ps.layers)

        ps = Plot.new() + Plot.caption("Caption")
        self.assertIn({"caption": "Caption"}, ps.layers)

        ps = Plot.new() + Plot.width(500)
        self.assertIn({"width": 500}, ps.layers)

        ps = Plot.new() + Plot.height(300)
        self.assertIn({"height": 300}, ps.layers)

        ps = Plot.new() + Plot.size(400)
        self.assertIn({"width": 400, "height": 400}, ps.layers)

        ps = Plot.new() + Plot.size(400, 300)
        self.assertIn({"width": 400, "height": 300}, ps.layers)

        ps = Plot.new() + Plot.aspect_ratio(1.5)
        self.assertIn({"aspectRatio": 1.5}, ps.layers)

        ps = Plot.new() + Plot.inset(10)
        self.assertIn({"inset": 10}, ps.layers)

        ps = Plot.new() + Plot.colorScheme("blues")
        self.assertIn({"color": {"scheme": "blues"}}, ps.layers)

        ps = Plot.new() + Plot.domainX([0, 10])
        self.assertIn({"x": {"domain": [0, 10]}}, ps.layers)

        ps = Plot.new() + Plot.domainY([0, 10])
        self.assertIn({"y": {"domain": [0, 10]}}, ps.layers)

        ps = Plot.new() + Plot.domain([0, 10], [0, 5])
        self.assertIn({"x": {"domain": [0, 10]}, "y": {"domain": [0, 5]}}, ps.layers)

        ps = Plot.new() + Plot.color_map({"A": "red", "B": "blue"})
        self.assertIn({"color_map": {"A": "red", "B": "blue"}}, ps.layers)

        ps = Plot.new() + Plot.margin(10)
        self.assertIn({"margin": 10}, ps.layers)

        ps = Plot.new() + Plot.margin(10, 20)
        self.assertIn(
            {
                "marginTop": 10,
                "marginBottom": 10,
                "marginLeft": 20,
                "marginRight": 20,
            },
            ps.layers,
        )

        ps = Plot.new() + Plot.margin(10, 20, 30)
        self.assertIn(
            {
                "marginTop": 10,
                "marginLeft": 20,
                "marginRight": 20,
                "marginBottom": 30,
            },
            ps.layers,
        )

        ps = Plot.new() + Plot.margin(10, 20, 30, 40)
        self.assertIn(
            {
                "marginTop": 10,
                "marginRight": 20,
                "marginBottom": 30,
                "marginLeft": 40,
            },
            ps.layers,
        )

    def test_plot_new(self):
        ps = Plot.new(Plot.dot({"x": self.xs, "y": self.ys}))
        self.assertIsInstance(ps, PlotSpec)
        self.assertEqual(len(ps.layers), 1)
        self.assertIsInstance(ps.layers[0], MarkSpec)

    def test_plot_function_docs(self):
        for mark in ["dot", "line", "rectY", "area", "barX", "barY", "text"]:
            self.assertIsInstance(getattr(Plot, mark).__doc__, str)

    def test_plot_options_merge(self):
        options1 = {"width": 500, "color": {"scheme": "reds"}}
        options2 = {"height": 400, "color": {"legend": True}}

        ps = Plot.new() + options1 + options2

        self.assertIn(options1, ps.layers)
        self.assertIn(options2, ps.layers)

        # Ensure the original options dictionaries are not mutated
        self.assertEqual(options1, {"width": 500, "color": {"scheme": "reds"}})
        self.assertEqual(options2, {"height": 400, "color": {"legend": True}})

    def test_mark_spec(self):
        ms = MarkSpec("dot", {"x": self.xs, "y": self.ys}, {"fill": "red"})
        self.assertIsInstance(ms._state_key, str)
        self.assertIsInstance(ms.ast, JSCall)
        self.assertEqual(ms.for_json(), ms.ast)

    def test_plot_spec_for_json(self):
        ps = Plot.new(Plot.dot({"x": self.xs, "y": self.ys}), width=100)
        json_data = ps.for_json()
        self.assertIsInstance(json_data, JSCall)
        self.assertEqual(json_data.path, "PlotSpec")


if __name__ == "__main__":
    unittest.main()
