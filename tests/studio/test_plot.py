# %%

import genstudio.plot as Plot
from genstudio.widget import Widget
from genstudio.plot_spec import PlotSpec, MarkSpec
from genstudio.js_modules import JSCall

xs = [1, 2, 3, 4, 5]
ys = [2, 3, 2, 1, 8]


def test_PlotSpec_init():
    ps = Plot.new()
    assert isinstance(ps, PlotSpec)
    assert len(ps.layers) == 0

    ps = Plot.dot({"x": xs, "y": ys})
    assert len(ps.layers) == 1
    assert isinstance(ps.layers[0], MarkSpec)

    ps = Plot.new(width=100)
    assert len(ps.layers) == 1
    assert ps.layers[0] == {"width": 100}

    # Test multiple arguments
    ps = Plot.new(
        Plot.dot({"x": xs, "y": ys}), Plot.line({"x": xs, "y": ys}), width=100
    )
    assert len(ps.layers) == 3
    assert isinstance(ps.layers[0], MarkSpec)
    assert isinstance(ps.layers[1], MarkSpec)
    assert ps.layers[2] == {"width": 100}


def test_PlotSpec_add():
    ps1 = Plot.new(Plot.dot({"x": xs, "y": ys}), width=100)
    ps2 = Plot.new(Plot.line({"x": xs, "y": ys}), height=200)

    ps3 = ps1 + ps2
    assert len(ps3.layers) == 4  # dot, width, line, height
    assert {"width": 100} in ps3.layers
    assert {"height": 200} in ps3.layers

    ps4 = ps1 + Plot.text("foo")
    assert len(ps4.layers) == 3  # dot, width, text

    ps5 = ps1 + {"color": "red"}
    assert {"color": "red"} in ps5.layers

    # Test right addition
    ps6 = {"color": "blue"} + ps1
    assert {"color": "blue"} in ps6.layers
    assert ps6.layers[0] == {"color": "blue"}


def test_PlotSpec_widget():
    ps = Plot.new(Plot.dot({"x": xs, "y": ys}), width=100)
    plot = ps.widget()
    assert isinstance(plot, Widget)


def test_sugar():
    ps = Plot.new() + Plot.grid()
    assert {"grid": True} in ps.layers

    ps = Plot.new() + Plot.colorLegend()
    assert {"color": {"legend": True}} in ps.layers

    ps = Plot.new() + Plot.clip()
    assert {"clip": True} in ps.layers

    ps = Plot.new() + Plot.title("My Plot")
    assert {"title": "My Plot"} in ps.layers

    ps = Plot.new() + Plot.subtitle("Subtitle")
    assert {"subtitle": "Subtitle"} in ps.layers

    ps = Plot.new() + Plot.caption("Caption")
    assert {"caption": "Caption"} in ps.layers

    ps = Plot.new() + Plot.width(500)
    assert {"width": 500} in ps.layers

    ps = Plot.new() + Plot.height(300)
    assert {"height": 300} in ps.layers

    ps = Plot.new() + Plot.size(400)
    assert {"width": 400, "height": 400} in ps.layers

    ps = Plot.new() + Plot.size(400, 300)
    assert {"width": 400, "height": 300} in ps.layers

    ps = Plot.new() + Plot.aspect_ratio(1.5)
    assert {"aspectRatio": 1.5} in ps.layers

    ps = Plot.new() + Plot.inset(10)
    assert {"inset": 10} in ps.layers

    ps = Plot.new() + Plot.color_scheme("blues")
    assert {"color": {"scheme": "blues"}} in ps.layers

    ps = Plot.new() + Plot.domainX([0, 10])
    assert {"x": {"domain": [0, 10]}} in ps.layers

    ps = Plot.new() + Plot.domainY([0, 10])
    assert {"y": {"domain": [0, 10]}} in ps.layers

    ps = Plot.new() + Plot.domain([0, 10], [0, 5])
    assert {"x": {"domain": [0, 10]}, "y": {"domain": [0, 5]}} in ps.layers

    ps = Plot.new() + Plot.color_map({"A": "red", "B": "blue"})
    assert {"color_map": {"A": "red", "B": "blue"}} in ps.layers

    ps = Plot.new() + Plot.margin(10)
    assert {"margin": 10} in ps.layers

    ps = Plot.new() + Plot.margin(10, 20)
    assert {
        "marginTop": 10,
        "marginBottom": 10,
        "marginLeft": 20,
        "marginRight": 20,
    } in ps.layers

    ps = Plot.new() + Plot.margin(10, 20, 30)
    assert {
        "marginTop": 10,
        "marginLeft": 20,
        "marginRight": 20,
        "marginBottom": 30,
    } in ps.layers

    ps = Plot.new() + Plot.margin(10, 20, 30, 40)
    assert {
        "marginTop": 10,
        "marginRight": 20,
        "marginBottom": 30,
        "marginLeft": 40,
    } in ps.layers


def test_plot_new():
    ps = Plot.new(Plot.dot({"x": xs, "y": ys}))
    assert isinstance(ps, PlotSpec)
    assert len(ps.layers) == 1
    assert isinstance(ps.layers[0], MarkSpec)


def test_plot_function_docs():
    for mark in ["dot", "line", "rectY", "area", "barX", "barY", "text"]:
        assert isinstance(getattr(Plot, mark).__doc__, str)


def test_plot_options_merge():
    options1 = {"width": 500, "color": {"scheme": "reds"}}
    options2 = {"height": 400, "color": {"legend": True}}

    ps = Plot.new() + options1 + options2

    assert options1 in ps.layers
    assert options2 in ps.layers

    # Ensure the original options dictionaries are not mutated
    assert options1 == {"width": 500, "color": {"scheme": "reds"}}
    assert options2 == {"height": 400, "color": {"legend": True}}


def test_mark_spec():
    ms = MarkSpec("dot", {"x": xs, "y": ys}, {"fill": "red"})
    assert isinstance(ms.id, str)
    assert isinstance(ms.ast, JSCall)
    assert ms.cache_id() == ms.id
    assert ms.for_json() == ms.ast


def test_plot_spec_for_json():
    ps = Plot.new(Plot.dot({"x": xs, "y": ys}), width=100)
    json_data = ps.for_json()
    assert isinstance(json_data, JSCall)
    assert json_data["module"] == "View"
    assert json_data["name"] == "PlotSpec"


def run_tests():
    test_PlotSpec_init()
    test_PlotSpec_add()
    test_PlotSpec_widget()
    test_sugar()
    test_plot_new()
    test_plot_function_docs()
    test_plot_options_merge()
    test_mark_spec()
    test_plot_spec_for_json()
    print("All tests passed!")


run_tests()

# %%
