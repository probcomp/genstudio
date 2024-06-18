# %%

import genstudio.plot as Plot
from genstudio.widget import Widget


xs = [1, 2, 3, 4, 5]
ys = [2, 3, 2, 1, 8]


def test_plotspec_init():
    ps = Plot.new()
    assert ps.spec == {"marks": []}

    ps = Plot.dot({"x": xs, "y": ys})
    assert len(ps.spec["marks"]) == 1
    assert "pyobsplot-type" in ps.spec["marks"][0]

    ps = Plot.new(width=100)
    assert ps.spec == {"marks": [], "width": 100}


def test_plotspec_add():
    ps1 = Plot.new(Plot.dot({"x": xs, "y": ys}), width=100)
    ps2 = Plot.new(Plot.line({"x": xs, "y": ys}), height=200)

    ps3 = ps1 + ps2
    assert len(ps3.spec["marks"]) == 2
    assert ps3.spec["width"] == 100
    assert ps3.spec["height"] == 200

    ps4 = ps1 + Plot.text("foo")
    assert len(ps4.spec["marks"]) == 2

    ps5 = ps1 + {"color": "red"}
    assert ps5.spec["color"] == "red"

    try:
        ps1 + "invalid"  # type: ignore
        assert False, "Expected TypeError"
    except TypeError:
        pass


def test_plotspec_plot():
    ps = Plot.new(Plot.dot({"x": xs, "y": ys}), width=100)
    assert ps.spec["width"] == 100
    plot = ps.plot()
    assert isinstance(plot, Widget)

    # Check plot is cached
    plot2 = ps.plot()
    assert plot is plot2


def test_sugar():
    ps = Plot.new() + Plot.grid_x()
    assert ps.spec["x"]["grid"] is True

    ps = Plot.new() + Plot.grid()
    assert ps.spec["grid"] is True

    ps = Plot.new() + Plot.color_legend()
    assert ps.spec["color"]["legend"] is True

    ps = Plot.new() + Plot.clip()
    assert ps.spec["clip"] is True

    ps = Plot.new() + Plot.aspect_ratio(0.5)
    assert ps.spec["aspectRatio"] == 0.5

    ps = Plot.new() + Plot.color_scheme("blues")
    assert ps.spec["color"]["scheme"] == "blues"

    ps = Plot.new() + Plot.domainX([0, 10])
    assert ps.spec["x"]["domain"] == [0, 10]

    ps = Plot.new() + Plot.domainY([0, 20])
    assert ps.spec["y"]["domain"] == [0, 20]

    ps = Plot.new() + Plot.domain([0, 10], [0, 20])
    assert ps.spec["x"]["domain"] == [0, 10]
    assert ps.spec["y"]["domain"] == [0, 20]

    ps = Plot.new() + Plot.margin(10)
    assert ps.spec["margin"] == 10

    ps = Plot.new() + Plot.margin(10, 20)
    assert ps.spec["marginTop"] == 10
    assert ps.spec["marginBottom"] == 10
    assert ps.spec["marginLeft"] == 20
    assert ps.spec["marginRight"] == 20

    ps = Plot.new() + Plot.margin(10, 20, 30)
    assert ps.spec["marginTop"] == 10
    assert ps.spec["marginLeft"] == 20
    assert ps.spec["marginRight"] == 20
    assert ps.spec["marginBottom"] == 30

    ps = Plot.new() + Plot.margin(10, 20, 30, 40)
    assert ps.spec["marginTop"] == 10
    assert ps.spec["marginRight"] == 20
    assert ps.spec["marginBottom"] == 30
    assert ps.spec["marginLeft"] == 40


def mark_name(mark):
    return mark["args"][0]


def test_plot_new():
    ps = Plot.new(Plot.dot({"x": xs, "y": ys}))
    assert isinstance(ps, Plot.PlotSpec)
    assert len(ps.spec["marks"]) == 1
    assert mark_name(ps.spec["marks"][0]) == "dot"


def test_plotspec_reset():
    ps = Plot.new(Plot.dot({"x": xs, "y": ys}), width=100)
    assert ps.spec["width"] == 100
    assert len(ps.spec["marks"]) == 1

    ps.reset(marks=[Plot.rectY(xs)], height=200)
    assert ps.spec.get("width", None) is None  # width removed
    assert ps.spec["height"] == 200
    assert len(ps.spec["marks"]) == 1
    assert mark_name(ps.spec["marks"][0]) == "rectY"


def test_plotspec_update():
    ps = Plot.new(Plot.dot({"x": xs, "y": ys}), width=100)
    assert ps.spec["width"] == 100
    assert len(ps.spec["marks"]) == 1

    ps.update(Plot.rectY(xs), height=200)
    assert ps.spec["width"] == 100
    assert ps.spec["height"] == 200
    assert len(ps.spec["marks"]) == 2
    assert mark_name(ps.spec["marks"][0]) == "dot"
    assert mark_name(ps.spec["marks"][1]) == "rectY"


def test_plot_function_docs():
    for mark in ["dot", "line", "rectY"]:
        assert isinstance(getattr(Plot, mark).__doc__, str)


def test_plot_options_merge_nested():
    options1 = {"width": 500, "style": {"color": "red", "border": {"width": 2}}}
    options2 = {"height": 400, "style": {"border": {"color": "blue"}}}

    # Create a new plot with merged options
    ps = Plot.new() + options1 + options2

    # Check that the plot spec has the merged options
    assert ps.spec["width"] == 500
    assert ps.spec["height"] == 400
    assert ps.spec["style"]["color"] == "red"
    assert ps.spec["style"]["border"]["width"] == 2
    assert ps.spec["style"]["border"]["color"] == "blue"

    # Ensure the original options dictionaries are not mutated
    assert options1 == {"width": 500, "style": {"color": "red", "border": {"width": 2}}}
    assert options2 == {"height": 400, "style": {"border": {"color": "blue"}}}


def run_tests():
    test_plotspec_init()
    test_plotspec_add()
    test_plotspec_plot()
    test_sugar()
    test_plot_new()
    test_plotspec_reset()
    test_plotspec_update()
    test_plot_function_docs()
    test_plot_options_merge_nested()
    print("All tests passed!")


run_tests()

# %%
