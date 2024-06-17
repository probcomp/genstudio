# %%
import json

import genstudio.util as util
from genstudio.js_modules import JSCall
from genstudio.plot_spec import PlotSpec

from pathlib import Path

OBSERVABLE_PLOT_METADATA = json.load(
    open(util.PARENT_PATH / "scripts" / "observable_plot_metadata.json")
)
OBSERVABLE_FNS = OBSERVABLE_PLOT_METADATA["entries"]
OBSERVABLE_VERSION = OBSERVABLE_PLOT_METADATA["version"]


def get_function_def(path, func_name):
    source = Path(util.PARENT_PATH / path).read_text()
    lines = source.split("\n")
    # Python functions start with 'def' followed by the function name and a colon
    start_index = next(
        (
            i
            for i, line in enumerate(lines)
            if line.strip().startswith(f"def {func_name}(")
        ),
        None,
    )
    if start_index is None:
        return None  # Function not found
    # Find the end of the function by looking for a line that is not indented
    end_index = next(
        (
            i
            for i, line in enumerate(lines[start_index + 1 :], start_index + 1)
            if not line.startswith((" ", "\t"))
        ),
        None,
    )
    # If the end is not found, assume the function goes until the end of the file
    end_index = end_index or len(lines)
    return "\n".join(lines[start_index:end_index])


# Templates for inclusion in output


def FN_VALUELESS(options={}, **kwargs):
    """DOC"""
    return JSCall("Plot", "FN_VALUELESS", [{**options, **kwargs}])


def FN_MARK(values, options={}, **kwargs):
    """DOC"""
    return PlotSpec(
        JSCall("View", "MarkSpec", ["FN_MARK", values, {**options, **kwargs}])
    )


def FN_OTHER(*args):
    """DOC"""
    return JSCall("Plot", "FN_OTHER", args)


sources = {
    name: get_function_def("scripts/make_observable_plot_defs.py", name)
    for name in ["FN_VALUELESS", "FN_MARK", "FN_OTHER"]
}


def def_source(name, meta):
    kind = meta["kind"]
    doc = meta["doc"]
    variant = None
    if name in ["hexgrid", "grid", "gridX", "gridY", "gridFx", "gridFy", "frame"]:
        variant = "FN_VALUELESS"
    elif kind == "marks":
        variant = "FN_MARK"
    else:
        variant = "FN_OTHER"
    return (
        sources[variant]
        .replace(variant, name)
        .replace('"""DOC"""', f"""\"\"\"\n{doc}\n\"\"\"""" if doc else "")
    )


plot_defs = f"""# Generated from version {OBSERVABLE_VERSION} of Observable Plot

from genstudio.js_modules import JSCall
from genstudio.plot_spec import PlotSpec


{"\n\n\n".join([def_source(name, meta) for name, meta in sorted(OBSERVABLE_FNS.items())])}

"""
plot_defs

with open(util.PARENT_PATH / "plot_defs.py", "w") as f:
    f.write(plot_defs)

# %%
import_statement = "from genstudio.plot_defs import " + ", ".join(
    sorted(OBSERVABLE_FNS).keys()
)
import_statement

# %%
