# %%
import json
from pathlib import Path
from typing import Any, Dict, Optional

import genstudio.util as util
from genstudio.layout import JSCall
from genstudio.plot_spec import MarkSpec, PlotSpec

PlotOptions = dict[str, Any] | JSCall

ELIDE = ["plot"]

OBSERVABLE_PLOT_METADATA: Dict[str, Any] = json.load(
    open(util.PARENT_PATH / "scripts" / "observable_plot_metadata.json")
)
OBSERVABLE_FNS: Dict[str, Any] = {
    k: v for k, v in OBSERVABLE_PLOT_METADATA["entries"].items() if k not in ELIDE
}
OBSERVABLE_VERSION: str = OBSERVABLE_PLOT_METADATA["version"]


def get_function_def(path: str, func_name: str) -> Optional[str]:
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

    while start_index > 0:
        line = lines[start_index - 1].strip()
        if line.startswith("@"):  # This line is a decorator
            start_index -= 1  # Update the start index to include the decorator
        else:
            break

    # Find the end of the function by looking for a line that is not indented
    end_index = next(
        (
            i
            for i, line in enumerate(lines[start_index + 1 :], start_index + 1)
            if not line.startswith((" ", "\t", ")", "#", '"""'))
        ),
        None,
    )
    # If the end is not found, assume the function goes until the end of the file
    end_index = end_index or len(lines)
    return "\n".join(lines[start_index:end_index])


# Templates for inclusion in output


def FN_MARK_WITH_OPTIONAL_DATA(*args, **kwargs: Any) -> PlotSpec:
    """DOC"""
    # This function accepts the following argument combinations:
    # 1. (data, options)
    # 2. (data, options=...)
    # 3. (options_dict)
    # 4. (data)
    # 5. (**kwargs)
    # 6. ()
    options = kwargs.pop("options", None)
    if len(args) == 2:
        data, options = args
    elif len(args) == 1 and options is not None:
        data = args[0]
    elif len(args) == 1 and isinstance(args[0], dict):
        data = None
        options = args[0]
    elif len(args) == 1:
        data = args[0]
        options = {}
    elif len(args) == 0:
        data = None
        options = {}
    else:
        raise ValueError("Invalid arguments")
    if data is not None:
        if isinstance(options, dict):
            merged_options = {**options, **kwargs}
        else:
            if kwargs:
                raise ValueError("Cannot use kwargs when options is not a dict")
            merged_options = options
        return PlotSpec(
            {
                "marks": [
                    JSCall(
                        "Plot.FN_MARK_WITH_OPTIONAL_DATA",
                        [data, merged_options],
                    )
                ]
            }
        )
    return PlotSpec(
        {
            "marks": [
                JSCall(
                    "Plot.FN_MARK_WITH_OPTIONAL_DATA",
                    [{**(options or {}), **kwargs}],
                )
            ]
        }
    )


def FN_MARK(
    data: Any,
    options: PlotOptions = {},
    **kwargs: Any,
) -> PlotSpec:
    """DOC"""
    if isinstance(options, dict):
        merged_options = {**options, **kwargs}
    else:
        if kwargs:
            raise ValueError("Cannot use kwargs when options is not a dict")
        merged_options = options
    return PlotSpec(MarkSpec("FN_MARK", data, merged_options))


def FN_OTHER(*args: Any) -> JSCall:
    """DOC"""
    return JSCall("Plot.FN_OTHER", args)


sources: Dict[str, Optional[str]] = {
    name: get_function_def("scripts/make_observable_plot_defs.py", name)
    for name in ["FN_MARK_WITH_OPTIONAL_DATA", "FN_MARK", "FN_OTHER"]
}


def def_source(name: str, meta: Dict[str, Any]) -> str:
    kind = meta.get("kind")
    doc = meta.get("doc")
    variant: Optional[str] = None
    if name in [
        "axisX",
        "axisY",
        "hexgrid",
        "grid",
        "gridX",
        "gridY",
        "gridFx",
        "gridFy",
        "frame",
    ]:
        variant = "FN_MARK_WITH_OPTIONAL_DATA"
    elif kind == "marks":
        variant = "FN_MARK"
    else:
        variant = "FN_OTHER"

    source_code = sources.get(variant)
    if source_code is None:
        raise ValueError(f"Source code for variant '{variant}' not found.")

    source_code = source_code.replace(variant, name)
    source_code = source_code.replace(
        '"""DOC"""', f"""\"\"\"\n{doc}\n\"\"\"""" if doc else ""
    )

    return source_code


plot_defs = "\n\n\n".join(
    [def_source(name, meta) for name, meta in sorted(OBSERVABLE_FNS.items())]
)

plot_defs_module = f"""# Generated from version {OBSERVABLE_VERSION} of Observable Plot

from genstudio.layout import JSCall
from genstudio.plot_spec import MarkSpec, PlotSpec
from typing import Any, Dict

PlotOptions = dict[str, Any] | JSCall


{plot_defs}

"""
plot_defs_module

# %%

with open(util.PARENT_PATH / "plot_defs.py", "w") as f:
    f.write(plot_defs_module)

# %%
import_statement = "from genstudio.plot_defs import " + ", ".join(
    sorted(OBSERVABLE_FNS.keys())
)
import_statement

# %%
