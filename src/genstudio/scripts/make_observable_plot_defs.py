# %%
import json
from typing import Dict, Any, List, Optional, Union

import genstudio.util as util
from genstudio.js_modules import JSCall
from genstudio.plot_spec import PlotSpec

from pathlib import Path

OBSERVABLE_PLOT_METADATA: Dict[str, Any] = json.load(
    open(util.PARENT_PATH / "scripts" / "observable_plot_metadata.json")
)
OBSERVABLE_FNS: Dict[str, Any] = OBSERVABLE_PLOT_METADATA["entries"]
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


def FN_VALUELESS(
    options: Dict[str, Any] = {}, **kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """DOC"""
    return JSCall("Plot", "FN_VALUELESS", [{**options, **kwargs}])


def FN_MARK(
    values: Union[Dict[str, Any], List[Any]],
    options: Dict[str, Any] = {},
    **kwargs: Dict[str, Any],
) -> PlotSpec:
    """DOC"""
    return PlotSpec(
        JSCall("View", "MarkSpec", ["FN_MARK", values, {**options, **kwargs}])
    )


def FN_OTHER(*args: Union[Any, List[Any]]) -> Dict[str, Any]:
    """DOC"""
    return JSCall("Plot", "FN_OTHER", args)


sources: Dict[str, Optional[str]] = {
    name: get_function_def("scripts/make_observable_plot_defs.py", name)
    for name in ["FN_VALUELESS", "FN_MARK", "FN_OTHER"]
}


def def_source(name: str, meta: Dict[str, Any]) -> str:
    kind = meta.get("kind")
    doc = meta.get("doc")
    variant: Optional[str] = None
    if name in ["hexgrid", "grid", "gridX", "gridY", "gridFx", "gridFy", "frame"]:
        variant = "FN_VALUELESS"
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

plot_defs = "\n\n\n".join([def_source(name, meta) for name, meta in sorted(OBSERVABLE_FNS.items())])

plot_defs_module = f"""# Generated from version {OBSERVABLE_VERSION} of Observable Plot

from genstudio.js_modules import JSCall
from genstudio.plot_spec import PlotSpec


{plot_defs}

"""
plot_defs_module

with open(util.PARENT_PATH / "plot_defs.py", "w") as f:
    f.write(plot_defs_module)

# %%
import_statement = "from genstudio.plot_defs import " + ", ".join(
    sorted(OBSERVABLE_FNS.keys())
)
import_statement

# %%
