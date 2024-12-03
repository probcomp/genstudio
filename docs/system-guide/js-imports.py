# %% [markdown]
# # GenStudio JavaScript Import Guide
#
# Import JavaScript code from URLs, files, or inline source into your GenStudio plots.
#
# %% [markdown]
# ## Quick examples
#
# %%
# Import from CDN
import genstudio.plot as Plot

Plot.Import(url="https://cdn.skypack.dev/lodash-es", refer=["sum"]) | Plot.js(
    "sum([1, 2, 3])"
)

# %%
# Import inline source
Plot.Import(
    source="""
    export const greet = name => `Hello ${name}!`;
    """,
    refer=["greet"],
) | Plot.js("greet('world')")

# ## Things to know
# - Imports don't affect the global namespace: a `Plot.Import` only applies to the plot it is included in.
# - ES Modules (ESM) format is supported by default. CommonJS modules can be used by setting format="commonjs".
#   When using CommonJS format, previous imports are available via `genstudio.imports`.
# - Imports are processed in the order they appear in a plot.
#
# ## Import Sources
# There are three ways to provide JavaScript code to import:
#
# - `url`: Import from a CDN or web URL
# - `path`: Import from a local file (relative to current working directory)
# - `source`: Import inline JavaScript source code
#
# ## Import Options
# Control how imports are exposed in your code:
#
# - `alias`: Create a namespace object containing all exports
# - `default`: Import the default export with a specific name
# - `refer`: List of named exports to import
# - `refer_all`: Import all named exports (except those in `exclude`)
# - `rename`: Rename specific imports to avoid conflicts
# - `exclude`: List of exports to exclude when using `refer_all`
# - `format`: Module format - "esm" (default) or "commonjs"
#
# ## GenStudio API Access
# Your JavaScript code can access `genstudio.api` which  for HTML rendering (`html` tagged template)
# %%
import genstudio.plot as Plot

# %%
# CDN import showing namespace alias and selective imports
Plot.Import(
    url="https://cdn.skypack.dev/lodash-es",
    alias="_",
    refer=["flattenDeep", "partition"],
    rename={"flattenDeep": "deepFlatten"},
) | Plot.js("deepFlatten([1, [2, [3, 4]]])")
# JS equivalent:
# import * as _ from "https://cdn.skypack.dev/lodash-es"
# import { flattenDeep as deepFlatten, partition } from "https://cdn.skypack.dev/lodash-es"

# %%
# Local file import - useful for project-specific code
Plot.Import(path="docs/system-guide/sample.js", refer=["formatDate"]) | Plot.js(
    "formatDate(new Date())"
)
# JS equivalent:
# import { formatDate } from "./docs/system-guide/sample.js"

# %%
# Inline source with namespace and selective exports
Plot.Import(
    source="""
    export const add = (a, b) => a + b;
    export const subtract = (a, b) => a - b;
    export const multiply = (a, b) => a * b;
    """,
    refer_all=True,
    alias="math",
    exclude=["multiply"],
) | Plot.js("[add(5, 3), subtract(5, 3), typeof multiply, math.multiply(3, 3), ]")
# JS equivalent:
# import * as math from "[inline module]"
# import { add, subtract } from "[inline module]"

# %%
# Cherry-picking specific functions from a module
Plot.Import(
    url="https://cdn.skypack.dev/d3-scale",
    refer=["scaleLinear", "scaleLog", "scaleTime"],
) | Plot.js("scaleLinear().domain([0, 1]).range([0, 100])(0.5)")
# JS equivalent:
# import { scaleLinear, scaleLog, scaleTime } from "https://cdn.skypack.dev/d3-scale"

# %%
Plot.Import(
    source="""
    const {html} = genstudio.api;
    export const greeting = (name) => html(["div.p-5.bg-green-100", name])
    """,
    refer=["greeting"],
) | Plot.js("greeting('friend')")
# JS equivalent:
# import { greeting } from "[inline module]"

# %%
# CommonJS modules can depend on previous imports
(
    Plot.Import(
        source="""
    export const add = (a, b) => a + b;
    """,
        refer=["add"],
    )
    | Plot.Import(
        source="""
    const {add} = genstudio.imports;
    module.exports.addTwice = (x) => add(x, x);
    """,
        format="commonjs",
        refer=["addTwice"],
    )
    | Plot.js("addTwice(5)")
)
