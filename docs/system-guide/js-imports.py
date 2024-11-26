# %%
import genstudio.plot as Plot

# %% [markdown]
# # GenStudio JavaScript Import Guide
#
# This guide explains how to import and use JavaScript code in GenStudio.
# The system supports two main types of imports: ESM modules from URLs and direct source code imports.
#
# ## Import Types
#
# ### 1. ESM Modules from URLs
# - Import modules directly from CDNs or other URLs
# - Run in isolation from the GenStudio environment
# - Access via the specified import key
#
# Example:
# %%
(
    Plot.Import("vega", {"url": "https://cdn.skypack.dev/vega@5.22.1"})
    | Plot.js("""
              ['div.flex.flex-wrap.gap-2', ...Object.keys(vega).slice(0, 20)]
              """)
)

# ### 2. Source Code Imports
#
# #### CommonJS Format
# - Full access to GenStudio environment (React, Plot, d3)
# - Can use previously imported modules
# - Ideal for integrating with existing GenStudio components
# - Can return components using hiccup array format, or use React
# Example:
# %%
(
    Plot.Import(
        "utils",
        """
    function MyComponent({data}) {
        return ['div', {},
            ['strong', {}, "Data: "],
            data.join(', ')
        ];
    }
    module.exports = { MyComponent };
    """,
    )
    | Plot.js("utils.MyComponent({data: [1,2,3]})")
)

# %% [markdown]
# #### ESM Format
# - Uses modern JavaScript module syntax
# - Runs in isolation from GenStudio environment
# - Can return components using hiccup array format
# - Access via the specified import key

# %%
# Define a component using ESM syntax
(
    Plot.Import(
        "components",
        {
            "source": """
            // ESM modules have no access to environment
            export function Chart({data}) {
                return ['div', {},
                    ['strong', {}, "Chart: "],
                    data.join(', ')
                ];
            }
            """,
            "format": "esm",
        },
    )
    | Plot.js("components.Chart({data: [1,2,3,4,5]})")
)

# %% [markdown]
# ## Import Behavior
#
# Imports follow these key principles:
# 1. Sequential execution: Modules load in the order they appear
# 2. Dependency chain: Later imports can use earlier ones
# 3. Scoped access: Each import is available through its assigned key


# %%
(
    # First import lodash as ESM module
    Plot.Import("_", {"url": "https://cdn.skypack.dev/lodash-es"})
    | Plot.Import(
        "demo",
        """
        // Then use lodash in this CommonJS module
        function List() {
            return ['div', {},
                ['strong', {}, "Numbers: "],
                _.range(5).join(', ')  // Uses lodash imported above
            ];
        }
        module.exports = { List };
    """,
    )
    | Plot.js("demo.List()")
)

# %% [markdown]
# ## Security Considerations
#
# When using the import system:
# - Import only from trusted sources
# - Be aware that code runs without sandboxing
# - Exercise caution with external code sources
