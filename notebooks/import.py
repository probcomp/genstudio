"""
# GenStudio JavaScript Import Guide

This module demonstrates the different ways to import and use JavaScript code in GenStudio.
The import system allows both CDN/URL modules and inline source code, with different
strengths and tradeoffs for each approach.

## Import Styles

### 1. CDN/URL Modules
Use when loading third-party libraries that don't need to interact with React components:
```python
Plot.importModule("vega", url="https://cdn.skypack.dev/vega@5.22.1")
```

Strengths:
- Load large third-party libraries
- Versioned dependencies
- CDN caching benefits

Limitations:
- Can't share React instance with main bundle
- No access to bundle's dependencies
- Subject to CORS restrictions

### 2. CommonJS-style Source
Use for utilities and helpers that need access to bundle dependencies:
```python
Plot.importSource(
    "utils",
    '''
    const { format } = d3;  # Has access to bundle's d3

    function formatData(data) {
        return data.map(format('.2f'));
    }
    module.exports = { formatData };
    '''
)
```

Strengths:
- Full access to bundle dependencies (React, d3, Plot)
- Simpler syntax for basic utilities
- Access to previously imported modules

### 3. ES Module-style Source
Use for React components and modern JavaScript code:
```python
Plot.importSource(
    "MyComponent",
    '''
    import * as React from 'react';  # Uses bundle's React
    import * as d3 from 'd3';       # Uses bundle's d3

    export function Chart({data}) {
        return <div>{data.join(', ')}</div>;
    }
    ''',
    module=True  # Enable ES Module syntax
)
```

Strengths:
- Modern module syntax
- Cleaner import statements
- Future JSX support
- Uses bundle dependencies

## Key Features

1. Dependency Order
   - Imports are processed in order
   - Later sources can access earlier imports
   - Bundle dependencies always available

2. Scope Sharing
   All imports share the same environment:
   ```python
   Plot.importSource("a", "module.exports = { x: 1 }")
   Plot.importSource("b", "module.exports = { y: a.x + 1 }")
   Plot.importSource("c", "export const z = b.y + 1", module=True)
   ```

3. Bundle Integration
   - Access to React, d3, Plot from bundle
   - Consistent React instance for components
   - Shared dependencies reduce bundle size

## Limitations

1. ES Module Imports
   - Can't import from arbitrary URLs in source modules
   - Must use bundle dependencies or previously imported modules

2. Build Process
   - No build-time transformations (yet)
   - JSX support planned but not implemented
   - No dependency resolution

3. Security
   - Code is evaluated at runtime
   - Need to trust source code
   - Limited sandboxing

## Best Practices

1. Module Organization
   ```python
   (
       # Third-party libraries first
       Plot.importModule("lib", "https://cdn.../lib.js")

       # Then utilities
       | Plot.importSource("utils", "...")

       # Then components
       | Plot.importSource("components", "...", module=True)

       # Then use them
       | Plot.dot(data)
   )
   ```

2. Dependency Management
   - Use CDN imports for large third-party libraries
   - Use source imports for custom code
   - Keep React components in ES module style
   - Share utilities via CommonJS style
"""

import genstudio.plot as Plot
import os
import pathlib


# Get current file path
def load(filename):
    """Load a file relative to the current script's directory and return its contents"""
    __dirname__ = pathlib.Path(__file__).parent.absolute()
    filepath = os.path.join(__dirname__, filename)
    with open(filepath, "r") as f:
        return f.read()


# load("some_module.js")

# Example usage patterns:
(
    Plot.Import("vega", {"url": "https://cdn.skypack.dev/vega@5.22.1"})
    | Plot.Import(
        "test",
        """
        exports.greeting = function(name) { return `Hello, ${name}!`}
        """,
    )
    | Plot.Import(
        "MyComponent",
        """
        exports.MyChart = function(data) {
            return React.createElement("div", {}, Object.entries(data).join(', '));
        }
        """,
    )
    | Plot.dot([[1, 1]])
    | Plot.js("MyComponent.MyChart({'foo': 'bar'})")
    | Plot.js("test.greeting('fuzzy rabbit')")
    | Plot.js("Object.keys(vega)[0]")
)
