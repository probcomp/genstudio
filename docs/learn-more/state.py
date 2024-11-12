# %% [markdown]
# # State
#
# State updates in GenStudio work bidirectionally between Python and JavaScript. Let's look at how updates flow in both directions:
#
# ## Initializing State
# Use `Plot.initialState()` to provide initial values and configure state syncing:
#
# ```python
# # Initialize state without syncing - JS updates won't be reflected in widget.state
# Plot.initialState({"count": 0, "name": "foo"})
#
# # Sync all state variables - changes in JS will update widget.state
# Plot.initialState({"count": 0}, sync=True)
#
# # Only sync specific variables - only x will update in widget.state
# Plot.initialState({"x": 0, "y": 1}, sync={"x"})
# ```
#
# ## Responding to Changes
# Use `Plot.onChange()` to register callbacks that run when state changes. Note that any variables with onChange listeners are automatically synced between JS and Python:
#
# ```python
# Plot.onChange({
#     "x": lambda widget, event: print(f"x changed to {event.value}"),
#     "y": lambda widget, event: print(f"y changed to {event.value}")
# })
#
# # x and y will now be synced and accessible via
# # widget.state.x and widget.state.y
# ```
#
# ## Python → JavaScript Updates
# When you update state from Python using `widget.state.update()` or by setting attributes directly:
#
# 1. The update is normalized into a list of `[key, operation, payload]` tuples
# 2. For synced state, the update is applied locally to `widget.state`
# 3. The update is serialized to JSON and sent to JavaScript via widget messaging
# 4. Any registered Python listeners are notified
#
# ```python
# # These all trigger Python → JS updates:
# widget.state.count = 1  # Direct attribute set
# widget.state.update({"count": 1})  # Single update
# widget.state.update({"x": 0, "y": 1})  # Multiple updates
# ```
#
# ## JavaScript → Python Updates
# When state is updated from JavaScript using `$state.update()`:
#
# 1. The update is normalized into `[key, operation, payload]` format
# 2. The update is applied locally to the JavaScript state store
# 3. For synced state keys, the update is sent back to Python
# 4. Python applies the update and notifies any listeners
#
# ```javascript
# // These trigger JS → Python updates for synced state:
# $state.count = 1  // Direct property set
# $state.update({"count": 1}) // Single update
# $state.update({"x": 0, "y": 1}) // Multiple updates
# ```
#
# ## Update Operations
# Updates support different operations beyond just setting values. These operations allow efficient updates by only sending the changes rather than entire values:
#
# In Python:
# ```python
# # Single operation update
# widget.state.update(["items", "append", "new item"])
#
# # Passing multiple updates
# widget.state.update(
#     ["items", "append", "new item"],
#     ["count", "reset", 0],
#     ["items", "setAt", [0, "first"]]
# )
# ```
#
# In JavaScript:
# ```javascript
# // Single operation update
# $state.update(["items", "append", "new item"])
#
# // Multiple operation updates - pass multiple arrays
# $state.update(
#     ["items", "append", "new item"],
#     ["count", "reset", 0],
#     ["items", "setAt", [0, "first"]]
# )
# ```
#
# Available operations:
# - `append`: Add item to end of list
# - `concat`: Join two lists together
# - `setAt`: Set value at specific index
# - `reset`: Reset value to initial state
