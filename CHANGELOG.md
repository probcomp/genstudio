### [2024.11.003] - Nov 08, 2024

- Run tests in CI
- bring back support for multi-dimensional arrays (serialize to js arrays + typed array leaves)
- Add a version query param to static assets on release

### [2024.11.002] - Nov 07, 2024

- Add support for binary data (in plot specifications, and updates in both directions python<>js). Numpy/jax arrays are converted to appropriate TypedArray types in js.
- Add a `Plot.pixels` mark for rendering a single image, given an array of rgb/rgba pixels and an imageWidth/imageHeight
- Use CDN for releases (vastly smaller file sizes for notebooks)
- Slider: add showFps option, improve layout

### [2024.11.001] - Nov 05, 2024

#### Breaking Changes
- `Plot.initialState({"name": "value"})` now takes **only** a dict, rather than a single key/value.
- `Plot.html` would previously create an element if passed a string as the first argument. Now it is required to use a list, eg. `Plot.html(["div", ...content])`. This allows for wrapping primitive values (strings, numbers) in `Plot.html` in order to compose them, eg. `Plot.html("Hello, world") & ["div", {...}, "my button"])`.
- `Plot.ref` now takes a `state_key` variable instead of `id` (but we expect to use `Plot.ref` less often, now with the new state features).
- Python callbacks now take two arguments, `(widget, data)` instead of only `data`.

#### Improvements
- `Row`/`Column`/`Grid` now accept more options (eg. widths/heights).
- `Plot.initialState(...)` accepts a `sync` option, `True` to sync all variables or a set of variable names, eg `sync={"foo"}`. Synced variables will send updates from js to python automatically.
- `widget.state` is a new interface for reading synced variables (`widget.state.foo`) and updating any variable (`widget.state.update({"foo": "bar"}, ["bax", "append", 1])`).
- `Plot.listen({state_key: listener})` is a layout item which subscribes listener functions to state changes. Adding a listener for a variable implicitly sets `sync=True` for that variable.

#### Documentation
- add rgb(a) section in colors
- add interactive-density example

### [2024.10.005] - Oct 30, 2024

- use `containerWidth` instead of a React context to set widths
- improve rendering of custom "height" on a plot

### [2024.10.003] - Oct 30, 2024

- Add API documentation to website
- Add `Plot.katex`
- Plot.js supports parameters
- Improved js event data parsing

### [2024.10.002] - Oct 25, 2024

- BREAKING: rename `Plot.draw` to `Plot.events`
- add `onClick`, `onMouseMove`, and `onMouseDown` callbacks
- add `startTime` to draw event data
- support dictionaries as arguments to Plot.initial_state and widget.update_state

### [2024.10.001] - Oct 21, 2024

- add _repr_html_ to LayoutItem (for treescope)

### [2024.09.007] - Sep 27, 2024

#### Bug Fixes
- ariaLabel is a default option, not a channel

### [2024.09.006] - Sep 27, 2024

#### New Features
- `Plot.img` mark for specifying image sizes in x/y coords
- use import maps for js deps

#### Bug Fixes
- apply scale correction to Plot.render.childEvents

### [2024.09.005] - Sep 18, 2024

- deps: add pillow as a required dependency

### [2024.09.004] - Sep 17, 2024

- rename: Plot.cache -> Plot.ref
- refactor: unify $state/cache implementations
- tests: add dependency tests using the new (simplified) state store, independent of React

### [2024.09.003] - Sep 13, 2024

#### New Features
- add Plot.draw mark (onDrawStart, onDraw, onDrawEnd)
- add Plot.render.draggableChildren (onDragStart, onDrag, onDragEnd, onClick)
- add widget.update_state([CachedObject, operation, payload]) for reactively updating cached data
- add Plot.initial_state for initializing js $state variables

### [2024.09.002] - Sep 11, 2024

#### New Features
- support Tailwind (via twind)

#### Bug Fixes
- Hiccup with one child

#### Other Changes
- ci: always build docs
- slim down genstudio deps
- refactor: added api.js module
- refactor: JSRef/JSCall use path instead of module/name
- tests: added widget.test.jsx
- update_cache accepts multiple updates

### [2024.08.010] - Aug 28, 2024

#### Bug Fixes
- Allow cache entries to reference each other (non-circular)


### [2024.08.008] - Aug 28, 2024

#### New Features
- Bylight code highlighting
- Plot.Reactive can animate, Plot.Frames accepts slider=False

#### Other Changes
- refactor: JSCall, JSCode, now inherit from LayoutItem

### [2024.08.007] - Aug 27, 2024

#### Documentation
- use bylight from a cdn
- use Google Font
- explain JSON serialization

#### Other Changes
- bump: Observable Plot 0.6.16

#### [2024.08.006] - August 26, 2024

#### New Features
- a reactive variable maintains its current value when a plot is reset, unless reactive variable definitions change

#### Documentation
- Plot.constantly for colors
- JSON serialization
- Exporting and Saving

#### Other Changes
- values => data (in arguments to Plot.{mark})

#### [2024.08.001]

- Initial release
