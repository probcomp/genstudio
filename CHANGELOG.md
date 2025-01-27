### [2025.1.11] - Jan 27, 2025

- deps: React 19
- Slider sets initial state if undefined
- Slider disposes interval correctly

### [2025.1.10] - Jan 27, 2025

- add maxWidth option to plots
- rename `yarn watch` to `yarn dev`

### [2025.1.9] - Jan 24, 2025

- release.py: remove padding from version numbers
- support alpha releases
- publish docs to github pages

### [2025.1.7] - Jan 23, 2025

- release to PyPI instead of artifact registry

### [2025.1.4] - Jan 16, 2025

- support html display mode

### [2025.1.3] - Jan 16, 2025

- FIX: breaking changes in esm.sh broke genstudio's js bundle

### [2024.12.4] - Dec 06, 2024

- fix replaceBuffers ndarray handling
- clarify Plot.js vs Plot.Import
- return html/svg elements directly in hiccup/html
- bring genstudio.api into Plot.js scope
- use a `source=` param with https?: and path: prefixes possible
- add Plot.Import

### [2024.12.3] - Dec 01, 2024

- Add html to evaluation env

### [2024.12.2] - Dec 01, 2024

- add React to evaluation env

### [2024.12.1] - Dec 01, 2024

- Plot.events: add event.key for draw events to distinguish lines in 32 bits

### [2024.11.23] - Nov 30, 2024

- Plot.events: only preventDefault on mousedown when in rectangle and dragging is enabled
- Slider: don't clobber initialState value

### [2024.11.22] - Nov 30, 2024

- in python, apply updates immutably
- add tab example to llms.py
- event.preventDefault() in Plot.events handleMouseDown
- relax types, because JSCode can be anywhere
- fix Row/Column ordering

### [2024.11.21] - Nov 28, 2024

- add types

### [2024.11.20] - Nov 28, 2024

- support dot notation (eg. event.value) in python callbacks
- add types

### [2024.11.16] - Nov 28, 2024

- serialize `attributes_dict` (for Pytree.dataclass, etc)
- support 'multi.segment.paths.0' in js $state, with reactivity preserved for nested changes.

### [2024.11.15] - Nov 26, 2024

- computed, transactional js state updates (GEN-867)
  - sync computed state with python
  - ensure that computed state and onChange listeners are transacted together
  - ensure that python sees a consistent view of js state, including computed state
- add Plot.cond and Plot.case for conditional rendering
- Plot.Slider accepts className/style, which are applied to an outer div.

### [2024.11.14] - Nov 22, 2024

- allow chained/dependent onChange listeners
- trimLeft js_source (so it's ok to put source on a new line inside `Plot.js`)

### [2024.11.13] - Nov 18, 2024

- add llms.py (instructions for LLMs)
- clean up Plot.grid() and Plot.hideAxis() argument handling
- add Plot.plot to mirror Observable Plot api
- rename Plot.domain args from xd, yd to x, y
- increase max-width of p/ul/ol in .prose
- fix Plot.events: allow onDraw and onMouseMove to co-exist
- Plot.events is now order-independent and won't block pointer events

### [2024.11.12] - Nov 12, 2024

- Plot.Slider accepts `"raf"` as an fps value
- Plot.Slider accepts `controls=["slider", "play", "fps"]` to flexibly control ui
- Plot.Slider's range option, if a single number, represents the upper bound (EXCLUSIVE) of a 0-n range, as in python's `range(n)`

### [2024.11.11] - Nov 11, 2024


- Fix pixels mark in Safari (poor foreignObject support)
- Add "className" option to plots to add classes to container
- Improve grid/hideAxis argument handling
- docs: add pixels, adev, and live edit examples to website
- docs: fix website css (twind fails, use tailwind)

### [2024.11.10] - Nov 11, 2024

- Prevent update cycles in onChange listeners

### [2024.11.9] - Nov 11, 2024

- support columnar data with array buffers

### [2024.11.8] - Nov 11, 2024

- fix asset versioning

### [2024.11.7] - Nov 11, 2024

- rename Plot.listen to Plot.onChange

### [2024.11.6] - Nov 08, 2024

- handle NaN values

### [2024.11.5] - Nov 08, 2024

- Slider: add back positional range arg

### [2024.11.4] - Nov 08, 2024

#### Bug Fixes
- binary data works in html display mode

### [2024.11.3] - Nov 08, 2024

- Run tests in CI
- bring back support for multi-dimensional arrays (serialize to js arrays + typed array leaves)
- Add a version query param to static assets on release

### [2024.11.2] - Nov 07, 2024

- Add support for binary data (in plot specifications, and updates in both directions python<>js). Numpy/jax arrays are converted to appropriate TypedArray types in js.
- Add a `Plot.pixels` mark for rendering a single image, given an array of rgb/rgba pixels and an imageWidth/imageHeight
- Use CDN for releases (vastly smaller file sizes for notebooks)
- Slider: add showFps option, improve layout

### [2024.11.1] - Nov 05, 2024

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

### [2024.10.5] - Oct 30, 2024

- use `containerWidth` instead of a React context to set widths
- improve rendering of custom "height" on a plot

### [2024.10.3] - Oct 30, 2024

- Add API documentation to website
- Add `Plot.katex`
- Plot.js supports parameters
- Improved js event data parsing

### [2024.10.2] - Oct 25, 2024

- BREAKING: rename `Plot.draw` to `Plot.events`
- add `onClick`, `onMouseMove`, and `onMouseDown` callbacks
- add `startTime` to draw event data
- support dictionaries as arguments to Plot.initial_state and widget.update_state

### [2024.10.1] - Oct 21, 2024

- add _repr_html_ to LayoutItem (for treescope)

### [2024.9.7] - Sep 27, 2024

#### Bug Fixes
- ariaLabel is a default option, not a channel

### [2024.9.6] - Sep 27, 2024

#### New Features
- `Plot.img` mark for specifying image sizes in x/y coords
- use import maps for js deps

#### Bug Fixes
- apply scale correction to Plot.render.childEvents

### [2024.9.5] - Sep 18, 2024

- deps: add pillow as a required dependency

### [2024.9.4] - Sep 17, 2024

- rename: Plot.cache -> Plot.ref
- refactor: unify $state/cache implementations
- tests: add dependency tests using the new (simplified) state store, independent of React

### [2024.9.3] - Sep 13, 2024

#### New Features
- add Plot.draw mark (onDrawStart, onDraw, onDrawEnd)
- add Plot.render.draggableChildren (onDragStart, onDrag, onDragEnd, onClick)
- add widget.update_state([CachedObject, operation, payload]) for reactively updating cached data
- add Plot.initial_state for initializing js $state variables

### [2024.9.2] - Sep 11, 2024

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

### [2024.8.10] - Aug 28, 2024

#### Bug Fixes
- Allow cache entries to reference each other (non-circular)


### [2024.8.8] - Aug 28, 2024

#### New Features
- Bylight code highlighting
- Plot.Reactive can animate, Plot.Frames accepts slider=False

#### Other Changes
- refactor: JSCall, JSCode, now inherit from LayoutItem

### [2024.8.7] - Aug 27, 2024

#### Documentation
- use bylight from a cdn
- use Google Font
- explain JSON serialization

#### Other Changes
- bump: Observable Plot 0.6.16

#### [2024.8.6] - August 26, 2024

#### New Features
- a reactive variable maintains its current value when a plot is reset, unless reactive variable definitions change

#### Documentation
- Plot.constantly for colors
- JSON serialization
- Exporting and Saving

#### Other Changes
- values => data (in arguments to Plot.{mark})

#### [2024.8.1]

- Initial release
