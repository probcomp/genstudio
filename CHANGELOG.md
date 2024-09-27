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
