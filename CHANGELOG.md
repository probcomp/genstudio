### [2024.09.001] - Sep 11, 2024

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

### [2024.08.009] - Aug 28, 2024

#### New Features
- add cycle option to Slider/Reactive/Frames

#### Documentation
- Bylight example

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
