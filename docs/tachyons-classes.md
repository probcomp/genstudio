GenStudio uses [Tachyons](https://tachyons.io), a functional CSS framework that enables rapid UI development through composable, single-purpose utility classes. Each class typically does one thing, making it easy to combine them for desired styles (e.g., `f3 blue ma2` creates large blue text with margin).

All classes listed here are available for use with Hiccup in GenStudio, allowing efficient component styling.

## Table of Contents

1. [Box Model](#box-model)
2. [Flexbox](#flexbox)
3. [Spacing](#spacing)
4. [Typography](#typography)
5. [Colors](#colors)
6. [Borders](#borders)
7. [Layout](#layout)
8. [Positioning](#positioning)
9. [Display](#display)
10. [Widths and Heights](#widths-and-heights)
11. [Opacity](#opacity)
12. [Background](#background)
13. [Hover Effects](#hover-effects)
14. [Z-Index](#z-index)
15. [Responsive Design](#responsive-design)

## Box Model

### Box Sizing

- `.border-box`: Sets `box-sizing: border-box`
- `.content-box`: Sets `box-sizing: content-box`
- `.border-box`: Sets `box-sizing: border-box`
- `.content-box`: Sets `box-sizing: content-box`

### Aspect Ratio

- `.aspect-ratio`: Sets a parent element to establish aspect ratio context
- `.aspect-ratio--16x9`: Sets aspect ratio to 16:9
- `.aspect-ratio--9x16`: Sets aspect ratio to 9:16
- `.aspect-ratio--4x3`: Sets aspect ratio to 4:3
- `.aspect-ratio--3x4`: Sets aspect ratio to 3:4
- `.aspect-ratio--6x4`: Sets aspect ratio to 6:4
- `.aspect-ratio--4x6`: Sets aspect ratio to 4:6
- `.aspect-ratio--8x5`: Sets aspect ratio to 8:5
- `.aspect-ratio--5x8`: Sets aspect ratio to 5:8
- `.aspect-ratio--7x5`: Sets aspect ratio to 7:5
- `.aspect-ratio--5x7`: Sets aspect ratio to 5:7
- `.aspect-ratio--1x1`: Sets aspect ratio to 1:1 (square)

## Flexbox

### Flex Container

- `.flex`: Sets `display: flex`
- `.inline-flex`: Sets `display: inline-flex`
- `.flex-auto`: Sets `flex: 1 1 auto`
- `.flex-none`: Sets `flex: none`

### Flex Direction

- `.flex-column`: Sets `flex-direction: column`
- `.flex-row`: Sets `flex-direction: row`
- `.flex-column-reverse`: Sets `flex-direction: column-reverse`
- `.flex-row-reverse`: Sets `flex-direction: row-reverse`

### Flex Wrap

- `.flex-wrap`: Sets `flex-wrap: wrap`
- `.flex-nowrap`: Sets `flex-wrap: nowrap`
- `.flex-wrap-reverse`: Sets `flex-wrap: wrap-reverse`

### Align Items

- `.items-start`: Sets `align-items: flex-start`
- `.items-end`: Sets `align-items: flex-end`
- `.items-center`: Sets `align-items: center`
- `.items-baseline`: Sets `align-items: baseline`
- `.items-stretch`: Sets `align-items: stretch`

### Justify Content

- `.justify-start`: Sets `justify-content: flex-start`
- `.justify-end`: Sets `justify-content: flex-end`
- `.justify-center`: Sets `justify-content: center`
- `.justify-between`: Sets `justify-content: space-between`
- `.justify-around`: Sets `justify-content: space-around`

### Align Self

- `.self-start`: Sets `align-self: flex-start`
- `.self-end`: Sets `align-self: flex-end`
- `.self-center`: Sets `align-self: center`
- `.self-baseline`: Sets `align-self: baseline`
- `.self-stretch`: Sets `align-self: stretch`

### Order

- `.order-0`, `.order-1`, ..., `.order-8`: Sets `order` property
- `.order-last`: Sets `order: 99999`

## Spacing

### Padding

- `.pa0` to `.pa7`: Sets padding on all sides
- `.pl0` to `.pl7`: Sets padding-left
- `.pr0` to `.pr7`: Sets padding-right
- `.pt0` to `.pt7`: Sets padding-top
- `.pb0` to `.pb7`: Sets padding-bottom
- `.ph0` to `.ph7`: Sets padding horizontally (left and right)
- `.pv0` to `.pv7`: Sets padding vertically (top and bottom)

### Margin

- `.ma0` to `.ma7`: Sets margin on all sides
- `.ml0` to `.ml7`: Sets margin-left
- `.mr0` to `.mr7`: Sets margin-right
- `.mt0` to `.mt7`: Sets margin-top
- `.mb0` to `.mb7`: Sets margin-bottom
- `.mh0` to `.mh7`: Sets margin horizontally (left and right)
- `.mv0` to `.mv7`: Sets margin vertically (top and bottom)

### Negative Margin

- `.na1` to `.na7`: Sets negative margin on all sides
- `.nl1` to `.nl7`: Sets negative margin-left
- `.nr1` to `.nr7`: Sets negative margin-right
- `.nt1` to `.nt7`: Sets negative margin-top
- `.nb1` to `.nb7`: Sets negative margin-bottom

## Typography

### Font Family

- `.sans-serif`: Sets font-family to a sans-serif stack
- `.serif`: Sets font-family to a serif stack
- `.system-sans-serif`: Sets font-family to the system's sans-serif font
- `.system-serif`: Sets font-family to the system's serif font
- `.code`: Sets font-family to a monospace stack
- `.courier`: Sets font-family to Courier
- `.helvetica`: Sets font-family to Helvetica
- `.avenir`: Sets font-family to Avenir
- `.athelas`: Sets font-family to Athelas
- `.georgia`: Sets font-family to Georgia
- `.times`: Sets font-family to Times
- `.bodoni`: Sets font-family to Bodoni
- `.calisto`: Sets font-family to Calisto
- `.garamond`: Sets font-family to Garamond
- `.baskerville`: Sets font-family to Baskerville

### Font Size

- `.f-headline`: Sets font-size to 6rem
- `.f-subheadline`: Sets font-size to 5rem
- `.f1` to `.f6`: Sets font-size from 3rem to .875rem

### Font Weight

- `.normal`: Sets `font-weight: normal`
- `.b`: Sets `font-weight: bold`
- `.fw1` to `.fw9`: Sets `font-weight` from 100 to 900

### Text Alignment

- `.tl`: Sets `text-align: left`
- `.tr`: Sets `text-align: right`
- `.tc`: Sets `text-align: center`
- `.tj`: Sets `text-align: justify`

### Text Decoration

- `.strike`: Sets `text-decoration: line-through`
- `.underline`: Sets `text-decoration: underline`
- `.no-underline`: Sets `text-decoration: none`

### Text Transform

- `.ttc`: Sets `text-transform: capitalize`
- `.ttl`: Sets `text-transform: lowercase`
- `.ttu`: Sets `text-transform: uppercase`
- `.ttn`: Sets `text-transform: none`

### Letter Spacing

- `.tracked`: Sets `letter-spacing: .1em`
- `.tracked-tight`: Sets `letter-spacing: -.05em`
- `.tracked-mega`: Sets `letter-spacing: .25em`

### Line Height

- `.lh-solid`: Sets `line-height: 1`
- `.lh-title`: Sets `line-height: 1.25`
- `.lh-copy`: Sets `line-height: 1.5`

## Colors

Tachyons provides a wide range of color classes. Here are some examples:

### Text Colors

- `.black`: Sets color to black
- `.near-black`: Sets color to near black
- `.dark-gray`: Sets color to dark gray
- `.mid-gray`: Sets color to mid gray
- `.gray`: Sets color to gray
- `.silver`: Sets color to silver
- `.light-silver`: Sets color to light silver
- `.moon-gray`: Sets color to moon gray
- `.light-gray`: Sets color to light gray
- `.near-white`: Sets color to near white
- `.white`: Sets color to white

(Similar classes exist for other colors like `.red`, `.orange`, `.yellow`, `.green`, `.blue`, `.indigo`, `.purple`, `.pink`)

### Background Colors

- `.bg-black`: Sets background-color to black
- `.bg-near-black`: Sets background-color to near black
- `.bg-dark-gray`: Sets background-color to dark gray
- `.bg-mid-gray`: Sets background-color to mid gray
- `.bg-gray`: Sets background-color to gray
- `.bg-silver`: Sets background-color to silver
- `.bg-light-silver`: Sets background-color to light silver
- `.bg-moon-gray`: Sets background-color to moon gray
- `.bg-light-gray`: Sets background-color to light gray
- `.bg-near-white`: Sets background-color to near white
- `.bg-white`: Sets background-color to white

(Similar classes exist for other colors like `.bg-red`, `.bg-orange`, `.bg-yellow`, `.bg-green`, `.bg-blue`, `.bg-indigo`, `.bg-purple`, `.bg-pink`)

## Borders

### Border Width

- `.ba`: Adds a border on all sides
- `.bt`: Adds a border to the top
- `.br`: Adds a border to the right
- `.bb`: Adds a border to the bottom
- `.bl`: Adds a border to the left
- `.bn`: Removes all borders

### Border Radius

- `.br0`: Sets `border-radius: 0`
- `.br1`: Sets `border-radius: .125rem`
- `.br2`: Sets `border-radius: .25rem`
- `.br3`: Sets `border-radius: .5rem`
- `.br4`: Sets `border-radius: 1rem`
- `.br-100`: Sets `border-radius: 100%`
- `.br-pill`: Sets `border-radius: 9999px`

### Border Style

- `.b--dotted`: Sets `border-style: dotted`
- `.b--dashed`: Sets `border-style: dashed`
- `.b--solid`: Sets `border-style: solid`
- `.b--none`: Sets `border-style: none`

### Border Color

- `.b--black`: Sets border-color to black
- `.b--near-black`: Sets border-color to near black
- `.b--dark-gray`: Sets border-color to dark gray
- `.b--mid-gray`: Sets border-color to mid gray
- `.b--gray`: Sets border-color to gray
- `.b--silver`: Sets border-color to silver
- `.b--light-silver`: Sets border-color to light silver
- `.b--moon-gray`: Sets border-color to moon gray
- `.b--light-gray`: Sets border-color to light gray
- `.b--near-white`: Sets border-color to near white
- `.b--white`: Sets border-color to white

(Similar classes exist for other colors like `.b--red`, `.b--orange`, `.b--yellow`, `.b--green`, `.b--blue`, `.b--indigo`, `.b--purple`, `.b--pink`)

## Layout

### Floats

- `.fl`: Sets `float: left`
- `.fr`: Sets `float: right`
- `.fn`: Sets `float: none`

### Clearfix

- `.cf`: Applies clearfix to contain floats

### Display

- `.dn`: Sets `display: none`
- `.di`: Sets `display: inline`
- `.db`: Sets `display: block`
- `.dib`: Sets `display: inline-block`
- `.dit`: Sets `display: inline-table`
- `.dt`: Sets `display: table`
- `.dtc`: Sets `display: table-cell`
- `.dt-row`: Sets `display: table-row`
- `.dt-row-group`: Sets `display: table-row-group`
- `.dt-column`: Sets `display: table-column`
- `.dt-column-group`: Sets `display: table-column-group`

### Visibility

- `.clip`: Hides content visually while keeping it accessible to screen readers

## Positioning

- `.static`: Sets `position: static`
- `.relative`: Sets `position: relative`
- `.absolute`: Sets `position: absolute`
- `.fixed`: Sets `position: fixed`

### Coordinates

- `.top-0`: Sets `top: 0`
- `.right-0`: Sets `right: 0`
- `.bottom-0`: Sets `bottom: 0`
- `.left-0`: Sets `left: 0`

(Similar classes exist for other values like `.top-1`, `.right-2`, etc.)

## Widths and Heights

### Widths

- `.w1` to `.w5`: Sets width from 1rem to 16rem
- `.w-10`: Sets width to 10%
- `.w-20`: Sets width to 20%
- `.w-25`: Sets width to 25%
- `.w-30`: Sets width to 30%
- `.w-33`: Sets width to 33%
- `.w-34`: Sets width to 34%
- `.w-40`: Sets width to 40%
- `.w-50`: Sets width to 50%
- `.w-60`: Sets width to 60%
- `.w-70`: Sets width to 70%
- `.w-75`: Sets width to 75%
- `.w-80`: Sets width to 80%
- `.w-90`: Sets width to 90%
- `.w-100`: Sets width to 100%
- `.w-third`: Sets width to calc(100% / 3)
- `.w-two-thirds`: Sets width to calc(100% / 1.5)
- `.w-auto`: Sets width to auto

### Heights

- `.h1` to `.h5`: Sets height from 1rem to 16rem
- `.h-25`: Sets height to 25%
- `.h-50`: Sets height to 50%
- `.h-75`: Sets height to 75%
- `.h-100`: Sets height to 100%
- `.min-h-100`: Sets min-height to 100%
- `.vh-25`: Sets height to 25vh
- `.vh-50`: Sets height to 50vh
- `.vh-75`: Sets height to 75vh
- `.vh-100`: Sets height to 100vh
- `.min-vh-100`: Sets min-height to 100vh
- `.h-auto`: Sets height to auto
- `.h-inherit`: Sets height to inherit

## Opacity

- `.o-100`: Sets opacity to 1
- `.o-90`: Sets opacity to .9
- `.o-80`: Sets opacity to .8
- ...
- `.o-10`: Sets opacity to .1
- `.o-05`: Sets opacity to .05
- `.o-025`: Sets opacity to .025

## Background

- `.cover`: Sets `background-size: cover`
- `.contain`: Sets `background-size: contain`
- `.bg-center`: Sets `background-position: center`
- `.bg-top`: Sets `background-position: top`
- `.bg-right`: Sets `background-position: right`
- `.bg-bottom`: Sets `background-position: bottom`
- `.bg-left`: Sets `background-position: left`

## Hover Effects

- `.dim`: Dims the element on hover
- `.glow`: Adds a subtle glow effect on hover
- `.hide-child`: Hides child elements until parent is hovered
- `.underline-hover`: Adds an underline on hover
- `.grow`: Scales the element up slightly on hover
- `.grow-large`: Scales the element up more significantly on hover
- `.pointer`: Changes the cursor to a pointer on hover

## Z-Index

- `.z-0`: Sets `z-index: 0`
- `.z-1`: Sets `z-index: 1`
- `.z-2`: Sets `z-index: 2`
- `.z-3`: Sets `z-index: 3`
- `.z-4`: Sets `z-index: 4`
- `.z-5`: Sets `z-index: 5`
- `.z-999`: Sets `z-index: 999`
- `.z-9999`: Sets `z-index: 9999`
- `.z-max`: Sets `z-index: 2147483647`
- `.z-inherit`: Sets `z-index: inherit`
- `.z-initial`: Sets `z-index: initial`
- `.z-unset`: Sets `z-index: unset`

## Responsive Design

Tachyons uses a mobile-first approach to responsive design. Many classes in Tachyons have responsive variants that allow you to apply different styles at different screen sizes. These variants are denoted by suffixes:

- No suffix: Applies to all screen sizes
- `-s`: Applies to small screens and up (30em and larger)
- `-m`: Applies to medium screens and up (48em and larger)
- `-l`: Applies to large screens and up (64em and larger)

For example:
- `.pa2` applies padding of 2 units at all screen sizes
- `.pa2-m` applies padding of 2 units only on medium and large screens
- `.pa2-l` applies padding of 2 units only on large screens

To use responsive classes effectively:
1. Start with the mobile layout (no suffix)
2. Add `-s`, `-m`, or `-l` classes to adjust the layout for larger screens as needed
