import * as React from "react";
import { AUTOGRID_MIN as AUTOGRID_MIN_WIDTH } from "./context";
import { tw, useContainerWidth, joinClasses } from "./utils";

/**
 * Converts width values into CSS grid-compatible values.
 * @param {number|string} width - The width value to convert
 * @returns {string} CSS grid-compatible width value
 */
function getGridValue(width) {
  if (typeof width === "number") {
    return `${width}fr`;
  }
  if (typeof width === "string") {
    if (width.includes("/")) {
      const [num, denom] = width.split("/");
      return `${(Number(num) / Number(denom)) * 100}%`;
    }
    return width;
  }
  return width;
}

/**
 * A responsive grid layout component that automatically arranges children in a grid.
 * @param {Object} props
 * @param {React.ReactNode} props.children - Child elements to arrange in the grid
 * @param {Object} [props.style] - Additional CSS styles
 * @param {number} [props.minWidth=165] - Minimum width for auto-calculated columns
 * @param {number} [props.gap=1] - Grid gap size (applies to both row and column gaps)
 * @param {number} [props.rowGap] - Vertical gap between rows
 * @param {number} [props.colGap] - Horizontal gap between columns
 * @param {number} [props.cols] - Fixed number of columns
 * @param {number} [props.minCols=1] - Minimum number of columns
 * @param {number} [props.maxCols] - Maximum number of columns
 * @param {Array<number|string>} [props.widths] - Array of column widths
 * @param {Array<number|string>} [props.heights] - Array of row heights
 * @param {string} [props.height] - Container height
 */
export function Grid({
  children,
  style,
  minWidth = AUTOGRID_MIN_WIDTH,
  gap = 1,
  rowGap,
  colGap,
  cols,
  minCols = 1,
  maxCols,
  widths,
  heights,
  height,
  className,
}) {
  const [containerRef, containerWidth] = useContainerWidth();

  // Handle gap values
  const gapX = colGap ?? gap;
  const gapY = rowGap ?? gap;
  const gapClass = `gap-x-${gapX} gap-y-${gapY}`;

  // Calculate number of columns if not explicitly set
  let numColumns;
  if (cols) {
    numColumns = cols;
  } else {
    const effectiveMinWidth = Math.min(minWidth, containerWidth);
    const autoColumns = Math.floor(containerWidth / effectiveMinWidth);
    numColumns = Math.max(
      minCols,
      maxCols ? Math.min(autoColumns, maxCols) : autoColumns,
      1
    );
    numColumns = Math.min(numColumns, React.Children.count(children));
  }

  const gridCols = widths
    ? `grid-cols-[${widths.map(getGridValue).join("_")}]`
    : `grid-cols-${numColumns}`;

  const gridRows = heights
    ? `grid-rows-[${heights.map(getGridValue).join("_")}]`
    : "grid-rows-[auto]";

  const containerStyle = {
    width: "100%",
    ...style,
  };

  const classes = joinClasses(
    "grid",
    gridCols,
    gridRows,
    gapClass,
    height && `h-[${height}]`
  );

  return (
    <div
      ref={containerRef}
      className={tw(joinClasses(classes, className))}
      style={containerStyle}
    >
      {children}
    </div>
  );
}

// wrap primitive children in a span,
// merge objects into parentProps
function flattenChildren(parentProps, children) {
  const processedChildren = [];
  const processedProps = { ...parentProps };

  children.forEach((child) => {
    if (child == null) return;
    if (typeof child === "string" || typeof child === "number") {
      processedChildren.push(child);
    } else if (child.constructor === Object) {
      // handle objects
      if (React.isValidElement(child)) {
        processedChildren.push(child);
      } else {
        Object.assign(processedProps, child);
      }
    }
  });

  return [processedChildren, processedProps];
}

/**
 * A component that arranges children in a horizontal row using CSS Grid.
 * @param {Object} props
 * @param {React.ReactNode} props.children - Child elements to arrange in the row
 * @param {number} [props.gap=1] - Gap between row items
 * @param {Array<number|string>} [props.widths] - Array of column widths
 * @param {string} [props.height] - Container height
 * @param {string} [props.width] - Container width
 * @param {string} [props.className] - Additional CSS classes
 */
export function Row({ children, ...props }) {

  [children, props] = flattenChildren(props, children);

  const { gap = 1, widths, height, width, className } = props;

  const gridCols = widths
    ? `grid-cols-[${widths.map(getGridValue).join("_")}]`
    : `grid-cols-[${Array(children.length).fill("auto").join("_")}]`;

  const classes = joinClasses(
    "grid",
    gap && `gap-${gap}`,
    height && `h-[${height}]`,
    width && `w-[${width}]`,
    gridCols,
    className
  );

  return (
    <div {...props} className={tw(classes)}>
      {children}
    </div>
  );
}

/**
 * A component that arranges children in a vertical column using CSS Grid.
 * @param {Object} props
 * @param {React.ReactNode} props.children - Child elements to arrange in the column
 * @param {number} [props.gap=1] - Gap between column items
 * @param {Array<number|string>} [props.heights] - Array of row heights
 * @param {string} [props.height] - Container height
 * @param {string} [props.width] - Container width
 * @param {string} [props.className] - Additional CSS classes
 */
export function Column({ children, ...props }) {
  [children, props] = flattenChildren(props, children);
  const { gap = 1, heights, height, width, className } = props;

  const gridRows = heights
    ? `grid-rows-[${heights.map(getGridValue).join("_")}]`
    : `grid-rows-[${Array(React.Children.count(children))
        .fill("auto")
        .join("_")}]`;

  const classes = joinClasses(
    "grid",
    gap && `gap-${gap}`,
    height ? `h-[${height}]` : "h-fit",
    width && `w-[${width}]`,
    gridRows,
    className
  );

  return (
    <div {...props} className={tw(classes)}>
      {children}
    </div>
  );
}
