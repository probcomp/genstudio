import * as React from "react";
import { useContext } from "react";
import { NodeContext, AUTOGRID_MIN as AUTOGRID_MIN_WIDTH } from "./context";
import { tw, useContainerWidth } from "./utils";

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
}) {
  const [containerRef, containerWidth] = useContainerWidth();
  const renderNode = useContext(NodeContext);

  // Handle gap values
  const gapX = colGap ?? gap;
  const gapY = rowGap ?? gap;
  const gapClass = `gap-x-${gapX} gap-y-${gapY}`;
  const gapSize = parseInt(gap); // Keep for width calculations

  // Calculate number of columns
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
    numColumns = Math.min(numColumns, children.length);
  }

  const itemWidth = (containerWidth - (numColumns - 1) * gapSize) / numColumns;

  const containerStyle = {
    display: "grid",
    gridTemplateColumns: `repeat(${numColumns}, 1fr)`,
    width: "100%",
    ...style,
  };

  return (
    <div ref={containerRef} className={tw(gapClass)} style={containerStyle}>
      {children.map((value, index) =>
        renderNode(value, { key: index, style: { width: itemWidth } })
      )}
    </div>
  );
}

function getFlexClasses(prefix, sizes, count) {
  if (!sizes) {
    return Array(count).fill("flex-1");
  }

  return sizes.map((size) => {
    if (typeof size === "string") {
      return size.includes("/") ? `${prefix}-${size}` : `${prefix}-[${size}]`;
    }
    return `flex-[${size}]`;
  });
}
function joinClasses(...classes) {
  let result = classes[0] || "";
  for (let i = 1; i < classes.length; i++) {
    if (classes[i]) result += " " + classes[i];
  }
  return result;
}

export function Row({
  children,
  gap = 1,
  widths,
  height,
  width,
  className,
  ...props
}) {
  const renderNode = useContext(NodeContext);
  const classes = joinClasses(
    "flex flex-row",
    gap && `gap-${gap}`,
    height && `h-[${height}]`,
    width && `w-[${width}]`,
    className
  );

  const flexClasses = getFlexClasses(
    "w",
    widths,
    React.Children.count(children)
  );

  return (
    <div {...props} className={tw(classes)}>
      {React.Children.map(children, (child, index) => (
        <div className={tw(flexClasses[index])} key={index}>
          {renderNode(child)}
        </div>
      ))}
    </div>
  );
}

export function Column({
  children,
  gap = 1,
  heights,
  height,
  width,
  className,
  ...props
}) {
  const renderNode = useContext(NodeContext);
  const classes = joinClasses(
    "flex flex-col",
    gap && `gap-${gap}`,
    height ? `h-[${height}]` : "h-full",
    width && `w-[${width}]`,
    className
  );

  const flexClasses = getFlexClasses(
    "h",
    heights,
    React.Children.count(children)
  );

  return (
    <div {...props} className={tw(classes)}>
      {React.Children.map(children, (child, index) => (
        <div key={index} className={tw(flexClasses[index])}>
          {renderNode(child)}
        </div>
      ))}
    </div>
  );
}
