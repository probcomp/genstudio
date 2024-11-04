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
    maxCols
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
        display: 'grid',
        gridTemplateColumns: `repeat(${numColumns}, 1fr)`,
        width: '100%',
        ...style
    };

    return (
        <div ref={containerRef} className={tw(gapClass)} style={containerStyle}>
            {children.map((value, index) => (
                renderNode(value, { key: index, style: { width: itemWidth } })
            ))}
        </div>
    );
}

export function Row({ children, gap=1, widths, ...props }) {
    const renderNode = useContext(NodeContext);
    const className = `flex flex-row gap-${gap} ${props.className || props.class || ''}`
    delete props["className"]

    let flexClasses = []
    if (widths) {
        flexClasses = widths.map(w => {
            if (typeof w === 'string') {
                return w.includes('/') ? `w-${w}` : `w-[${w}]`
            }
            return `flex-[${w}]`
        })
    } else {
        flexClasses = Array(React.Children.count(children)).fill("flex-1")
    }

    return (
        <div {...props} className={tw(className)}>
            {React.Children.map(children, (child, index) => (
                <div className={tw(flexClasses[index])} key={index}>
                    {renderNode(child)}
                </div>
            ))}
        </div>
    );
}

export function Column({ children, gap=1, ...props }) {
    const renderNode = useContext(NodeContext);
    return (
        <div {...props} className={tw(`flex flex-col gap-${gap}`)}>
            {React.Children.map(children, (child, index) => (
                <div key={index}>
                    {renderNode(child)}
                </div>
            ))}
        </div>
    );
}
