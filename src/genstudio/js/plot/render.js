
export function childEvents({onDragStart, onDrag, onDragEnd, onClick}) {
    function render (index, scales, values, dimensions, context, next) {
        // Call the next render function to get the base SVG group
        const g = next(index, scales, values, dimensions, context);
        let activeElement = null;
        let totalDx = 0;
        let totalDy = 0;
        let initialUnscaledX, initialUnscaledY;
        let initialIndex;
        let initialTransform = '';
        let isDragging = false;
        const dragThreshold = 2; // pixels
        let clickStartTime;

        // Create empty local objects for scaled and unscaled values.
        // These are used to track the current positions of children that
        // have been dragged, without mutating the original values.
        const localScaledValues = {
            x: {},
            y: {}
        };
        const localUnscaledValues = {
            x: {},
            y: {}
        };

        // Helper function to create a payload for callbacks
        // This includes both scaled (pixel) and unscaled (data) coordinates
        const createPayload = (index, unscaledX, unscaledY, type) => ({
            index,
            x: unscaledX,
            y: unscaledY,
            pixels: {
                // Use the scales provided by Observable Plot to convert data coordinates to pixel coordinates
                x: scales.x(unscaledX),
                y: scales.y(unscaledY)
            },
            type
        });

        // Helper function to parse existing SVG transforms
        // This allows us to combine our offsets with any existing transforms
        const parseTransform = (transform) => {
            const match = transform.match(/translate\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)/);
            return match ? { x: parseFloat(match[1]), y: parseFloat(match[2]) } : { x: 0, y: 0 };
        };

        const findDirectChild = (element) => {
            while (element && element.parentNode !== g) {
                element = element.parentNode;
            }
            return element;
        };

        const handleDragStart = (event) => {
            // Find the first element for which g is the direct parent
            activeElement = findDirectChild(event.target);
            if (!activeElement) return;

            event.preventDefault();
            initialIndex = Array.from(g.children).indexOf(activeElement);
            // Use local values if available, otherwise fall back to original values
            initialUnscaledX = localUnscaledValues.x[initialIndex] ?? values.channels.x.value[initialIndex];
            initialUnscaledY = localUnscaledValues.y[initialIndex] ?? values.channels.y.value[initialIndex];
            initialTransform = activeElement.getAttribute('transform') || '';

            clickStartTime = new Date().getTime();

            document.addEventListener('mousemove', handleDrag);
            document.addEventListener('mouseup', handleDragEnd);
        };

        const handleDrag = (event) => {
            if (!activeElement) return;

            totalDx += event.movementX;
            totalDy += event.movementY;

            if (!isDragging && (Math.abs(totalDx) > dragThreshold || Math.abs(totalDy) > dragThreshold)) {
                isDragging = true;
                if (onDragStart) onDragStart(createPayload(initialIndex, initialUnscaledX, initialUnscaledY, "dragstart"));
            }

            if (isDragging) {
                // Use the scales' invert function to convert pixel offsets back to data coordinates
                const currentUnscaledX = scales.x.invert(scales.x(initialUnscaledX) + totalDx);
                const currentUnscaledY = scales.y.invert(scales.y(initialUnscaledY) + totalDy);

                const initialTranslate = parseTransform(initialTransform);
                const newTranslateX = initialTranslate.x + totalDx;
                const newTranslateY = initialTranslate.y + totalDy;

                if (onDrag) onDrag(createPayload(initialIndex, currentUnscaledX, currentUnscaledY, "drag"))

                // Update the SVG transform to move the element
                activeElement.setAttribute('transform', `translate(${newTranslateX}, ${newTranslateY})`);
            }
        };

        const handleDragEnd = (event) => {
            if (!activeElement) return;

            const clickEndTime = new Date().getTime();
            const clickDuration = clickEndTime - clickStartTime;

            if (isDragging) {
                // Calculate final positions in both unscaled (data) and scaled (pixel) coordinates
                const finalUnscaledX = scales.x.invert(scales.x(initialUnscaledX) + totalDx);
                const finalUnscaledY = scales.y.invert(scales.y(initialUnscaledY) + totalDy);

                // Update local values to reflect the new position
                localUnscaledValues.x[initialIndex] = finalUnscaledX;
                localUnscaledValues.y[initialIndex] = finalUnscaledY;
                localScaledValues.x[initialIndex] = finalUnscaledX;
                localScaledValues.y[initialIndex] = finalUnscaledY;

                const initialTranslate = parseTransform(initialTransform);
                const finalTranslateX = initialTranslate.x + totalDx;
                const finalTranslateY = initialTranslate.y + totalDy;

                if (onDragEnd) onDragEnd(createPayload(initialIndex, finalUnscaledX, finalUnscaledY, "dragend"));

                // Set the final transform on the SVG element
                activeElement.setAttribute('transform', `translate(${finalTranslateX}, ${finalTranslateY})`);
            } else if (clickDuration < 200 && Math.abs(totalDx) < dragThreshold && Math.abs(totalDy) < dragThreshold) {
                // If it's a short click and hasn't moved much, treat it as a click
                if (onClick) onClick(createPayload(initialIndex, initialUnscaledX, initialUnscaledY, "click"));
            }

            // Clean up event listeners
            document.removeEventListener('mousemove', handleDrag);
            document.removeEventListener('mouseup', handleDragEnd);

            // Reset state
            activeElement = null;
            totalDx = 0;
            totalDy = 0;
            isDragging = false;
        };

        // Add mousedown event listener to the SVG group
        g.addEventListener('mousedown', handleDragStart);

        return g;
    }

    return render;
}
