import React, {useRef, useCallback} from "react"

interface FPSCounterProps {
    fpsRef: React.RefObject<HTMLDivElement>;
}

export function FPSCounter({ fpsRef }: FPSCounterProps) {
    return (
        <div
            ref={fpsRef}
            style={{
                position: 'absolute',
                top: '10px',
                left: '10px',
                color: 'white',
                backgroundColor: 'rgba(0, 0, 0, 0.5)',
                padding: '5px',
                borderRadius: '3px',
                fontSize: '14px'
            }}
        >
            0 FPS
        </div>
    );
}

export function useFPSCounter() {
    const fpsDisplayRef = useRef<HTMLDivElement>(null);
    const renderTimesRef = useRef<number[]>([]);
    const MAX_SAMPLES = 10;

    const updateDisplay = useCallback((renderTime: number) => {
        renderTimesRef.current.push(renderTime);
        if (renderTimesRef.current.length > MAX_SAMPLES) {
            renderTimesRef.current.shift();
        }

        const avgRenderTime = renderTimesRef.current.reduce((a, b) => a + b, 0) /
            renderTimesRef.current.length;

        const avgFps = 1000 / avgRenderTime;

        if (fpsDisplayRef.current) {
            fpsDisplayRef.current.textContent =
                `${avgFps.toFixed(1)} FPS`;
        }
    }, []);

    return { fpsDisplayRef, updateDisplay };
}
