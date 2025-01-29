import React, {useRef, useCallback, useEffect} from "react"

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
    const frameTimesRef = useRef<number[]>([]);
    const lastRenderTimeRef = useRef<number>(0);
    const lastUpdateTimeRef = useRef<number>(0);
    const MAX_SAMPLES = 60;

    const updateDisplay = useCallback((renderTime: number) => {
        const now = performance.now();

        // If this is a new frame (not just a re-render)
        if (now - lastUpdateTimeRef.current > 1) {
            // Calculate total frame time (render time + time since last frame)
            const totalTime = renderTime + (now - lastRenderTimeRef.current);
            frameTimesRef.current.push(totalTime);

            if (frameTimesRef.current.length > MAX_SAMPLES) {
                frameTimesRef.current.shift();
            }

            // Calculate average FPS from frame times
            const avgFrameTime = frameTimesRef.current.reduce((a, b) => a + b, 0) /
                frameTimesRef.current.length;
            const fps = 1000 / avgFrameTime;

            if (fpsDisplayRef.current) {
                fpsDisplayRef.current.textContent = `${Math.round(fps)} FPS`;
            }

            lastUpdateTimeRef.current = now;
            lastRenderTimeRef.current = now;
        }
    }, []);

    return { fpsDisplayRef, updateDisplay };
}
