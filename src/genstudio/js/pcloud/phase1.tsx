import React, { useState } from 'react';
import { Points3D } from '../../../components/Points3D';

function Phase1() {
  const [hoveredPoints, setHoveredPoints] = useState<number[]>([]);
  const [selectedPoints, setSelectedPoints] = useState<number[]>([]);
  const [showRegions, setShowRegions] = useState(false);

  const decorations = [
    // Hover effect - slightly larger, semi-transparent yellow
    {
      indexes: hoveredPoints,
      color: [1, 1, 0],
      scale: 1.2,
      alpha: 0.8,
      blendMode: 'screen',
      blendStrength: 0.7
    },
    // Selection - solid red, larger
    {
      indexes: selectedPoints,
      color: [1, 0, 0],
      scale: 1.5,
      blendMode: 'replace'
    },
    // Region visualization
    ...(showRegions ? [
      {
        indexes: regionAPoints,
        color: [0, 1, 0],
        blendMode: 'multiply',
        blendStrength: 0.8
      },
      {
        indexes: regionBPoints,
        color: [0, 0, 1],
        blendMode: 'multiply',
        blendStrength: 0.8
      }
    ] : [])
  ];

  return (
    <Points3D
      points={points}
      decorations={decorations}
      onPointHover={(idx) => setHoveredPoints(idx ? [idx] : [])}
      // ... other props
    />
  );
}

export default Phase1;
