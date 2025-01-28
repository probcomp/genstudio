import numpy as np
import math

import genstudio.plot as Plot
from genstudio.plot import js
from genstudio.scene3d import PointCloud, Ellipsoid, deco


# ----------------- 1) Ripple Grid (Point Cloud) -----------------
def create_ripple_grid(n_x=50, n_y=50, n_frames=60):
    """Create frames of a 2D grid of points in the XY plane with sinusoidal ripple over time.

    Returns:
        xyz_frames: shape (n_frames, n_points*3), the coordinates for each frame, flattened
        rgb: shape (n_points*3), constant color data
    """
    # 1. Create the base grid in [x, y]
    x_vals = np.linspace(-1.0, 1.0, n_x)
    y_vals = np.linspace(-1.0, 1.0, n_y)
    xx, yy = np.meshgrid(x_vals, y_vals)

    # Flatten to a list of (x,y) pairs
    n_points = n_x * n_y
    base_xy = np.column_stack([xx.flatten(), yy.flatten()])

    # 2. We'll pre-allocate an array to hold the animated frames.
    xyz_frames = np.zeros((n_frames, n_points * 3), dtype=np.float32)

    # 3. Create ripple in Z dimension. We'll do something like:
    #    z = amplitude * sin( wavefreq*(x + y) + time*speed )
    #    You can adjust amplitude, wavefreq, speed as desired.
    amplitude = 0.2
    wavefreq = 4.0
    speed = 2.0

    # 4. Generate each frame
    for frame_i in range(n_frames):
        t = frame_i / float(n_frames) * 2.0 * math.pi * speed
        # Calculate the z for each point
        z_vals = amplitude * np.sin(wavefreq * (base_xy[:, 0] + base_xy[:, 1]) + t)
        # Combine x, y, z
        frame_xyz = np.column_stack([base_xy[:, 0], base_xy[:, 1], z_vals])
        # Flatten to shape (n_points*3,)
        xyz_frames[frame_i] = frame_xyz.flatten()

    # 5. Assign colors. Here we do a simple grayscale or you can do something fancy.
    #    We'll map (x,y) to a color range for fun.
    #    Convert positions to 0..1 range for the color mapping
    #    We'll just do a simple gradient color based on x,y for demonstration.
    x_norm = (base_xy[:, 0] + 1) / 2
    y_norm = (base_xy[:, 1] + 1) / 2
    # Let's make a color scheme that transitions from green->blue with x,y
    # R = x, G = y, B = 1 - x
    # or do whatever you like
    r = x_norm
    g = y_norm
    b = 1.0 - x_norm
    rgb = np.column_stack([r, g, b]).astype(np.float32).flatten()

    return xyz_frames, rgb


# ----------------- 2) Morphing Ellipsoids -----------------
def create_morphing_ellipsoids(
    n_ellipsoids=90, n_frames=180
):  # More frames for smoother motion
    """
    Generate per-frame positions/radii for a vehicle-like convoy navigating a virtual city.
    The convoy follows a distinct path as if following city streets.

    Returns:
        centers_frames: shape (n_frames, n_ellipsoids, 3)
        radii_frames:   shape (n_frames, n_ellipsoids, 3)
        colors:         shape (n_ellipsoids, 3)
    """
    n_snakes = 1
    n_per_snake = n_ellipsoids // n_snakes

    # Create colors that feel more vehicle-like
    colors = np.zeros((n_ellipsoids, 3), dtype=np.float32)
    t = np.linspace(0, 1, n_per_snake)
    # Second convoy: red to orange glow
    colors[:n_per_snake] = np.column_stack(
        [0.9 - 0.1 * t, 0.3 + 0.3 * t, 0.2 + 0.1 * t]
    )

    centers_frames = np.zeros((n_frames, n_ellipsoids, 3), dtype=np.float32)
    radii_frames = np.zeros((n_frames, n_ellipsoids, 3), dtype=np.float32)

    # City grid parameters
    block_size = 0.8

    for frame_i in range(n_frames):
        t = 2.0 * math.pi * frame_i / float(n_frames)

        for i in range(n_per_snake):
            idx = i
            s = i / n_per_snake

            # Second convoy: follows rectangular blocks
            block_t = (t * 0.2 + s * 0.5) % (2.0 * math.pi)
            if block_t < math.pi / 2:  # First segment
                x = block_size * (block_t / (math.pi / 2))
                y = block_size
            elif block_t < math.pi:  # Second segment
                x = block_size
                y = block_size * (2 - block_t / (math.pi / 2))
            elif block_t < 3 * math.pi / 2:  # Third segment
                x = block_size * (3 - block_t / (math.pi / 2))
                y = -block_size
            else:  # Fourth segment
                x = -block_size
                y = block_size * (block_t / (math.pi / 2) - 4)
            z = 0.15 + 0.05 * math.sin(block_t * 4)

            centers_frames[frame_i, idx] = [x, y, z]

            # Base size smaller for vehicle feel
            base_size = 0.08 * (0.9 + 0.1 * math.sin(s * 2.0 * math.pi))

            # Elongate in direction of motion
            radii_frames[frame_i, idx] = [
                base_size * 1.5,  # longer
                base_size * 0.8,  # narrower
                base_size * 0.6,  # lower profile
            ]

    return centers_frames, radii_frames, colors


# ----------------- Putting it all together in a Plot -----------------
def create_ripple_and_morph_scene():
    """
    Create a scene with:
      1) A ripple grid of points
      2) Three vehicle-like convoys navigating a virtual city

    Returns a Plot layout.
    """
    # 1. Generate data for the ripple grid
    n_frames = 120  # More frames for slower motion
    grid_xyz_frames, grid_rgb = create_ripple_grid(n_x=60, n_y=60, n_frames=n_frames)

    # 2. Generate data for morphing ellipsoids
    ellipsoid_centers, ellipsoid_radii, ellipsoid_colors = create_morphing_ellipsoids(
        n_ellipsoids=90, n_frames=n_frames
    )

    # Debug prints to verify data
    print("Ellipsoid centers shape:", ellipsoid_centers.shape)
    print("Sample center frame 0:", ellipsoid_centers[0])
    print("Ellipsoid radii shape:", ellipsoid_radii.shape)
    print("Sample radii frame 0:", ellipsoid_radii[0])
    print("Ellipsoid colors shape:", ellipsoid_colors.shape)
    print("Colors:", ellipsoid_colors)

    # We'll set up a default camera that can see everything nicely
    camera = {
        "position": [2.0, 2.0, 1.5],  # Adjusted for better view
        "target": [0, 0, 0],
        "up": [0, 0, 1],
        "fov": 45,
        "near": 0.01,
        "far": 100.0,
    }

    # 3. Create the Scenes
    # Note: we can define separate scenes or combine them into a single scene
    #       by simply adding the geometry. Here, let's show them side-by-side.

    # First scene: the ripple grid
    scene_grid = PointCloud(
        positions=js("$state.grid_xyz[$state.frame]"),
        colors=js("$state.grid_rgb"),
        scales=np.ones(60 * 60, dtype=np.float32) * 0.03,  # each point scale
        onHover=js("(i) => $state.update({hover_point: i})"),
        decorations=[
            deco(
                js("$state.hover_point ? [$state.hover_point] : []"),
                color=[1, 1, 0],
                scale=1.5,
            ),
        ],
    ) + {
        "onCameraChange": js("(cam) => $state.update({camera: cam})"),
        "camera": js("$state.camera"),
    }

    # Second scene: the morphing ellipsoids with opacity decorations
    scene_ellipsoids = Ellipsoid(
        centers=js("$state.ellipsoid_centers[$state.frame]"),
        radii=js("$state.ellipsoid_radii[$state.frame]"),
        colors=js("$state.ellipsoid_colors"),
        decorations=[
            # Vary opacity based on position in snake
            deco(list(range(30)), alpha=0.7),  # First snake more transparent
            deco(list(range(30, 60)), alpha=0.9),  # Second snake more solid
            deco(list(range(60, 90)), alpha=0.9),  # Third snake more solid
            # Add some highlights
            deco(
                [0, 30], color=[1, 1, 0], alpha=0.8, scale=1.2
            ),  # Highlight lead ellipsoids
            deco(
                [30, 60], color=[1, 1, 0], alpha=0.8, scale=1.2
            ),  # Highlight lead ellipsoids
            deco(
                [60, 90], color=[1, 1, 0], alpha=0.8, scale=1.2
            ),  # Highlight lead ellipsoids
        ],
    ) + {
        "onCameraChange": js("(cam) => $state.update({camera: cam})"),
        "camera": js("$state.camera"),
    }

    layout = (
        Plot.initialState(
            {
                "camera": camera,
                "frame": 0,  # current frame in the animation
                "grid_xyz": grid_xyz_frames,
                "grid_rgb": grid_rgb,
                "ellipsoid_centers": ellipsoid_centers.reshape(
                    n_frames, -1
                ),  # Flatten to (n_frames, n_ellipsoids*3)
                "ellipsoid_radii": ellipsoid_radii.reshape(
                    n_frames, -1
                ),  # Flatten to (n_frames, n_ellipsoids*3)
                "ellipsoid_colors": ellipsoid_colors.flatten(),  # Flatten to (n_ellipsoids*3,)
                "hover_point": None,
            }
        )
        | Plot.Slider("frame", range=n_frames, fps="raf")
        | (scene_grid & scene_ellipsoids)
    )

    return layout


# Call create_ripple_and_morph_scene to get the final layout
create_ripple_and_morph_scene()
