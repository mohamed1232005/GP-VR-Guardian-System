# ===========================================================================
# geometry/backproject.py — Phase 5: 2D Floor Pixels → 3D Camera Points
# Converts floor-masked pixels with metric depth into camera-space 3D points
# using the camera intrinsics.
# ===========================================================================

import numpy as np


def backproject_floor_points(
    depth_map: np.ndarray,
    floor_mask: np.ndarray,
    fx: float,
    fy: float,
    cx: float = None,
    cy: float = None,
    min_depth: float = 0.1,
    max_depth: float = 10.0,
    max_points: int = 5000,
    frame_id: str = "?",
) -> tuple:
    """
    Backproject floor pixels from 2D image to 3D camera coordinates.

    Uses the pinhole camera model:
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        Z = depth[v, u]

    Args:
        depth_map: HxW float32 depth in meters.
        floor_mask: HxW bool mask (True = floor pixel).
        fx, fy: Focal lengths (pixels).
        cx, cy: Principal point (defaults to image center).
        min_depth: Minimum valid depth (meters).
        max_depth: Maximum valid depth (meters).
        max_points: Maximum number of points to return (downsampled).
        frame_id: For logging.

    Returns:
        (points_3d, stats) where:
        - points_3d: Nx3 float32 numpy array of [X, Y, Z] in camera space.
        - stats: dict with point count, depth range info.
    """
    h, w = depth_map.shape[:2]

    # Default principal point = image center
    if cx is None:
        cx = w / 2.0
    if cy is None:
        cy = h / 2.0

    # Build valid mask: floor AND finite depth AND within range
    valid_mask = (
        floor_mask &
        np.isfinite(depth_map) &
        (depth_map > min_depth) &
        (depth_map < max_depth)
    )

    # Get valid pixel coordinates
    vs, us = np.where(valid_mask)  # row, col indices

    if len(vs) == 0:
        print(f"[P5_BACKPROJECT] frame={frame_id} points=0 "
              f"(no valid floor+depth pixels)")
        return np.empty((0, 3), dtype=np.float32), {
            "points": 0, "depth_min": 0, "depth_max": 0
        }

    # Downsample if too many points
    if len(vs) > max_points:
        indices = np.random.choice(len(vs), max_points, replace=False)
        vs = vs[indices]
        us = us[indices]

    # Get depth values
    z_vals = depth_map[vs, us].astype(np.float32)

    # Backproject to camera coordinates
    x_vals = (us.astype(np.float32) - cx) * z_vals / fx
    y_vals = (vs.astype(np.float32) - cy) * z_vals / fy

    points_3d = np.stack([x_vals, y_vals, z_vals], axis=1)  # Nx3

    depth_min = float(np.min(z_vals))
    depth_max = float(np.max(z_vals))

    stats = {
        "points": len(points_3d),
        "depth_min": depth_min,
        "depth_max": depth_max,
    }

    print(f"[P5_BACKPROJECT] frame={frame_id} points={len(points_3d)} "
          f"depth_min={depth_min:.2f} depth_max={depth_max:.2f}")

    return points_3d, stats