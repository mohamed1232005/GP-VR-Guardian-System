"""Geometry: ray-plane projection and guardian polygon generation."""

import numpy as np
from scipy.spatial import ConvexHull

from config import MIN_BOUNDARY_POINTS, MIN_AREA_M2


def project_fingertip_to_floor(
    smoothed_landmarks: np.ndarray,
    camera_to_world_flat: list,
    floor: dict,
) -> list | None:
    """
    Project the index fingertip (landmark 8) onto the floor plane.

    Parameters
    ----------
    smoothed_landmarks   : np.array (21, 3) — normalised [0,1] image coords
    camera_to_world_flat : list[float] — 16 floats, row-major 4×4 ARCore pose
    floor                : dict with intrinsics, floor_y_world,
                           plane_normal_world, plane_center_world

    Returns [x, y, z] world point, or None if ray is parallel / behind camera.
    """
    intr = floor["camera_intrinsics"]
    fx, fy = intr["fx"], intr["fy"]
    cx, cy = intr["cx"], intr["cy"]
    W,  H  = intr["width"], intr["height"]

    # 1. Index fingertip normalised -> pixel
    tip = smoothed_landmarks[8]
    u   = tip[0] * W
    v   = tip[1] * H

    # 2. Unproject to camera-space ray.
    #    CRITICAL: negate (v-cy) — image Y is down, camera Y is up.
    #    ARCore camera Z points behind the camera (negative-Z-forward).
    ray_cam = np.array([
         (u - cx) / fx,
        -(v - cy) / fy,
        -1.0,
    ])
    ray_cam /= np.linalg.norm(ray_cam)

    # 3. Camera-space ray -> world-space ray
    M          = np.array(camera_to_world_flat, dtype=float).reshape(4, 4)
    R          = M[:3, :3]
    cam_pos    = M[:3, 3]
    ray_world  = R @ ray_cam
    ray_world /= np.linalg.norm(ray_world)

    # 4. Ray-plane intersection
    normal   = np.array(floor["plane_normal_world"], dtype=float)
    plane_pt = np.array(floor["plane_center_world"], dtype=float)
    denom    = np.dot(normal, ray_world)

    if abs(denom) < 1e-6:
        return None  # ray parallel to floor

    t = np.dot(normal, plane_pt - cam_pos) / denom
    if t < 0:
        return None  # intersection is behind the camera

    world_pt    = cam_pos + t * ray_world
    world_pt[1] = floor["floor_y_world"]   # clamp Y (float safety)

    return world_pt.tolist()


def generate_guardian_polygon(
    points_3d: list,
) -> tuple[list, float, list]:
    """
    Build a convex hull from the boundary points.

    Returns
    -------
    ordered  : list of [x, y, z] — hull vertices in order
    area_m2  : float
    centroid : [x, y, z]
    """
    if len(points_3d) < MIN_BOUNDARY_POINTS:
        raise ValueError(f"Need at least {MIN_BOUNDARY_POINTS} points, got {len(points_3d)}")

    pts_xz = np.array([[p[0], p[2]] for p in points_3d])

    hull     = ConvexHull(pts_xz, qhull_options="QJ")
    ordered  = [points_3d[i] for i in hull.vertices]
    area_m2  = float(hull.volume)   # hull.volume == area in 2-D

    xz_verts = pts_xz[hull.vertices]
    centroid = [
        float(np.mean(xz_verts[:, 0])),
        float(points_3d[0][1]),
        float(np.mean(xz_verts[:, 1])),
    ]

    if area_m2 < MIN_AREA_M2:
        raise ValueError(f"Guardian area {area_m2:.2f} m² is below minimum {MIN_AREA_M2} m²")

    return ordered, area_m2, centroid