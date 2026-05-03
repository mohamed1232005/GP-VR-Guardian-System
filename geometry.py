import math
from typing import Iterable

import numpy as np

from config import MIN_AREA_M2


def _as_vec3(value, fallback=None):
    if value is None:
        return fallback
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return np.array([float(value[0]), float(value[1]), float(value[2])], dtype=np.float32)
    return fallback


def _pose_to_matrix(pose_matrix: list) -> np.ndarray:
    if pose_matrix is None or len(pose_matrix) != 16:
        return np.eye(4, dtype=np.float32)
    return np.array(pose_matrix, dtype=np.float32).reshape((4, 4))


def project_fingertip_to_floor(smoothed_landmarks: np.ndarray, pose_matrix: list, floor: dict):
    """
    Projects index fingertip landmark 8 to the confirmed floor plane.

    This remains for legacy boundary point placement.
    Ball ray/skeleton do not depend on this; they use landmarks directly in Unity.
    """
    if smoothed_landmarks is None or len(smoothed_landmarks) <= 8:
        return None

    pose = _pose_to_matrix(pose_matrix)

    floor_point = _as_vec3(
        floor.get("floor_point_world")
        or floor.get("center_world")
        or floor.get("world_position"),
        None,
    )

    if floor_point is None:
        y = float(floor.get("floor_y_world", floor.get("y", 0.0)))
        floor_point = np.array([0.0, y, 0.0], dtype=np.float32)

    floor_normal = _as_vec3(
        floor.get("floor_normal_world")
        or floor.get("normal_world")
        or floor.get("normal"),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
    )

    norm = float(np.linalg.norm(floor_normal))
    if norm < 1e-6:
        floor_normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    else:
        floor_normal = floor_normal / norm

    # Camera origin and approximate forward from Unity localToWorld matrix.
    cam_origin = pose[:3, 3]
    cam_forward = pose[:3, 2]
    if np.linalg.norm(cam_forward) < 1e-6:
        cam_forward = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    # Use fingertip normalized viewport offset to build a stable ray approximation.
    tip = smoothed_landmarks[8]
    x = float(tip[0]) - 0.5
    y = 0.5 - float(tip[1])

    right = pose[:3, 0]
    up = pose[:3, 1]
    direction = cam_forward + right * x * 1.2 + up * y * 1.2

    d_norm = float(np.linalg.norm(direction))
    if d_norm < 1e-6:
        return None
    direction = direction / d_norm

    denom = float(np.dot(direction, floor_normal))
    if abs(denom) < 1e-5:
        return None

    t = float(np.dot(floor_point - cam_origin, floor_normal) / denom)
    if t <= 0.02 or t > 12.0:
        return None

    p = cam_origin + direction * t
    return [float(p[0]), float(p[1]), float(p[2])]


def generate_guardian_polygon(points: Iterable[list]):
    pts = [list(map(float, p[:3])) for p in points if p is not None and len(p) >= 3]
    if len(pts) < 4:
        raise ValueError("Need at least 4 boundary points.")

    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    cz = sum(p[2] for p in pts) / len(pts)

    ordered = sorted(pts, key=lambda p: math.atan2(p[2] - cz, p[0] - cx))

    area = 0.0
    for i in range(len(ordered)):
        a = ordered[i]
        b = ordered[(i + 1) % len(ordered)]
        area += a[0] * b[2] - b[0] * a[2]
    area = abs(area) * 0.5

    if area < MIN_AREA_M2:
        raise ValueError(f"Boundary area too small: {area:.2f} m²")

    return ordered, area, [cx, cy, cz]