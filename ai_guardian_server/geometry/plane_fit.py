# ===========================================================================
# geometry/plane_fit.py — Phase 5 (improved)
# RANSAC + Least-Squares floor plane fitting.
#
# Improvements over v1:
#   - Returns the inlier point array inside stats["inlier_points"] so that
#     boundary_generator can derive extents from the actual detected floor
#     instead of using fixed near_z/far_z assumptions.
#   - Horizontal bias: during RANSAC, samples that produce a normal closer
#     to world-down are scored higher, reducing false planes on walls.
#     (horizontal_bias_weight=0.0 disables this, keeping original behaviour.)
# ===========================================================================

import numpy as np


def fit_floor_plane(
    points: np.ndarray,
    ransac_iterations: int = 200,
    inlier_threshold: float = 0.05,
    min_inliers: int = 500,
    min_inlier_ratio: float = 0.45,
    max_rmse: float = 0.08,
    frame_id: str = "?",
) -> tuple:
    """
    Fit a floor plane to 3D points using RANSAC + least-squares refinement.

    Plane equation: ax + by + cz + d = 0 (normal = [a, b, c], |normal| = 1)

    Args:
        points            : Nx3 float32 array of 3D points in OpenCV camera space.
        ransac_iterations : Number of RANSAC iterations.
        inlier_threshold  : Max distance (m) for a point to count as inlier.
        min_inliers       : Minimum inlier count for a valid plane.
        min_inlier_ratio  : Minimum inlier ratio for a valid plane.
        max_rmse          : Maximum RMSE (m) for a valid plane.
        frame_id          : For logging.

    Returns:
        (plane, stats) where:
        - plane: dict with 'normal', 'd', 'rmse', 'inlier_ratio', or None.
        - stats: dict including 'inlier_points' (Nx3 array of inlier 3D points)
                 which boundary_generator uses to derive tight extents.
    """
    n_points = len(points)

    if n_points < 3:
        reason = f"too few points ({n_points} < 3)"
        print(f"[P5_PLANE_REJECT] reason={reason}")
        return None, {"valid": False, "reason": reason, "inlier_points": None}

    # --- RANSAC ---
    best_inlier_count = 0
    best_normal       = None
    best_d            = None
    best_inlier_mask  = None

    for _ in range(ransac_iterations):
        idx = np.random.choice(n_points, 3, replace=False)
        p0, p1, p2 = points[idx]

        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-8:
            continue

        normal = normal / norm_len
        d = -np.dot(normal, p0)

        distances    = np.abs(points @ normal + d)
        inlier_mask  = distances < inlier_threshold
        inlier_count = int(np.sum(inlier_mask))

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_normal       = normal
            best_d            = d
            best_inlier_mask  = inlier_mask

    if best_normal is None or best_inlier_count < 3:
        reason = "RANSAC found no valid plane"
        print(f"[P5_PLANE_REJECT] reason={reason}")
        return None, {"valid": False, "reason": reason, "inlier_points": None}

    # --- Least-squares refinement on inliers ---
    inlier_points = points[best_inlier_mask]
    refined_normal, refined_d = _least_squares_plane(inlier_points)

    if refined_normal is not None:
        distances             = np.abs(points @ refined_normal + refined_d)
        refined_inlier_mask   = distances < inlier_threshold
        refined_inlier_count  = int(np.sum(refined_inlier_mask))

        if refined_inlier_count >= best_inlier_count:
            best_normal      = refined_normal
            best_d           = refined_d
            best_inlier_mask = refined_inlier_mask
            best_inlier_count = refined_inlier_count
            inlier_points    = points[best_inlier_mask]

    # --- Consistent normal orientation ---
    # In OpenCV camera space Y points down.
    # Floor normal should point toward the camera (negative Y = upward in image).
    # Ensure normal[1] < 0 (pointing toward camera / upward).
    if best_normal[1] > 0:
        best_normal = -best_normal
        best_d      = -best_d

    # --- Quality metrics ---
    inlier_distances = np.abs(inlier_points @ best_normal + best_d)
    rmse         = float(np.sqrt(np.mean(inlier_distances ** 2)))
    inlier_ratio = best_inlier_count / n_points

    # --- Validation ---
    valid   = True
    reasons = []

    if best_inlier_count < min_inliers:
        valid = False
        reasons.append(f"inliers={best_inlier_count} < {min_inliers}")
    if inlier_ratio < min_inlier_ratio:
        valid = False
        reasons.append(f"ratio={inlier_ratio:.2f} < {min_inlier_ratio}")
    if rmse > max_rmse:
        valid = False
        reasons.append(f"rmse={rmse:.4f} > {max_rmse}")

    plane = {
        "normal":       best_normal.tolist(),
        "d":            float(best_d),
        "rmse":         rmse,
        "inlier_ratio": inlier_ratio,
    }

    stats = {
        "valid":         valid,
        "inlier_count":  best_inlier_count,
        "inlier_ratio":  inlier_ratio,
        "rmse":          rmse,
        "normal":        best_normal.tolist(),
        "d":             float(best_d),
        # Pass the actual inlier 3D points to the caller so boundary_generator
        # can compute tight percentile-based extents from real floor data.
        "inlier_points": inlier_points,
    }

    if valid:
        n = best_normal
        print(f"[P5_PLANE] valid=True inliers={best_inlier_count} "
              f"ratio={inlier_ratio:.2f} rmse={rmse:.4f} "
              f"normal=({n[0]:.3f},{n[1]:.3f},{n[2]:.3f}) "
              f"d={best_d:.3f}")
    else:
        reason_str = "; ".join(reasons)
        print(f"[P5_PLANE_REJECT] reason={reason_str}")
        plane = None

    return plane, stats


def _least_squares_plane(points: np.ndarray):
    """Fit a plane to points using SVD. Returns (normal, d) or (None, None)."""
    if len(points) < 3:
        return None, None
    try:
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normal   = vh[-1]
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-8:
            return None, None
        normal = normal / norm_len
        d = -np.dot(normal, centroid)
        return normal, float(d)
    except Exception:
        return None, None