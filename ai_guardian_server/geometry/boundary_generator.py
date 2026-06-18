# ===========================================================================
# geometry/boundary_generator.py — Phase 9.1
# Smart Safe Rectangle — generates the largest stable guardian boundary
# centered on the user/camera-forward floor region.
#
# Phase 9.1 fixes:
#   - Camera/user position used as preferred center (not arbitrary median)
#   - Outlier removal via IQR before rectangle fitting
#   - Rectangle grows from preferred center toward available floor edges
#   - Minimum 1.0m x 1.0m, maximum 3.0m x 3.0m
#   - Prefers largest safe rectangle centered on usable floor
#   - [P9_SMART_CUBE] diagnostic logging
#
# Coordinate conversion:
#   OpenCV camera space: X=right, Y=down,  Z=forward
#   Unity  camera space: X=right, Y=up,    Z=forward
#   Mapping: X → x, Y → -y, Z → z
# ===========================================================================

import numpy as np

from config import (
    GUARDIAN_FLOOR_LIFT_M,
    FLOOR_CANDIDATE_MIN_W,
    FLOOR_CANDIDATE_MIN_D,
    FLOOR_CANDIDATE_MAX_W,
    FLOOR_CANDIDATE_MAX_D,
    FLOOR_CANDIDATE_SAFETY_M,
    FLOOR_CANDIDATE_COUNT,
)


def generate_boundary(
    plane: dict,
    inlier_points: np.ndarray = None,
    frame_id: str = "?",
    **kwargs,
) -> tuple:
    """
    Generate Guardian boundary using Smart Safe Rectangle algorithm.

    Phase 9.1 algorithm:
    1. Project inlier floor points to XZ plane
    2. Remove outliers using IQR filtering
    3. Compute usable floor bounds
    4. Use camera-forward position (z=0, x=0 in camera space) as preferred center
    5. Grow rectangle from preferred center to fill available floor
    6. Clamp to min/max size
    7. Generate 3 candidates: conservative, medium, largest

    Returns:
        (result, stats)
    """
    normal = np.array(plane["normal"], dtype=np.float64)
    d_val  = float(plane["d"])
    a, b, c = normal

    if abs(b) < 1e-6:
        reason = f"plane normal Y too small (b={b:.6f})"
        print(f"[P9_SMART_CUBE_REJECT] frame={frame_id} reason={reason}")
        return None, {"valid": False, "reason": reason}

    lift = GUARDIAN_FLOOR_LIFT_M

    if inlier_points is None or len(inlier_points) < 20:
        reason = "insufficient_inlier_points"
        count = len(inlier_points) if inlier_points is not None else 0
        print(f"[P9_SMART_CUBE_REJECT] frame={frame_id} reason={reason} count={count}")
        return None, {
            "valid": False,
            "reason": reason,
            "message": "Not enough floor data. Point camera at the floor.",
        }

    # ==================================================================
    # Step 1: Project to XZ and remove outliers
    # ==================================================================
    xs = inlier_points[:, 0]
    zs = inlier_points[:, 2]

    # IQR-based outlier removal (more robust than simple percentiles)
    xs_clean, zs_clean = _iqr_filter(xs, zs, factor=1.5)

    if len(xs_clean) < 15:
        # Fallback to basic percentile if IQR removes too many
        xs_clean = xs
        zs_clean = zs
        print(f"[P9_SMART_CUBE] IQR too aggressive, using full points n={len(xs)}")

    # Compute floor bounds from cleaned data.
    # Phase 13 [LARGER_AREA]: widen the percentile window 5/95 → 2/98. The IQR
    # filter above already removed gross outliers, so the remaining 2nd–98th
    # percentile band is reliable floor. The old 5/95 trim discarded ~10% of the
    # *measured* floor extent on every axis, which (combined with the safety
    # margin) was the dominant reason the locked boundary came out ~1.2–1.5m
    # instead of reaching the 3m cap. Recovering that band lets the high-
    # confidence overlapping floor region grow toward FLOOR_CANDIDATE_MAX (3.0m).
    x_lo = float(np.percentile(xs_clean, 2))
    x_hi = float(np.percentile(xs_clean, 98))
    z_lo = float(np.percentile(zs_clean, 2))
    z_hi = float(np.percentile(zs_clean, 98))

    visible_w = x_hi - x_lo
    visible_d = z_hi - z_lo

    # ==================================================================
    # Step 2: Compute safe floor region with safety margin
    # ==================================================================
    margin = FLOOR_CANDIDATE_SAFETY_M
    safe_x_lo = x_lo + margin
    safe_x_hi = x_hi - margin
    safe_z_lo = z_lo + margin
    safe_z_hi = z_hi - margin

    safe_w = safe_x_hi - safe_x_lo
    safe_d = safe_z_hi - safe_z_lo

    # Phase 10.1: removed the hard-coded `max(MIN, 1.4)` floor that forced the
    # safe region to ≥1.4m per axis. Combined with FLOOR_CANDIDATE_SAFETY_M=0.15
    # it required the visible floor to be ≥1.7m × 1.7m — session_log.csv showed
    # this was the dominant rejection cause (`floor_too_small`). Now we honour
    # the config value FLOOR_CANDIDATE_MIN_W/D directly (default 1.0m) so a
    # ~1.2m visible floor patch passes through the candidate-pool gate.
    min_w = float(FLOOR_CANDIDATE_MIN_W)
    min_d = float(FLOOR_CANDIDATE_MIN_D)
    # Phase 2K: honour the config cap directly (was hard-capped at 3.0, which
    # silently shrank the smart-cube gate to a 3 m ceiling even after the world
    # accumulator was raised to 5 m). The LOCKED cube comes from the accumulator;
    # this only governs the per-frame acceptance candidate.
    max_w = float(FLOOR_CANDIDATE_MAX_W)
    max_d = float(FLOOR_CANDIDATE_MAX_D)

    print(f"[P9_SMART_CUBE] frame={frame_id} "
          f"raw_points={len(xs)} clean_points={len(xs_clean)} "
          f"visible=({visible_w:.2f},{visible_d:.2f}) "
          f"safe=({safe_w:.2f},{safe_d:.2f}) "
          f"bounds_x=[{x_lo:.2f},{x_hi:.2f}] bounds_z=[{z_lo:.2f},{z_hi:.2f}]")

    if safe_w < min_w or safe_d < min_d:
        print(f"[P9_SMART_CUBE_REJECT] frame={frame_id} "
              f"reason=floor_too_small safe=({safe_w:.2f},{safe_d:.2f}) "
              f"min=({min_w:.1f},{min_d:.1f})")
        return None, {
            "valid": False,
            "reason": "visible_floor_too_small",
            "message": "Move camera to see more floor.",
        }

    # ==================================================================
    # Step 3: Smart center selection
    # Prefer camera-forward position (x≈0, z=median of forward points)
    # This centers the boundary where the user is looking/standing
    # ==================================================================
    # Camera center in floor coordinates is roughly (0, 0, forward)
    # Use point density to find the most supported center
    cx_median = float(np.median(xs_clean))
    cz_median = float(np.median(zs_clean))

    # Prefer center near camera (x ≈ 0), but use floor data
    # Weight toward x=0 (camera center) if data supports it
    cx_camera = 0.0  # Camera is at x=0 in its own space
    if safe_x_lo <= cx_camera <= safe_x_hi:
        cx_preferred = cx_camera * 0.4 + cx_median * 0.6
    else:
        cx_preferred = cx_median

    # Z: prefer closer to camera (smaller z) for comfort
    cz_forward = float(np.percentile(zs_clean, 40))  # slightly forward-biased
    if safe_z_lo <= cz_forward <= safe_z_hi:
        cz_preferred = cz_forward * 0.5 + cz_median * 0.5
    else:
        cz_preferred = cz_median

    print(f"[P9_SMART_CUBE] center_selection: "
          f"median=({cx_median:.2f},{cz_median:.2f}) "
          f"preferred=({cx_preferred:.2f},{cz_preferred:.2f})")

    # ==================================================================
    # Step 4: Generate candidates — grow from preferred center
    # ==================================================================
    # Phase 13 [LARGER_AREA]: bias the candidate ladder toward larger rectangles.
    # The selector below prefers the largest candidate that clears the corner-
    # coverage gate, so raising the floor of the ladder (0.60→0.78) means even
    # the conservative fallback fills most of the usable floor instead of
    # collapsing to ~60% of it.
    scales = [0.78, 0.90, 1.00]
    labels = ["conservative", "medium", "largest"]
    candidates = []

    for i, (scale, label) in enumerate(zip(scales, labels)):
        cand_w = min(safe_w * scale, max_w)
        cand_d = min(safe_d * scale, max_d)

        # Enforce minimum
        cand_w = max(cand_w, min_w)
        cand_d = max(cand_d, min_d)

        # Skip if candidate exceeds available safe area
        if cand_w > safe_w + 0.01 or cand_d > safe_d + 0.01:
            continue

        half_w = cand_w / 2.0
        half_d = cand_d / 2.0

        # Center the rectangle around preferred center, constrained to safe region
        ccx = max(safe_x_lo + half_w, min(cx_preferred, safe_x_hi - half_w))
        ccz = max(safe_z_lo + half_d, min(cz_preferred, safe_z_hi - half_d))

        # Keep minimum distance from camera
        if ccz - half_d < 0.3:
            ccz = 0.3 + half_d

        corners = [
            (ccx - half_w, ccz - half_d),
            (ccx + half_w, ccz - half_d),
            (ccx + half_w, ccz + half_d),
            (ccx - half_w, ccz + half_d),
        ]

        # Verify all corners have floor coverage.
        # Phase 13 [LARGER_AREA]: radius 0.2 → 0.35m. A 0.2m search disc around
        # each corner is unforgiving at the *edge* of a depth map where points
        # are sparser — it kept failing the outer corners of the larger
        # candidates and forcing selection of a smaller one. 0.35m still
        # requires real floor near each corner but tolerates the natural point
        # thinning at the boundary so the largest safe rectangle can be picked.
        corner_coverage = _check_corner_coverage(corners, xs_clean, zs_clean, radius=0.35)

        points_camera = _project_corners_to_plane(corners, a, b, c, d_val, lift)
        area = cand_w * cand_d
        coverage = (cand_w * cand_d) / (safe_w * safe_d) if safe_w * safe_d > 0 else 0

        cand = {
            "id": i,
            "label": label,
            "points_camera": points_camera,
            "width_m": round(cand_w, 3),
            "depth_m": round(cand_d, 3),
            "area_m2": round(area, 3),
            "center_x": round(ccx, 3),
            "center_z": round(ccz, 3),
            "confidence": _compute_confidence(plane),
            "coverage_score": round(coverage, 3),
            "corner_coverage": round(corner_coverage, 3),
            "source": f"floor_candidate_{label}",
        }
        candidates.append(cand)

        print(f"[P9_SMART_CUBE] frame={frame_id} candidate={label} "
              f"w={cand_w:.2f} d={cand_d:.2f} area={area:.2f} "
              f"center=({ccx:.2f},{ccz:.2f}) corner_cov={corner_coverage:.2f}")

    if not candidates:
        print(f"[P9_SMART_CUBE_REJECT] frame={frame_id} reason=no_valid_candidates")
        return None, {
            "valid": False,
            "reason": "no_valid_candidates",
            "message": "Floor area too small for any candidate.",
        }

    # Select best candidate: prefer largest with good corner coverage.
    # Phase 13 [LARGER_AREA]: gate 0.5 → 0.35. Requiring half the corners to
    # have nearby floor data was rejecting otherwise-good large rectangles where
    # one outer corner sat just past the dense depth region. 0.35 (≈ at least
    # 1.5 of 4 corners covered, combined with the wider 0.35m search radius)
    # admits the largest stable rectangle while still rejecting candidates that
    # float over a hole in the floor scan.
    best_idx = len(candidates) - 1  # largest by default
    for idx in range(len(candidates) - 1, -1, -1):
        if candidates[idx]["corner_coverage"] >= 0.35:
            best_idx = idx
            break

    best = candidates[best_idx]
    confidence = best["confidence"]

    boundary = {
        "boundary_camera": best["points_camera"],
        "width": best["width_m"],
        "depth": best["depth_m"],
        "confidence": confidence,
        "source": best["source"],
    }

    stats = {
        "valid": True,
        "points": 4,
        "width": best["width_m"],
        "depth": best["depth_m"],
        "confidence": confidence,
        "source": best["source"],
        "visible_w": round(visible_w, 3),
        "visible_d": round(visible_d, 3),
        "safe_w": round(safe_w, 3),
        "safe_d": round(safe_d, 3),
        "selected_w": best["width_m"],
        "selected_d": best["depth_m"],
        "center_x": best["center_x"],
        "center_z": best["center_z"],
        "floor_candidates": candidates,
        "recommended_candidate": best_idx,
    }

    print(f"[P9_SMART_CUBE] frame={frame_id} SELECTED={best['label']} "
          f"w={best['width_m']:.2f} d={best['depth_m']:.2f} "
          f"center=({best['center_x']:.2f},{best['center_z']:.2f}) "
          f"floor_area={best['area_m2']:.2f}m²")

    return boundary, stats


# ==================================================================
# Helpers
# ==================================================================

def _iqr_filter(xs, zs, factor=1.5):
    """Remove outliers using Interquartile Range (IQR) method."""
    q1_x, q3_x = np.percentile(xs, [25, 75])
    q1_z, q3_z = np.percentile(zs, [25, 75])
    iqr_x = q3_x - q1_x
    iqr_z = q3_z - q1_z

    mask_x = (xs >= q1_x - factor * iqr_x) & (xs <= q3_x + factor * iqr_x)
    mask_z = (zs >= q1_z - factor * iqr_z) & (zs <= q3_z + factor * iqr_z)
    mask = mask_x & mask_z

    return xs[mask], zs[mask]


def _check_corner_coverage(corners, xs, zs, radius=0.2):
    """Check how many corners have nearby floor data points.
    Returns fraction (0-1) of corners with coverage."""
    covered = 0
    for cx, cz in corners:
        dist = np.sqrt((xs - cx)**2 + (zs - cz)**2)
        if np.any(dist < radius):
            covered += 1
    return covered / len(corners) if corners else 0


def _project_corners_to_plane(corners_xz, a, b, c, d_val, lift=0.0):
    """Project (x,z) corners onto floor plane, convert to Unity camera space."""
    boundary_points = []
    for x, z in corners_xz:
        y_opencv = -(a * x + c * z + d_val) / b
        boundary_points.append([
            round(x, 4),
            round(-y_opencv + lift, 4),
            round(z, 4),
        ])
    return boundary_points


def _compute_confidence(plane):
    """Compute confidence from plane quality metrics."""
    plane_rmse  = plane.get("rmse", 0.1)
    plane_ratio = plane.get("inlier_ratio", 0.5)
    rmse_score  = max(0.0, 1.0 - plane_rmse / 0.1)
    return round(0.5 * rmse_score + 0.5 * min(1.0, plane_ratio), 3)