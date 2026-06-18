# ===========================================================================
# geometry/floor_from_depth.py — Geometry-first floor detection.
#
# WHY THIS EXISTS
#   SegFormer-b0 (ADE20K) is unreliable on glossy / patterned tile floors: it
#   confidently mislabels the floor as "bed", "sky", "wall" or "bathtub"
#   (verified on the debug frames — floor_prob ~0.005-0.05 on a floor filling
#   ~50% of the frame). Because the old pipeline *gated* all geometry on that
#   semantic label, the guardian never locked ("semantic_floor_too_low",
#   boundary w=0 d=0 for an entire scan).
#
#   The floor, however, is the dominant near-horizontal PLANE in the lower
#   image — a property the monocular depth model (Depth-Anything V2) captures
#   reliably even on reflective tile. This module detects the floor from that
#   plane instead of from a class label.
#
# APPROACH
#   1. Restrict to a geometric ROI (lower part of the image, below the horizon).
#   2. Back-project ROI pixels with metric depth + intrinsics → 3D point cloud.
#   3. RANSAC + LSQ fit the dominant plane (reuses geometry.plane_fit).
#   4. Floor mask = every ROI pixel whose 3D point lies within `inlier_threshold`
#      of that plane. Clean up with morphology + largest connected component.
#
#   HYBRID: an optional `semantic_floor_mask` (SegFormer) is fused in
#   ADDITIVELY — semantic-floor pixels that also lie near the fitted plane are
#   unioned into the mask, so SegFormer recovers floor that strict geometry
#   banding/morphology dropped, and agreement boosts confidence. Semantics are
#   never used to SUBTRACT floor here (the model is too wrong on glossy tile to
#   be trusted for subtraction). An optional `exclude_mask` is supported for
#   callers that have a *trustworthy* non-floor mask, but it defaults to None.
# ===========================================================================

import numpy as np

from geometry.plane_fit import fit_floor_plane


def detect_floor_from_depth(
    depth_map: np.ndarray,
    fx: float,
    fy: float,
    cx: float = None,
    cy: float = None,
    roi_top_frac: float = 0.40,
    min_depth: float = 0.1,
    max_depth: float = 10.0,
    inlier_threshold: float = 0.06,
    ransac_iterations: int = 300,
    max_ransac_points: int = 8000,
    min_camera_horizontal: float = 0.35,
    morph_kernel: int = 9,
    exclude_mask: np.ndarray = None,
    semantic_floor_mask: np.ndarray = None,
    semantic_band_mult: float = 2.0,
    semantic_prob: np.ndarray = None,
    semantic_high_conf: float = 0.60,
    semantic_high_band_mult: float = 2.5,
    frame_id: str = "?",
) -> tuple:
    """
    Detect the floor as the dominant horizontal plane in the depth map.

    Args:
        depth_map         : HxW float32 metric depth (meters).
        fx, fy            : Focal lengths (pixels).
        cx, cy            : Principal point (defaults to image center).
        roi_top_frac      : Ignore pixels above this fraction of image height
                            (0.40 → only the lower 60% can be floor).
        inlier_threshold  : Max distance (m) from the plane to count as floor.
        min_camera_horizontal : Soft guard — |normal.y| in CAMERA space must be
                            at least this (rejects a near-vertical wall plane).
                            Kept lenient; the authoritative gravity-aligned
                            horizontality check happens downstream using pose.
        exclude_mask      : Optional HxW bool of pixels to drop from the ROI
                            (only pass a *trusted* non-floor mask). Default None.
        semantic_floor_mask : Optional HxW bool of SegFormer "floor" pixels. When
                            given (HYBRID mode), semantic-floor pixels that lie
                            within `semantic_band_mult * inlier_threshold` of the
                            fitted plane are UNIONED into the geometry mask. This
                            lets SegFormer recover floor that morphology/strict
                            banding dropped, while the plane distance check keeps
                            the model's mislabels (floor→bed) from leaking in.
                            Semantic is never used to SUBTRACT floor here.
        semantic_band_mult : Relaxed plane band (×inlier_threshold) for the
                            semantic union. Default 2.0 (e.g. 12 cm at 6 cm inlier).
        frame_id          : For logging.

    Returns:
        (floor_mask, stats)
        floor_mask : HxW bool (True = floor). Empty mask on rejection.
        stats keys : floor_pixels, ratio, confidence, used_fallback (False),
                     rejected (bool), reject_reason (str|None),
                     plane (dict|None), plane_normal, inlier_ratio, rmse,
                     camera_horizontal, mode="geometry".
    """
    h, w = depth_map.shape[:2]
    if cx is None:
        cx = w / 2.0
    if cy is None:
        cy = h / 2.0

    total_pixels = h * w
    empty = np.zeros((h, w), dtype=bool)

    def _reject(reason):
        return empty, {
            "floor_pixels": 0, "ratio": 0.0, "confidence": 0.0,
            "used_fallback": False, "rejected": True, "reject_reason": reason,
            "plane": None, "plane_normal": None, "inlier_ratio": 0.0,
            "rmse": 0.0, "camera_horizontal": 0.0, "mode": "geometry",
        }

    # ------------------------------------------------------------------
    # Step 1: geometric ROI (lower image) ∩ valid depth
    # ------------------------------------------------------------------
    roi = np.zeros((h, w), dtype=bool)
    roi[int(h * roi_top_frac):, :] = True
    if exclude_mask is not None:
        roi &= ~exclude_mask.astype(bool)

    valid = (
        roi &
        np.isfinite(depth_map) &
        (depth_map > min_depth) &
        (depth_map < max_depth)
    )
    vs, us = np.where(valid)
    if len(vs) < 200:
        return _reject(f"too_few_roi_depth_pixels ({len(vs)})")

    # full-resolution 3D for the ROI (vectorized pinhole back-projection)
    z_all = depth_map[vs, us].astype(np.float32)
    x_all = (us.astype(np.float32) - cx) * z_all / fx
    y_all = (vs.astype(np.float32) - cy) * z_all / fy
    pts_all = np.stack([x_all, y_all, z_all], axis=1)

    # ------------------------------------------------------------------
    # Step 2: RANSAC plane on a downsampled subset (speed)
    # ------------------------------------------------------------------
    if len(pts_all) > max_ransac_points:
        sel = np.random.choice(len(pts_all), max_ransac_points, replace=False)
        pts_fit = pts_all[sel]
    else:
        pts_fit = pts_all

    plane, pstats = fit_floor_plane(
        pts_fit,
        ransac_iterations=ransac_iterations,
        inlier_threshold=inlier_threshold,
        min_inliers=max(100, len(pts_fit) // 20),
        min_inlier_ratio=0.30,
        max_rmse=0.12,
        frame_id=frame_id,
    )
    if plane is None:
        return _reject("no_dominant_plane")

    normal = np.array(plane["normal"], dtype=np.float64)
    d_val = float(plane["d"])
    camera_horizontal = float(abs(normal[1]))  # |Y| → 1.0 means horizontal

    if camera_horizontal < min_camera_horizontal:
        return _reject(
            f"plane_not_horizontal_camera ({camera_horizontal:.2f} "
            f"< {min_camera_horizontal:.2f})"
        )

    # ------------------------------------------------------------------
    # Step 3: dense floor mask = all ROI pixels near the plane
    # ------------------------------------------------------------------
    dist_all = np.abs(pts_all @ normal + d_val)
    near = dist_all < inlier_threshold
    include = near

    # HYBRID: union SegFormer floor pixels that are also near the plane (relaxed
    # band). Semantic extends the geometry mask where it agrees with the floor
    # plane, but the plane test blocks the model's floor→bed/wall mislabels.
    semantic_agreement = 0.0
    semantic_high_pixels = 0
    if semantic_floor_mask is not None:
        sem_at = semantic_floor_mask.astype(bool)[vs, us]
        near_relaxed = dist_all < (semantic_band_mult * inlier_threshold)
        include = near | (sem_at & near_relaxed)

        # Phase 14: trust HIGH-confidence SegFormer floor pixels strongly — union
        # them within a WIDER plane band so confident floor is recovered even
        # where the depth plane is noisy. The (still bounded) band keeps confident
        # mislabels that sit far off the trusted RANSAC plane out.
        if semantic_prob is not None:
            sem_prob_at = semantic_prob.astype(np.float32)[vs, us]
            sem_high = sem_prob_at >= float(semantic_high_conf)
            near_high = dist_all < (semantic_high_band_mult * inlier_threshold)
            high_include = sem_high & near_high
            include = include | high_include
            semantic_high_pixels = int(high_include.sum())

        if near.any():
            semantic_agreement = float((sem_at & near).sum()) / float(near.sum())

    floor_mask = np.zeros((h, w), dtype=bool)
    floor_mask[vs[include], us[include]] = True

    floor_mask = _morph_clean(floor_mask, morph_kernel)

    floor_pixels = int(floor_mask.sum())
    ratio = floor_pixels / total_pixels if total_pixels > 0 else 0.0

    # confidence from plane quality (mirrors boundary_generator._compute_confidence)
    rmse = float(pstats.get("rmse", 0.1))
    inlier_ratio = float(pstats.get("inlier_ratio", 0.0))
    rmse_score = max(0.0, 1.0 - rmse / 0.1)
    # Keep the RAW (unclamped) score and the CLIPPED score separate so the log
    # doesn't always read conf=1.00 — the operator can see how much headroom the
    # semantic rewards added past saturation.
    raw_score = 0.5 * rmse_score + 0.5 * min(1.0, inlier_ratio)
    # HYBRID: reward frames where SegFormer agrees with the floor plane.
    if semantic_floor_mask is not None:
        raw_score += 0.15 * semantic_agreement
    # Phase 14: extra reward when HIGH-confidence SegFormer floor agrees.
    if semantic_high_pixels > 0 and floor_pixels > 0:
        raw_score += 0.10 * min(1.0, semantic_high_pixels / float(floor_pixels))
    clipped_score = min(1.0, raw_score)
    confidence = round(clipped_score, 3)

    print(
        f"[GEO_FLOOR] id={frame_id} ratio={ratio:.3f} pixels={floor_pixels} "
        f"conf={confidence:.3f} raw_score={raw_score:.3f} clipped_score={clipped_score:.3f} "
        f"cam_horiz={camera_horizontal:.2f} "
        f"inlier_ratio={inlier_ratio:.2f} rmse={rmse:.4f} "
        f"sem_agree={semantic_agreement:.2f} sem_high_px={semantic_high_pixels} "
        f"normal=({normal[0]:.2f},{normal[1]:.2f},{normal[2]:.2f})"
    )

    return floor_mask, {
        "floor_pixels": floor_pixels,
        "ratio": ratio,
        "confidence": confidence,
        "used_fallback": False,
        "rejected": False,
        "reject_reason": None,
        "plane": plane,
        "plane_normal": normal.tolist(),
        "inlier_ratio": inlier_ratio,
        "rmse": rmse,
        "camera_horizontal": camera_horizontal,
        "semantic_agreement": round(semantic_agreement, 3),
        "semantic_high_pixels": semantic_high_pixels,
        "raw_score": round(raw_score, 3),
        "clipped_score": round(clipped_score, 3),
        "mode": "hybrid" if semantic_floor_mask is not None else "geometry",
    }


def _morph_clean(mask: np.ndarray, kernel: int = 9) -> np.ndarray:
    """Close gaps, drop speckle, keep the largest blob, fill interior holes."""
    import cv2

    m = (mask.astype(np.uint8)) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    m = cv2.morphologyEx(
        m, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )

    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return m.astype(bool)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = (lab == biggest).astype(np.uint8) * 255

    # flood-fill interior holes (e.g. reflection spots that escaped the plane band)
    hh, ww = out.shape
    ff = out.copy()
    fmask = np.zeros((hh + 2, ww + 2), np.uint8)
    cv2.floodFill(ff, fmask, (0, 0), 255)
    out = out | cv2.bitwise_not(ff)
    return out.astype(bool)
