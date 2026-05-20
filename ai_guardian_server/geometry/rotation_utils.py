# ===========================================================================
# geometry/rotation_utils.py — Image Rotation + Intrinsics Remapping
#
# When the phone camera frame arrives sideways, SegFormer (trained on upright
# images) confuses wall/floor.  This module provides:
#   - rotate_image(img, degrees)                 — pure image rotation
#   - rotate_frame_and_intrinsics(...)           — rotate image + remap fx/fy/cx/cy
#
# Rotation is always clockwise by 0 / 90 / 180 / 270 degrees.
#
# Intrinsics remapping formulas (exact for pinhole cameras):
#
#   0°:          unchanged
#   90° CW:      new_w=h, new_h=w, fx'=fy, fy'=fx, cx'=h-cy, cy'=cx
#   180°:        new_w=w, new_h=h, fx'=fx, fy'=fy, cx'=w-cx,  cy'=h-cy
#   270° CW:     new_w=h, new_h=w, fx'=fy, fy'=fx, cx'=cy,    cy'=w-cx
# ===========================================================================

import numpy as np
import cv2


def rotate_image(img: np.ndarray, degrees: int) -> np.ndarray:
    """
    Rotate an image clockwise by 0, 90, 180, or 270 degrees.
    Returns a new array (never modifies the input).
    """
    degrees = degrees % 360
    if degrees == 0:
        return img.copy()
    elif degrees == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif degrees == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif degrees == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        raise ValueError(f"rotation must be 0/90/180/270, got {degrees}")


def rotate_frame_and_intrinsics(
    rgb: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    degrees: int,
) -> tuple:
    """
    Rotate *both* the image and the camera intrinsics consistently.

    This must be used whenever a frame is rotated before segmentation or
    backprojection, so that pixel-to-3D unprojection stays correct.

    Args:
        rgb     : HxW(x3) image (BGR or RGB, doesn't matter).
        fx, fy  : Focal lengths in pixels (from Unity camera intrinsics).
        cx, cy  : Principal point in pixels.
        degrees : Clockwise rotation — 0, 90, 180, or 270.

    Returns:
        (rotated_image, fx_new, fy_new, cx_new, cy_new)
    """
    degrees = degrees % 360
    h, w = rgb.shape[:2]

    if degrees == 0:
        return rgb.copy(), fx, fy, cx, cy

    rotated = rotate_image(rgb, degrees)

    if degrees == 90:
        # 90° clockwise: width and height swap
        fx_new = fy
        fy_new = fx
        cx_new = h - cy
        cy_new = cx
    elif degrees == 180:
        # 180°: width and height stay, principal point mirrors
        fx_new = fx
        fy_new = fy
        cx_new = w - cx
        cy_new = h - cy
    elif degrees == 270:
        # 270° clockwise (= 90° counter-clockwise): width and height swap
        fx_new = fy
        fy_new = fx
        cx_new = cy
        cy_new = w - cx
    else:
        raise ValueError(f"rotation must be 0/90/180/270, got {degrees}")

    return rotated, fx_new, fy_new, cx_new, cy_new