"""
HandDetector — MediaPipe hand landmark + custom gesture recognizer.

Final gesture fix:
- MediaPipe's default gesture_recognizer.task does not provide Pinch in this setup.
- We use MediaPipe only for landmarks and compute POINT / Pinch / Open_Palm / Closed_Fist ourselves.
- Thumb_Up and Thumb_Down are ignored for this project because they were noisy in your logs.
"""

import os
import time
from typing import Optional

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from config import (
    FIST_TIP_MCP_MAX_RATIO,
    HAND_DETECT_CONF,
    HAND_PRESENCE_CONF,
    HAND_TRACK_CONF,
    OPEN_EXTEND_MARGIN,
    PINCH_INDEX_MIN_RATIO,
    PINCH_MAX_RATIO,
    POINT_CURLED_MAX,
    POINT_EXTENDED_MIN,
)
from geometry import generate_guardian_polygon, project_fingertip_to_floor
from smoother import GestureCooldown, GestureVoteBuffer, LandmarkSmoother


def _dist_lm(a, b) -> float:
    return float(((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2) ** 0.5)


def _scale(landmarks) -> float:
    lm = landmarks
    palm_a = _dist_lm(lm[0], lm[9])
    palm_b = _dist_lm(lm[5], lm[17])
    palm_c = _dist_lm(lm[0], lm[5])
    return max(palm_a, palm_b, palm_c, 1e-6)


def _ratio(landmarks, a: int, b: int, scale: float) -> float:
    return _dist_lm(landmarks[a], landmarks[b]) / max(scale, 1e-6)


def _finger_extended(landmarks, mcp: int, pip: int, tip: int, scale: float) -> bool:
    wrist = landmarks[0]
    tip_dist = _dist_lm(wrist, landmarks[tip])
    pip_dist = _dist_lm(wrist, landmarks[pip])
    mcp_dist = _dist_lm(wrist, landmarks[mcp])
    return tip_dist > pip_dist + OPEN_EXTEND_MARGIN * scale and tip_dist > mcp_dist + 0.16 * scale


def _finger_curled(landmarks, mcp: int, tip: int, scale: float) -> bool:
    return _ratio(landmarks, tip, mcp, scale) < FIST_TIP_MCP_MAX_RATIO


def _is_point_gesture(landmarks) -> bool:
    lm = landmarks
    hand_span = _scale(lm)

    index_r = _ratio(lm, 8, 5, hand_span)
    middle_r = _ratio(lm, 12, 9, hand_span)
    ring_r = _ratio(lm, 16, 13, hand_span)
    pinky_r = _ratio(lm, 20, 17, hand_span)

    return (
        index_r > POINT_EXTENDED_MIN
        and middle_r < POINT_CURLED_MAX
        and ring_r < POINT_CURLED_MAX
        and pinky_r < POINT_CURLED_MAX
    )


def _is_pinch_gesture(landmarks) -> bool:
    lm = landmarks
    hand_span = _scale(lm)

    pinch_ratio = _ratio(lm, 4, 8, hand_span)
    index_mcp_ratio = _ratio(lm, 8, 5, hand_span)

    middle_extended = _finger_extended(lm, 9, 10, 12, hand_span)
    ring_extended = _finger_extended(lm, 13, 14, 16, hand_span)
    pinky_extended = _finger_extended(lm, 17, 18, 20, hand_span)

    # Pinch: thumb tip close to index tip, with index not fully collapsed into the MCP.
    # Reject open palm-like poses where the other fingers are all strongly extended.
    return (
        pinch_ratio <= PINCH_MAX_RATIO
        and index_mcp_ratio >= PINCH_INDEX_MIN_RATIO
        and not (middle_extended and ring_extended and pinky_extended)
    )


def _is_open_palm_gesture(landmarks) -> bool:
    lm = landmarks
    hand_span = _scale(lm)

    index_extended = _finger_extended(lm, 5, 6, 8, hand_span)
    middle_extended = _finger_extended(lm, 9, 10, 12, hand_span)
    ring_extended = _finger_extended(lm, 13, 14, 16, hand_span)
    pinky_extended = _finger_extended(lm, 17, 18, 20, hand_span)

    extended_count = sum([index_extended, middle_extended, ring_extended, pinky_extended])
    pinch_ratio = _ratio(lm, 4, 8, hand_span)

    return extended_count >= 3 and pinch_ratio > PINCH_MAX_RATIO * 1.18


def _is_fist_gesture(landmarks) -> bool:
    lm = landmarks
    hand_span = _scale(lm)

    index_curled = _finger_curled(lm, 5, 8, hand_span)
    middle_curled = _finger_curled(lm, 9, 12, hand_span)
    ring_curled = _finger_curled(lm, 13, 16, hand_span)
    pinky_curled = _finger_curled(lm, 17, 20, hand_span)

    return sum([index_curled, middle_curled, ring_curled, pinky_curled]) >= 3


def _choose_project_gesture(landmarks, mp_category: str) -> str:
    """
    Project-specific gesture classifier.

    Priority matters:
    - Pinch must win over generic None / Thumb noise.
    - Open_Palm must be reliable for release.
    - Thumb_Up / Thumb_Down are ignored because they were noisy in the production logs.
    """
    if _is_pinch_gesture(landmarks):
        return "Pinch"

    if _is_open_palm_gesture(landmarks):
        return "Open_Palm"

    if _is_point_gesture(landmarks):
        return "POINT"

    if _is_fist_gesture(landmarks):
        return "Closed_Fist"

    if mp_category == "Pointing_Up":
        return "POINT"

    if mp_category in ("Closed_Fist", "Open_Palm"):
        return mp_category

    return "None"


def _build_hand_data(
    smoothed: Optional[np.ndarray],
    gesture: str,
    confirmed: Optional[str],
    cursor,
    confidence: float = 0.0,
    handedness: str = "",
) -> dict:
    return {
        "type": "HAND_DATA",
        "gesture": gesture or "None",
        "gesture_confirmed": confirmed is not None,
        "landmarks_smoothed": smoothed.tolist() if smoothed is not None else [],
        "projected_floor_point": cursor,
        "confidence": float(confidence or 0.0),
        "handedness": handedness or "",
    }


def _build_point_added(pt: list, session: dict) -> dict:
    return {
        "type": "POINT_ADDED",
        "point_index": len(session["boundary_pts"]) - 1,
        "world_position": pt,
        "total_points": len(session["boundary_pts"]),
    }


def _build_point_removed(session: dict) -> dict:
    return {
        "type": "POINT_ADDED",
        "action": "REMOVED",
        "point_index": len(session["boundary_pts"]) - 1,
        "world_position": [],
        "total_points": len(session["boundary_pts"]),
    }


def _build_guardian_ready(session: dict) -> dict:
    try:
        ordered, area, centroid = generate_guardian_polygon(session["boundary_pts"])
    except ValueError as exc:
        return {"type": "WARNING", "message": str(exc)}

    session["guardian_polygon"] = ordered

    xs = [p[0] for p in ordered]
    ys = [p[1] for p in ordered]
    zs = [p[2] for p in ordered]

    return {
        "type": "GUARDIAN_READY",
        "point_count": len(ordered),
        "polygon": ordered,
        "area_m2": round(area, 3),
        "centroid": centroid,
        "bounding_box": {
            "min": [min(xs), min(ys), min(zs)],
            "max": [max(xs), max(ys), max(zs)],
        },
    }


def _decode_jpeg(jpeg_bytes: bytes):
    import cv2

    buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        return None

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class HandDetector:
    def __init__(self, model_path: str = "gesture_recognizer.task"):
        model_path = os.path.abspath(model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"gesture_recognizer.task not found at: {model_path}")

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=HAND_DETECT_CONF,
            min_hand_presence_confidence=HAND_PRESENCE_CONF,
            min_tracking_confidence=HAND_TRACK_CONF,
        )

        self._recognizer = mp_vision.GestureRecognizer.create_from_options(options)
        self._smoother = LandmarkSmoother()
        self._vote_buf = GestureVoteBuffer()
        self._cooldown = GestureCooldown()
        self._last_timestamp_ms = -1
        self._last_log_time = 0.0

        print(f"[HAND] detector initialized with model={model_path}", flush=True)

    def process_frame(
        self,
        jpeg_bytes: bytes,
        pose_matrix: list,
        timestamp_ms: int,
        session: dict,
    ) -> dict:
        state = session.get("state") or "INIT"

        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        rgb = _decode_jpeg(jpeg_bytes)
        if rgb is None:
            print("[HAND] JPEG decode failed", flush=True)
            return {"type": "WARNING", "message": "JPEG decode failed"}

        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._recognizer.recognize_for_video(mp_img, timestamp_ms)

        now = time.monotonic()

        if not result.hand_landmarks:
            self._vote_buf.push("None")
            if now - self._last_log_time > 2.0:
                print(f"[HAND] no hand landmarks detected state={state}", flush=True)
                self._last_log_time = now
            return _build_hand_data(None, "None", None, None, 0.0, "")

        raw_lm = result.hand_landmarks[0]
        smoothed = self._smoother.smooth(raw_lm)

        mp_category = result.gestures[0][0].category_name if result.gestures else "None"
        mp_confidence = float(result.gestures[0][0].score) if result.gestures else 0.0
        handedness = result.handedness[0][0].category_name if result.handedness else ""

        gesture = _choose_project_gesture(raw_lm, mp_category)
        confirmed = self._vote_buf.push(gesture)

        if now - self._last_log_time > 0.45:
            print(
                f"[HAND] gesture={gesture} confirmed={confirmed} "
                f"mp={mp_category} confidence={mp_confidence:.3f} handedness={handedness} state={state}",
                flush=True,
            )
            self._last_log_time = now

        placing_points = state == "PLACING_POINTS"
        has_floor = bool(session.get("floor"))

        if placing_points and confirmed and self._cooldown.is_allowed(confirmed):
            if confirmed == "Pinch":
                if has_floor:
                    pt = project_fingertip_to_floor(smoothed, pose_matrix, session["floor"])
                    if pt:
                        session["boundary_pts"].append(pt)
                        self._cooldown.mark_fired("Pinch")
                        print(f"[HAND] point added at {pt}", flush=True)
                        return _build_point_added(pt, session)

                return _build_hand_data(smoothed, gesture, confirmed, None, mp_confidence, handedness)

            if confirmed in ("Closed_Fist", "Fist"):
                if len(session["boundary_pts"]) >= 4:
                    self._cooldown.mark_fired(confirmed)
                    session["state"] = "GUARDIAN_READY"
                    print("[HAND] guardian ready", flush=True)
                    return _build_guardian_ready(session)

                return {
                    "type": "WARNING",
                    "message": f"Need 4 points, have {len(session['boundary_pts'])}",
                }

            if confirmed == "Open_Palm":
                if session["boundary_pts"]:
                    removed = session["boundary_pts"].pop()
                    self._cooldown.mark_fired("Open_Palm")
                    print(f"[HAND] removed point {removed}", flush=True)
                    return _build_point_removed(session)

        cursor = None
        if gesture == "POINT" and has_floor:
            cursor = project_fingertip_to_floor(smoothed, pose_matrix, session["floor"])

        return _build_hand_data(smoothed, gesture, confirmed, cursor, mp_confidence, handedness)

    def reset(self) -> None:
        print("[HAND] detector reset", flush=True)
        self._smoother.reset()
        self._vote_buf.reset()
        self._cooldown.reset()
        self._last_timestamp_ms = -1