"""
HandDetector — wraps MediaPipe GestureRecognizer.
Runs entirely inside the CV worker process (own GIL).
"""

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from config import (
    HAND_DETECT_CONF,
    HAND_PRESENCE_CONF,
    HAND_TRACK_CONF,
    POINT_CURLED_MAX,
    POINT_EXTENDED_MIN,
)
from geometry import generate_guardian_polygon, project_fingertip_to_floor
from smoother import GestureCooldown, GestureVoteBuffer, LandmarkSmoother


def _dist(a, b) -> float:
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2) ** 0.5


def _is_point_gesture(landmarks) -> bool:
    """Custom POINT gesture: index extended, others curled. Scale-invariant."""
    lm = landmarks
    hand_span = _dist(lm[0], lm[9])
    if hand_span < 1e-6:
        return False

    index_r = _dist(lm[8], lm[5]) / hand_span
    middle_r = _dist(lm[12], lm[9]) / hand_span
    ring_r = _dist(lm[16], lm[13]) / hand_span
    pinky_r = _dist(lm[20], lm[17]) / hand_span

    return (
        index_r > POINT_EXTENDED_MIN
        and middle_r < POINT_CURLED_MAX
        and ring_r < POINT_CURLED_MAX
        and pinky_r < POINT_CURLED_MAX
    )


def _build_hand_data(
    smoothed: np.ndarray | None,
    gesture: str,
    confirmed: str | None,
    cursor,
) -> dict:
    return {
        "type": "HAND_DATA",
        "gesture": gesture,
        "gesture_confirmed": confirmed is not None,
        "landmarks_smoothed": smoothed.tolist() if smoothed is not None else [],
        "projected_floor_point": cursor,
    }


def _build_point_added(pt: list, session: dict) -> dict:
    return {
        "type": "POINT_ADDED",
        "point_index": len(session["boundary_pts"]) - 1,
        "world_position": pt,
        "total_points": len(session["boundary_pts"]),
    }


def _build_guardian_ready(session: dict) -> dict | None:
    try:
        ordered, area, centroid = generate_guardian_polygon(session["boundary_pts"])
    except ValueError as exc:
        return {"type": "WARNING", "message": str(exc)}

    pts = ordered
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    zs = [p[2] for p in pts]

    return {
        "type": "GUARDIAN_READY",
        "point_count": len(pts),
        "polygon": pts,
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

        print(f"[HAND] detector initialized with model={model_path}")

    def process_frame(
        self,
        jpeg_bytes: bytes,
        pose_matrix: list,
        timestamp_ms: int,
        session: dict,
    ) -> dict | None:
        """Returns a response dict to be sent back to Unity, or None."""
        state = session.get("state")
        print(f"[HAND] process_frame state={state} ts={timestamp_ms}")

        if state != "PLACING_POINTS":
            print("[HAND] skipping frame because state is not PLACING_POINTS")
            return None

        rgb = _decode_jpeg(jpeg_bytes)
        if rgb is None:
            print("[HAND] JPEG decode failed")
            return {"type": "WARNING", "message": "JPEG decode failed"}

        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._recognizer.recognize_for_video(mp_img, timestamp_ms)

        if not result.hand_landmarks:
            print("[HAND] no hand landmarks detected")
            self._vote_buf.push("NONE")
            return _build_hand_data(None, "NONE", None, None)

        raw_lm = result.hand_landmarks[0]
        smoothed = self._smoother.smooth(raw_lm)

        mp_cat = result.gestures[0][0].category_name if result.gestures else "NONE"
        gesture = "POINT" if mp_cat == "NONE" and _is_point_gesture(raw_lm) else mp_cat

        confirmed = self._vote_buf.push(gesture)
        print(f"[HAND] gesture={gesture} confirmed={confirmed}")

        if confirmed and self._cooldown.is_allowed(confirmed):
            if confirmed == "Pinch":
                if not session.get("floor"):
                    print("[HAND] pinch confirmed but floor data missing")
                    return {"type": "WARNING", "message": "Floor not confirmed yet"}

                pt = project_fingertip_to_floor(smoothed, pose_matrix, session["floor"])
                if pt:
                    session["boundary_pts"].append(pt)
                    self._cooldown.mark_fired("Pinch")
                    print(f"[HAND] point added at {pt}")
                    return _build_point_added(pt, session)

                print("[HAND] pinch confirmed but no valid floor projection")
                return _build_hand_data(smoothed, gesture, confirmed, None)

            elif confirmed == "Closed_Fist":
                if len(session["boundary_pts"]) >= 4:
                    self._cooldown.mark_fired("Closed_Fist")
                    session["state"] = "GUARDIAN_READY"
                    print("[HAND] guardian ready")
                    return _build_guardian_ready(session)

                print(f"[HAND] finalize attempted with only {len(session['boundary_pts'])} points")
                return {
                    "type": "WARNING",
                    "message": f"Need 4 points, have {len(session['boundary_pts'])}",
                }

            elif confirmed == "Open_Palm":
                if session["boundary_pts"]:
                    removed = session["boundary_pts"].pop()
                    self._cooldown.mark_fired("Open_Palm")
                    print(f"[HAND] removed point {removed}")

                return _build_hand_data(smoothed, gesture, confirmed, None)

        cursor = None
        if gesture == "POINT" and session.get("floor"):
            cursor = project_fingertip_to_floor(smoothed, pose_matrix, session["floor"])
            print(f"[HAND] cursor={cursor}")

        return _build_hand_data(smoothed, gesture, confirmed, cursor)

    def reset(self):
        print("[HAND] detector reset")
        self._smoother.reset()
        self._vote_buf.reset()
        self._cooldown.reset()