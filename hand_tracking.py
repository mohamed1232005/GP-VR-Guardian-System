"""Hand tracking pipeline for the active Mode 2/System A Python service.

This module owns the CV worker, MediaPipe hand detection, project-specific
landmark gesture classification, landmark smoothing, and gesture vote buffering.
It emits HAND_DATA only; Unity owns safe-space state and rehab interactions.
"""

import os
import time
import traceback
from collections import Counter, deque
from typing import Any, Optional

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from config import (
    dlog,
    EMA_ALPHA,
    FIST_TIP_MCP_MAX_RATIO,
    HAND_DETECT_CONF,
    HAND_PRESENCE_CONF,
    HAND_TRACK_CONF,
    MAX_JPEG_BYTES,
    OPEN_EXTEND_MARGIN,
    PINCH_INDEX_MIN_RATIO,
    PINCH_MAX_RATIO,
    POINT_CURLED_MAX,
    POINT_EXTENDED_MIN,
    VOTE_THRESHOLD,
    VOTE_WINDOW,
)


class LandmarkSmoother:
    """Exponential moving average for MediaPipe hand landmarks."""

    def __init__(self, alpha: float = EMA_ALPHA):
        self.alpha = float(alpha)
        self._last: np.ndarray | None = None

    def smooth(self, landmarks) -> np.ndarray:
        current = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

        if self._last is None or self._last.shape != current.shape:
            self._last = current
            return current

        self._last = self.alpha * current + (1.0 - self.alpha) * self._last
        return self._last

    def reset(self) -> None:
        self._last = None


class GestureVoteBuffer:
    """Confirm a gesture only after it wins a short rolling vote window."""

    def __init__(self, window: int = VOTE_WINDOW, threshold: int = VOTE_THRESHOLD):
        self.window = int(window)
        self.threshold = int(threshold)
        self._items: deque[str] = deque(maxlen=self.window)

    def push(self, gesture: str) -> str | None:
        gesture = gesture or "None"
        self._items.append(gesture)

        winner, count = Counter(self._items).most_common(1)[0]
        if winner != "None" and count >= self.threshold:
            return winner

        return None

    def reset(self) -> None:
        self._items.clear()


class HandDetector:
    """MediaPipe hand landmarks + project-specific gesture classification."""

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
        self._last_timestamp_ms = -1
        self._last_log_time = 0.0

        print(f"[HAND] detector initialized with model={model_path}", flush=True)

    def process_frame(self, jpeg_bytes: bytes, timestamp_ms: int) -> dict:
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        rgb = _decode_jpeg(jpeg_bytes)
        if rgb is None:
            dlog("[HAND] JPEG decode failed")
            return {"type": "WARNING", "message": "JPEG decode failed"}

        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._recognizer.recognize_for_video(mp_img, timestamp_ms)
        now = time.monotonic()

        if not result.hand_landmarks:
            self._vote_buf.push("None")
            if now - self._last_log_time > 2.0:
                dlog("[HAND] no hand landmarks detected")
                self._last_log_time = now
            return _build_hand_data(None, "None", None, 0.0, "")

        raw_lm = result.hand_landmarks[0]
        smoothed = self._smoother.smooth(raw_lm)

        mp_category = result.gestures[0][0].category_name if result.gestures else "None"
        mp_confidence = float(result.gestures[0][0].score) if result.gestures else 0.0
        handedness = result.handedness[0][0].category_name if result.handedness else ""

        gesture = _choose_project_gesture(raw_lm, mp_category)
        confirmed = self._vote_buf.push(gesture)

        if now - self._last_log_time > 0.45:
            dlog(
                f"[HAND] gesture={gesture} confirmed={confirmed} "
                f"mp={mp_category} confidence={mp_confidence:.3f} handedness={handedness}"
            )
            self._last_log_time = now

        return _build_hand_data(smoothed, gesture, confirmed, mp_confidence, handedness)

    def reset(self) -> None:
        print("[HAND] detector reset", flush=True)
        self._smoother.reset()
        self._vote_buf.reset()
        self._last_timestamp_ms = -1


def run_cv_worker(frame_queue: Any, result_queue: Any) -> None:
    """Read latest Unity frames, run hand tracking, and enqueue responses."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)

    detector = HandDetector(model_path=os.path.join(base_dir, "gesture_recognizer.task"))
    frame_counter = 0

    print("[CV] worker ready", flush=True)

    while True:
        item = frame_queue.get()
        client_id = item.get("client_id")
        if not client_id:
            continue

        try:
            if item.get("_reset_detector") or item.get("_state_override") == "INIT":
                detector.reset()
                dlog(f"[CV] detector reset for {client_id}")
                continue

            jpeg = item.get("jpeg", b"")
            ts = item.get("timestamp_ms")

            if not jpeg or ts is None:
                dlog("[CV] skipped malformed frame")
                continue

            # Defense in depth alongside the transport-level UDP check: never
            # hand an oversized buffer to cv2.imdecode.
            if len(jpeg) > MAX_JPEG_BYTES:
                dlog(
                    f"[CV] skipped oversized frame: {len(jpeg)} bytes "
                    f"> MAX_JPEG_BYTES={MAX_JPEG_BYTES}"
                )
                continue

            frame_counter += 1
            if frame_counter % 30 == 1:
                dlog(
                    f"[CV] frame #{frame_counter} client={client_id} "
                    f"ts={ts} jpeg_bytes={len(jpeg)}"
                )

            response = detector.process_frame(jpeg_bytes=jpeg, timestamp_ms=int(ts))

            if response:
                if frame_counter % 30 == 1 or response.get("type") != "HAND_DATA":
                    dlog(
                        f"[CV] produced response type={response.get('type')} "
                        f"gesture={response.get('gesture')}"
                    )
                _put_result(result_queue, client_id, response)

        except BaseException as exc:
            print("[CV] recovered from frame-processing error:", repr(exc), flush=True)
            traceback.print_exc()
            _put_result(
                result_queue,
                client_id,
                {
                    "type": "WARNING",
                    "message": "CV worker error. Check Python terminal for traceback.",
                },
            )


def _put_result(result_queue: Any, client_id: str, response: dict) -> None:
    if response:
        result_queue.put({"client_id": client_id, "response": response})


def _decode_jpeg(jpeg_bytes: bytes):
    import cv2

    buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _build_hand_data(
    smoothed: Optional[np.ndarray],
    gesture: str,
    confirmed: Optional[str],
    confidence: float = 0.0,
    handedness: str = "",
) -> dict:
    return {
        "type": "HAND_DATA",
        "gesture": gesture or "None",
        "gesture_confirmed": confirmed is not None,
        "landmarks_smoothed": smoothed.tolist() if smoothed is not None else [],
        # Kept for backward-compatible JSON shape. Active Mode 2 does not use it.
        "projected_floor_point": None,
        "confidence": float(confidence or 0.0),
        "handedness": handedness or "",
    }


def _choose_project_gesture(landmarks, mp_category: str) -> str:
    """Return the project gesture Unity expects in HAND_DATA."""
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


def _dist_lm(a, b) -> float:
    return float(((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2) ** 0.5)


def _scale(landmarks) -> float:
    return max(
        _dist_lm(landmarks[0], landmarks[9]),
        _dist_lm(landmarks[5], landmarks[17]),
        _dist_lm(landmarks[0], landmarks[5]),
        1e-6,
    )


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
    hand_span = _scale(landmarks)
    return (
        _ratio(landmarks, 8, 5, hand_span) > POINT_EXTENDED_MIN
        and _ratio(landmarks, 12, 9, hand_span) < POINT_CURLED_MAX
        and _ratio(landmarks, 16, 13, hand_span) < POINT_CURLED_MAX
        and _ratio(landmarks, 20, 17, hand_span) < POINT_CURLED_MAX
    )


def _is_pinch_gesture(landmarks) -> bool:
    hand_span = _scale(landmarks)
    pinch_ratio = _ratio(landmarks, 4, 8, hand_span)
    index_mcp_ratio = _ratio(landmarks, 8, 5, hand_span)

    middle_extended = _finger_extended(landmarks, 9, 10, 12, hand_span)
    ring_extended = _finger_extended(landmarks, 13, 14, 16, hand_span)
    pinky_extended = _finger_extended(landmarks, 17, 18, 20, hand_span)

    return (
        pinch_ratio <= PINCH_MAX_RATIO
        and index_mcp_ratio >= PINCH_INDEX_MIN_RATIO
        and not (middle_extended and ring_extended and pinky_extended)
    )


def _is_open_palm_gesture(landmarks) -> bool:
    hand_span = _scale(landmarks)
    extended_count = sum(
        [
            _finger_extended(landmarks, 5, 6, 8, hand_span),
            _finger_extended(landmarks, 9, 10, 12, hand_span),
            _finger_extended(landmarks, 13, 14, 16, hand_span),
            _finger_extended(landmarks, 17, 18, 20, hand_span),
        ]
    )
    pinch_ratio = _ratio(landmarks, 4, 8, hand_span)
    return extended_count >= 3 and pinch_ratio > PINCH_MAX_RATIO * 1.18


def _is_fist_gesture(landmarks) -> bool:
    hand_span = _scale(landmarks)
    curled_count = sum(
        [
            _finger_curled(landmarks, 5, 8, hand_span),
            _finger_curled(landmarks, 9, 12, hand_span),
            _finger_curled(landmarks, 13, 16, hand_span),
            _finger_curled(landmarks, 17, 20, hand_span),
        ]
    )
    return curled_count >= 3
