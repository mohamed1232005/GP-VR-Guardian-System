"""Body-pose tracking pipeline for the active Mode 2/System A Python service.

This module mirrors ``hand_tracking.py`` but for full-body pose from a laptop
webcam. It is a fully independent pipeline: its own capture thread, queue, and
worker. Python stays stateless CV here too -- it emits only raw, smoothed
landmarks as BODY_POSE; Unity owns all rehab interpretation (form/reps/holds).

The hand pipeline is untouched and cannot be starved: the only shared object is
the generic ``result_queue`` drained by ``transport.TCPHandler.result_dispatcher``.
"""

import json
import os
import queue
import time
import traceback
from typing import Any, Callable, Optional

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from config import (
    dlog,
    POSE_CAMERA_INDEX,
    POSE_CAPTURE_HEIGHT,
    POSE_CAPTURE_WIDTH,
    POSE_DETECT_CONF,
    POSE_EMA_ALPHA,
    POSE_MODEL_FILENAME,
    POSE_PRESENCE_CONF,
    POSE_RECORD_FILENAME,
    POSE_RECORD_RAW,
    POSE_TARGET_FPS,
    POSE_TRACK_CONF,
)

# Reuse the generic landmark smoother and result enqueue helper from the hand
# pipeline -- LandmarkSmoother is shape-agnostic (works for 33 pose joints too).
from hand_tracking import LandmarkSmoother, _put_result


class PoseDetector:
    """MediaPipe Pose landmarks with EMA smoothing. Emits raw landmarks only."""

    def __init__(self, model_path: str = POSE_MODEL_FILENAME):
        model_path = os.path.abspath(model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{POSE_MODEL_FILENAME} not found at: {model_path}")

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=POSE_DETECT_CONF,
            min_pose_presence_confidence=POSE_PRESENCE_CONF,
            min_tracking_confidence=POSE_TRACK_CONF,
        )

        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        self._smoother = LandmarkSmoother(alpha=POSE_EMA_ALPHA)
        self._last_timestamp_ms = -1
        self._last_log_time = 0.0

        print(f"[POSE] detector initialized with model={model_path}", flush=True)

    def process_frame(self, rgb: np.ndarray, timestamp_ms: int) -> dict:
        # MediaPipe VIDEO mode requires strictly increasing timestamps.
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_img, timestamp_ms)
        now = time.monotonic()

        if not result.pose_landmarks:
            if now - self._last_log_time > 2.0:
                dlog("[POSE] no body landmarks detected")
                self._last_log_time = now
            return _build_body_pose(None, None, timestamp_ms)

        raw_lm = result.pose_landmarks[0]
        smoothed_xyz = self._smoother.smooth(raw_lm)
        visibility = np.array(
            [float(getattr(lm, "visibility", 0.0)) for lm in raw_lm], dtype=np.float32
        )

        if now - self._last_log_time > 0.5:
            dlog(f"[POSE] tracked {len(raw_lm)} landmarks ts={timestamp_ms}")
            self._last_log_time = now

        return _build_body_pose(smoothed_xyz, visibility, timestamp_ms)

    def reset(self) -> None:
        print("[POSE] detector reset", flush=True)
        self._smoother.reset()
        self._last_timestamp_ms = -1


def _build_body_pose(
    smoothed_xyz: Optional[np.ndarray],
    visibility: Optional[np.ndarray],
    frame_timestamp_ms: int,
) -> dict:
    """Pack raw pose landmarks as [x, y, z, visibility] rows (33 of them)."""
    if smoothed_xyz is None or visibility is None:
        return {
            "type": "BODY_POSE",
            "landmarks": [],
            "frame_timestamp_ms": int(frame_timestamp_ms),
            "tracked": False,
        }

    rows = [
        [float(smoothed_xyz[i][0]), float(smoothed_xyz[i][1]),
         float(smoothed_xyz[i][2]), float(visibility[i])]
        for i in range(len(smoothed_xyz))
    ]
    return {
        "type": "BODY_POSE",
        "landmarks": rows,
        "frame_timestamp_ms": int(frame_timestamp_ms),
        "tracked": True,
    }


def run_pose_worker(
    pose_frame_queue: Any,
    result_queue: Any,
    client_id_provider: Callable[[], Optional[str]],
) -> None:
    """Read latest webcam frames, run pose tracking, and enqueue BODY_POSE."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    detector = PoseDetector(model_path=os.path.join(base_dir, POSE_MODEL_FILENAME))

    record_file = None
    if POSE_RECORD_RAW:
        record_path = os.path.join(base_dir, POSE_RECORD_FILENAME)
        record_file = open(record_path, "a", encoding="utf-8")
        print(f"[POSE] raw landmark logging enabled -> {record_path}", flush=True)

    frame_counter = 0
    print("[POSE] worker ready", flush=True)

    while True:
        item = pose_frame_queue.get()
        try:
            if item.get("_reset_detector") or item.get("_state_override") == "INIT":
                detector.reset()
                continue

            rgb = item.get("rgb")
            ts = item.get("timestamp_ms")
            if rgb is None or ts is None:
                continue

            client_id = client_id_provider()
            if not client_id:
                continue

            frame_counter += 1
            response = detector.process_frame(rgb, int(ts))

            if response:
                if record_file is not None:
                    record_file.write(json.dumps(response) + "\n")
                    record_file.flush()
                _put_result(result_queue, client_id, response)

        except BaseException as exc:
            print("[POSE] recovered from frame-processing error:", repr(exc), flush=True)
            traceback.print_exc()


def run_pose_capture(
    pose_frame_queue: Any,
    client_id_provider: Callable[[], Optional[str]],
) -> None:
    """Capture laptop webcam frames and push the latest into pose_frame_queue.

    Mirrors UDPFrameReceiver._put_latest (drop-oldest at maxsize=1) so the pose
    worker always sees the freshest frame and latency stays low.
    """
    import cv2

    cap = cv2.VideoCapture(POSE_CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, POSE_CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, POSE_CAPTURE_HEIGHT)

    if not cap.isOpened():
        print(f"[POSE] FATAL: cannot open camera index {POSE_CAMERA_INDEX}", flush=True)
        return

    frame_interval = 1.0 / max(POSE_TARGET_FPS, 1)
    print(f"[POSE] capture started on camera {POSE_CAMERA_INDEX} @ {POSE_TARGET_FPS} fps", flush=True)

    try:
        while True:
            loop_start = time.monotonic()

            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                time.sleep(frame_interval)
                continue

            # Only do work when a Unity client is connected to consume it.
            if client_id_provider() is None:
                time.sleep(frame_interval)
                continue

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            packet = {
                "rgb": rgb,
                "timestamp_ms": int(time.monotonic() * 1000),
            }
            _put_latest_pose(pose_frame_queue, packet)

            elapsed = time.monotonic() - loop_start
            sleep_for = frame_interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)
    finally:
        cap.release()


def _put_latest_pose(pose_frame_queue: Any, packet: dict) -> None:
    """Drop-oldest enqueue, mirroring transport.UDPFrameReceiver._put_latest."""
    try:
        pose_frame_queue.put_nowait(packet)
    except queue.Full:
        try:
            pose_frame_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            pose_frame_queue.put_nowait(packet)
        except queue.Full:
            return
