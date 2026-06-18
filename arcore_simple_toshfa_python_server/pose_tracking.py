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
import threading
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
    POSE_PREVIEW,
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
    if POSE_PREVIEW:
        print("[POSE] debug preview window ENABLED (GUARDIAN_POSE_PREVIEW=1). Press 'q' in the window to close it.", flush=True)

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

            # Run detection when there is a Unity consumer OR the local preview is on,
            # so the debug window shows the skeleton even before Unity connects. Emitting
            # BODY_POSE still requires a connected client (Python stays a passive emitter).
            if not client_id and not POSE_PREVIEW:
                continue

            frame_counter += 1
            response = detector.process_frame(rgb, int(ts))

            if POSE_PREVIEW:
                _render_preview(rgb, response)

            if response and client_id:
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
        # Non-fatal: the server (and hand tracking) stays alive. The supervisor restarts
        # this capture after a short delay, so it self-heals if the webcam appears later.
        # Back off here so a permanently-absent webcam does not spin the supervisor.
        try:
            cap.release()
        except Exception:
            pass
        print(
            f"[POSE] webcam UNAVAILABLE: cannot open camera index {POSE_CAMERA_INDEX}. "
            f"Body tracking will keep retrying; hand tracking is unaffected.",
            flush=True,
        )
        time.sleep(2.0)
        return

    frame_interval = 1.0 / max(POSE_TARGET_FPS, 1)
    print(f"[POSE] capture STARTED on camera index {POSE_CAMERA_INDEX} @ {POSE_TARGET_FPS} fps", flush=True)

    try:
        while True:
            loop_start = time.monotonic()

            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                time.sleep(frame_interval)
                continue

            # Push frames when a Unity client is connected, OR when the local preview is
            # on (so the debug window stays live even before Unity connects).
            if client_id_provider() is None and not POSE_PREVIEW:
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


# --- Laptop-side debug preview (GUARDIAN_POSE_PREVIEW=1) -----------------------
# Two parts so the OpenCV window paints reliably on Windows:
#   1) _render_preview(): runs on the POSE WORKER thread. It draws the live webcam frame
#      + MediaPipe skeleton (from the BODY_POSE response, no 2nd detector) and stores the
#      annotated BGR image in a shared slot. It NEVER calls imshow.
#   2) run_pose_preview_window(): a DEDICATED GUI thread that owns the window and pumps
#      cv2.imshow + cv2.waitKey in a tight loop. A single consistent GUI thread is the
#      reliable cross-platform pattern (imshow from the sporadic worker thread leaves the
#      window blank/unresponsive on Windows). 'q' closes the window; the server keeps
#      serving; any GUI failure self-disables the preview and continues headless.
_PREVIEW_WINDOW = "Toshfa Pose Preview (press q to close)"
_preview_lock = threading.Lock()
_latest_preview_bgr = None          # annotated BGR frame from the worker
_preview_window_closed = False      # set by the GUI thread on 'q' / error
_preview_last_time = 0.0
_preview_fps = 0.0
# BlazePose 33-landmark skeleton edges (subset that reads clearly as a stick figure).
_POSE_EDGES = (
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 31), (24, 26), (26, 28), (28, 32),
    (0, 11), (0, 12),
)


def _render_preview(rgb: np.ndarray, response: Optional[dict]) -> None:
    """WORKER thread: annotate the frame and stash it for the GUI thread. No imshow here."""
    global _latest_preview_bgr, _preview_last_time, _preview_fps
    if _preview_window_closed:
        return
    try:
        import cv2

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]

        tracked = bool(response.get("tracked")) if response else False
        lms = (response.get("landmarks") or []) if response else []

        if tracked and lms:
            for a, b in _POSE_EDGES:
                if a < len(lms) and b < len(lms):
                    cv2.line(
                        bgr,
                        (int(lms[a][0] * w), int(lms[a][1] * h)),
                        (int(lms[b][0] * w), int(lms[b][1] * h)),
                        (0, 255, 0), 2,
                    )
            for lm in lms:
                cv2.circle(bgr, (int(lm[0] * w), int(lm[1] * h)), 3, (0, 200, 255), -1)

        now = time.monotonic()
        if _preview_last_time:
            dt = now - _preview_last_time
            if dt > 0:
                _preview_fps = 0.9 * _preview_fps + 0.1 * (1.0 / dt)
        _preview_last_time = now

        rows = len(lms)
        cols = len(lms[0]) if rows else 0
        label = (f"BODY_POSE tracked={tracked} landmarks={rows}x{cols} "
                 f"cam={POSE_CAMERA_INDEX} fps={_preview_fps:0.0f}")
        cv2.putText(bgr, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(bgr, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        with _preview_lock:
            _latest_preview_bgr = bgr
    except Exception as exc:
        print(f"[POSE] preview render error ({exc!r}); skipping frame.", flush=True)


# The window is opened + pumped on the MAIN thread (server.py drives these from the asyncio
# loop). On Windows, OpenCV HighGUI only paints reliably from the main thread, so a dedicated
# worker thread left the window blank — this is the fix. preview_open() once, preview_tick()
# repeatedly, preview_close() at shutdown.
def preview_open() -> bool:
    """Create the preview window on the CALLING (main) thread. False if GUI is unavailable."""
    global _preview_window_closed
    try:
        import cv2
        cv2.namedWindow(_PREVIEW_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(_PREVIEW_WINDOW, 720, 540)
        try:
            cv2.setWindowProperty(_PREVIEW_WINDOW, cv2.WND_PROP_TOPMOST, 1)  # bring to front
        except Exception:
            pass
        _preview_window_closed = False
        print("[POSE] preview window opened (main thread). Press 'q' to close.", flush=True)
        return True
    except Exception as exc:
        _preview_window_closed = True
        print(f"[POSE] preview window could not open ({exc!r}); continuing headless.", flush=True)
        return False


def preview_tick() -> bool:
    """Show the latest annotated frame once on the CALLING (main) thread. False to stop."""
    global _preview_window_closed
    if _preview_window_closed:
        return False
    try:
        import cv2
        import numpy as _np
        with _preview_lock:
            frame = _latest_preview_bgr
        if frame is None:
            frame = _np.zeros((540, 720, 3), dtype=_np.uint8)
            cv2.putText(frame, "Waiting for webcam frames...", (110, 270),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow(_PREVIEW_WINDOW, frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            preview_close()
            return False
        return True
    except Exception as exc:
        _preview_window_closed = True
        print(f"[POSE] preview error ({exc!r}); continuing headless.", flush=True)
        return False


def preview_close() -> None:
    global _preview_window_closed
    _preview_window_closed = True
    try:
        import cv2
        cv2.destroyWindow(_PREVIEW_WINDOW)
    except Exception:
        pass
    print("[POSE] preview window closed; server keeps running.", flush=True)
