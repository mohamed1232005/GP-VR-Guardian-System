# ===========================================================================
# models/ai_pipeline_worker.py — Phase 7A
# Runs depth + segmentation + geometry + hand tracking in a background thread.
#
# Phase 7A changes:
#   - MediaPipe hand tracking runs on every frame (separate FPS throttle).
#   - Hand ray data (AI_HAND_DATA) sent to Unity for ray interaction.
#   - PLACEMENT_MODE="user_confirm" changes auto-lock to ready_to_confirm.
#     Python no longer forces final lock; Unity waits for user button.
#   - Phase 7B: Room modes removed. 3 floor candidates from real data.
#   - All Phase 5/6 pipeline stages (depth, seg, RANSAC, boundary) intact.
# ===========================================================================

import threading
import time
import os
import csv
import numpy as np
import json

from config import (
    ALLOW_FALLBACK_LOCK, CANDIDATE_POOL_SIZE,
    MAX_CONSECUTIVE_INVALID_BEFORE_POOL_RESET,
    WRITE_DEBUG_CSV, SAVE_DEBUG_IMAGES, DEBUG_IMAGE_EVERY_N,
    INPUT_ROTATION_DEGREES, USE_FALLBACK_FLOOR_MASK,
    # Phase 7A
    HAND_TRACKING_ENABLED, HAND_TRACKING_TARGET_FPS,
    HAND_MIN_DETECTION_CONF, HAND_MIN_TRACKING_CONF, HAND_MAX_NUM_HANDS,
    PLACEMENT_MODE,
    # Phase 9: Post-lock optimization
    POST_LOCK_ENABLED, POST_LOCK_DIAGNOSTIC_FPS,
    POST_LOCK_STOP_DEPTH, POST_LOCK_STOP_SEGMENTATION,
    POST_LOCK_STOP_PLANE_FIT, POST_LOCK_KEEP_HAND,
)

from models.depth_model       import DepthModelBase, create_depth_model
from models.floor_segmenter   import FloorSegmenter
from geometry.backproject     import backproject_floor_points
from geometry.plane_fit       import fit_floor_plane
from geometry.boundary_generator import generate_boundary
from geometry.rotation_utils  import rotate_frame_and_intrinsics


# ---------------------------------------------------------------------------
# Composite score for ranking candidates before locking.
# Higher = better candidate.
# ---------------------------------------------------------------------------
def _score_candidate(horizontal_score, inlier_ratio, confidence, rmse, floor_pixel_ratio):
    rmse_penalty = max(0.0, rmse / 0.10)
    score = (
        0.30 * horizontal_score
        + 0.25 * inlier_ratio
        + 0.20 * confidence
        + 0.15 * min(1.0, floor_pixel_ratio / 0.10)
        - 0.10 * rmse_penalty
    )
    return float(score)


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


class AIPipelineWorker:
    """
    Combined AI pipeline worker for Phase 5.

    Pipeline per frame:
        1. Depth estimation (Depth Anything V2)
        2. Floor segmentation (SegFormer ± geometric fallback)
        3. Backproject floor pixels → 3D camera-space points
        4. RANSAC + LSQ floor plane fit
        5. Normal conversion test (CV → Unity cam → world) × 2 matrix orders
        6. Horizontal-score gate (loose scan / strict lock)
        7. Boundary generation from RANSAC inlier extents
        8. Candidate pool → lock best after CANDIDATE_POOL_SIZE
        9. Push result to TCP server
    """

    def __init__(
        self,
        # Depth
        depth_model_name: str = "depth_anything_v2_small",
        depth_allow_dummy: bool = False,
        depth_use_metric_indoor: bool = True,
        depth_try_depth_pro: bool = False,
        # Segmentation
        seg_model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512",
        # Pipeline
        target_fps: float = 1.0,
        # Geometry
        min_depth: float = 0.1,
        max_depth: float = 10.0,
        max_points: int = 5000,
        ransac_iterations: int = 200,
        ransac_threshold: float = 0.05,
        plane_min_inliers: int = 500,
        plane_min_ratio: float = 0.45,
        plane_max_rmse: float = 0.08,
        # Boundary
        boundary_width: float = 2.0,
        boundary_depth: float = 2.0,
        boundary_near_z: float = 1.0,
        boundary_far_z: float = 3.0,
        boundary_safety_margin: float = 0.15,
        floor_min_ratio: float = 0.005,
        # Scan gate
        lower_image_roi: float = 0.45,
        plane_min_horizontal_score: float = 0.75,
        # Lock gate
        lock_min_horizontal_score: float = 0.90,
        lock_min_floor_ratio: float = 0.05,
        lock_min_inlier_ratio: float = 0.75,
        lock_max_rmse: float = 0.03,
        lock_min_confidence: float = 0.80,
        # Legacy stability (kept for server.py compatibility)
        stable_required: int = 5,
        stable_max_angle_diff: float = 10.0,
        stable_max_drift_m: float = 0.15,
        stable_min_confidence: float = 0.75,
        stable_max_rmse: float = 0.04,
        stable_min_inlier_ratio: float = 0.70,
    ):
        # Depth
        self._depth_model_name        = depth_model_name
        self._depth_allow_dummy       = depth_allow_dummy
        self._depth_use_metric_indoor = depth_use_metric_indoor
        self._depth_try_depth_pro     = depth_try_depth_pro

        # Segmentation
        self._seg_model_name = seg_model_name

        # Pipeline
        self._target_fps   = target_fps
        self._min_interval = 1.0 / max(0.01, target_fps)

        # Geometry
        self._min_depth         = min_depth
        self._max_depth         = max_depth
        self._max_points        = max_points
        self._ransac_iterations = ransac_iterations
        self._ransac_threshold  = ransac_threshold
        self._plane_min_inliers = plane_min_inliers
        self._plane_min_ratio   = plane_min_ratio
        self._plane_max_rmse    = plane_max_rmse

        # Boundary
        self._boundary_width         = boundary_width
        self._boundary_depth         = boundary_depth
        self._boundary_near_z        = boundary_near_z
        self._boundary_far_z         = boundary_far_z
        self._boundary_safety_margin = boundary_safety_margin
        self._floor_min_ratio        = floor_min_ratio

        # Scan gate
        self._lower_image_roi            = lower_image_roi
        self._plane_min_horizontal_score = plane_min_horizontal_score

        # Lock gate
        self._lock_min_horizontal_score = lock_min_horizontal_score
        self._lock_min_floor_ratio      = lock_min_floor_ratio
        self._lock_min_inlier_ratio     = lock_min_inlier_ratio
        self._lock_max_rmse             = lock_max_rmse
        self._lock_min_confidence       = lock_min_confidence

        # Legacy stability
        self._stable_required         = stable_required
        self._stable_max_angle_diff   = stable_max_angle_diff
        self._stable_max_drift_m      = stable_max_drift_m
        self._stable_min_confidence   = stable_min_confidence
        self._stable_max_rmse         = stable_max_rmse
        self._stable_min_inlier_ratio = stable_min_inlier_ratio

        # Fallback lock policy (from config)
        self._allow_fallback_lock     = ALLOW_FALLBACK_LOCK

        # State
        self._boundary_state       = "scanning"
        self._locked_boundary_msg  = None
        self._candidate_pool       = []
        self._candidate_pool_size  = CANDIDATE_POOL_SIZE
        self._consecutive_invalid  = 0
        self._max_consecutive_invalid = MAX_CONSECUTIVE_INVALID_BEFORE_POOL_RESET

        # Phase 7B: placement mode (no room modes)
        self._placement_mode   = PLACEMENT_MODE      # "user_confirm" or "auto_lock"
        self._floor_confirmed  = False   # True when candidate pool is full

        # Phase 7A: hand tracking state
        self._hand_tracking_enabled = HAND_TRACKING_ENABLED
        self._hand_target_fps      = HAND_TRACKING_TARGET_FPS
        self._hand_min_interval    = 1.0 / max(0.01, HAND_TRACKING_TARGET_FPS)
        self._last_hand_time       = 0.0
        self._last_depth_map       = None   # cached for hand 3D backprojection
        self._last_rotated_frame   = None   # cached rotated frame for hand tracking
        self._last_fx = 250.0
        self._last_fy = 250.0
        self._last_cx = 160.0
        self._last_cy = 120.0

        # Phase 9: Post-lock optimization
        self._post_lock_mode       = False
        self._post_lock_enabled    = POST_LOCK_ENABLED
        self._post_lock_diag_fps   = POST_LOCK_DIAGNOSTIC_FPS
        self._post_lock_diag_interval = 1.0 / max(0.001, POST_LOCK_DIAGNOSTIC_FPS)
        self._last_diag_time       = 0.0

        # Phase 9: Intrinsics version tracking (cache lifecycle fix)
        self._intrinsics_version   = 0
        self._cached_intrinsics_version = -1

        # Phase 9.1: Proper timestamp-window-based metrics
        self._metrics = {
            "mode": "warm",
            "start_time": time.time(),

            # WARM phase (pre-lock): depth+seg+plane ON
            "warm_start_time": time.time(),
            "warm_end_time": None,
            "warm_frames": 0,
            "warm_depth_ms_total": 0.0,
            "warm_seg_ms_total": 0.0,
            "warm_plane_ms_total": 0.0,
            "warm_hand_ms_total": 0.0,
            "warm_hand_frames": 0,
            "warm_pipeline_ms_total": 0.0,
            "warm_tcp_guardian_sends": 0,
            "warm_tcp_hand_sends": 0,

            # IDLE phase (post-lock): depth+seg OFF, hand ON
            "idle_start_time": None,
            "idle_frames": 0,
            "idle_hand_ms_total": 0.0,
            "idle_hand_frames": 0,
            "idle_tcp_hand_sends": 0,

            # Global
            "lock_frame_id": None,
            "lock_time": None,
            "hand_frames_total": 0,
            "udp_input_frames": 0,
            "dropped_frames": 0,
            "errors": 0,
            "warnings": 0,

            # Rolling window (last N samples for live FPS)
            "_warm_timestamps": [],
            "_idle_timestamps": [],
            "_hand_timestamps": [],
            "_udp_timestamps": [],   # Phase 9.4: UDP input frame timestamps
        }

        # Models
        self._depth_model:    DepthModelBase = None
        self._floor_segmenter: FloorSegmenter = None
        self._hand_tracker = None

        # Frame input (latest-only, thread-safe)
        self._input_lock       = threading.Lock()
        self._pending_frame    = None
        self._pending_metadata = None
        self._pending_ready    = False

        # Result output
        self._output_lock   = threading.Lock()
        self._latest_result = None

        # TCP server
        self._tcp_server = None

        # Thread
        self._thread  = None
        self._running = False
        self._event   = threading.Event()

        # Stats
        self._frame_count = 0
        self._fps_count   = 0
        self._fps_start   = time.time()

        # CSV logging
        self._write_csv  = WRITE_DEBUG_CSV
        self._csv_file   = None
        self._csv_writer = None
        if self._write_csv:
            os.makedirs("debug", exist_ok=True)
            self._csv_file   = open("debug/session_log.csv", "w", newline="")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow([
                "frame_id", "seg_mode", "used_fallback",
                "semantic_ratio", "semantic_confidence",
                "roi_ratio", "depth_mean", "backproject_points",
                "plane_rmse", "inlier_ratio",
                "score_col", "score_row", "selected_matrix_order",
                "horizontal_score", "input_rotation_degrees",
                "boundary_state", "lock_allowed",
                "reject_stage", "reject_reason", "total_ms",
            ])
            self._csv_file.flush()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_tcp_server(self, tcp_server):
        self._tcp_server = tcp_server

    def start(self):
        print(f"[P5_PIPELINE] starting (target_fps={self._target_fps})")

        self._depth_model = create_depth_model(
            model_name=self._depth_model_name,
            allow_dummy=self._depth_allow_dummy,
            use_metric_indoor=self._depth_use_metric_indoor,
            try_depth_pro=self._depth_try_depth_pro,
        )
        if not self._depth_model.is_real:
            print("[P5_PIPELINE_WARN] depth model is NOT real")

        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self._floor_segmenter = FloorSegmenter(
            model_id=self._seg_model_name,
            device=device,
            save_debug=SAVE_DEBUG_IMAGES,
            debug_every_n=DEBUG_IMAGE_EVERY_N,
        )

        # Phase 7B: Hand tracker (with robust error handling)
        if self._hand_tracking_enabled:
            try:
                from models.hand_tracker import HandTracker
                print("[P7_HAND_INIT] HandTracker import OK")
                self._hand_tracker = HandTracker(
                    min_detection_confidence=HAND_MIN_DETECTION_CONF,
                    min_tracking_confidence=HAND_MIN_TRACKING_CONF,
                    max_num_hands=HAND_MAX_NUM_HANDS,
                )
                if self._hand_tracker.is_available:
                    print(f"[P7_HAND_INIT] ok api=MediaPipe "
                          f"target_fps={self._hand_target_fps}")
                    print(f"[P7_PIPELINE] hand tracking available")
                else:
                    print("[P7_HAND_INIT] FAILED — hand tracking NOT available")
                    print("[P7_PIPELINE] hand tracking NOT available")
                    self._hand_tracker = None
            except Exception as e:
                import traceback
                print(f"[P7_HAND_INIT] CRASH during import/init: {e}")
                traceback.print_exc()
                self._hand_tracker = None
                print("[P7_PIPELINE] hand tracking DISABLED due to import error")

        self._running   = True
        self._fps_start = time.time()
        self._fps_count = 0

        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

        print(f"[P7_PIPELINE] started depth={self._depth_model.model_name} "
              f"seg={self._seg_model_name} real={self._depth_model.is_real} "
              f"placement_mode={self._placement_mode}")

    def stop(self):
        self._running = False
        self._event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        if self._csv_file:
            self._csv_file.close()
        if self._hand_tracker is not None:
            self._hand_tracker.close()
        print(f"[P7_PIPELINE] stopped. frames={self._frame_count}")

    def push_frame(self, frame: np.ndarray, metadata: dict):
        with self._input_lock:
            self._pending_frame    = frame
            self._pending_metadata = metadata
            self._pending_ready    = True
        self._event.set()
        # Phase 9.4: Track UDP input timestamps
        self._metrics["udp_input_frames"] += 1
        self._metrics["_udp_timestamps"].append(time.time())

    def get_latest_result(self) -> dict:
        with self._output_lock:
            return self._latest_result

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------

    def _worker_loop(self):
        while self._running:
            self._event.wait(timeout=self._min_interval)
            self._event.clear()
            if not self._running:
                break

            frame = metadata = None
            with self._input_lock:
                if self._pending_ready:
                    frame    = self._pending_frame
                    metadata = self._pending_metadata
                    self._pending_ready = False

            if frame is None:
                continue

            frame_id = metadata.get("frame_id", "?") if metadata else "?"
            t_loop_start = time.time()

            # ============================================================
            # Phase 9: Post-lock mode — skip expensive processing
            # ============================================================
            if self._post_lock_mode and self._post_lock_enabled:
                # Only run hand tracking in post-lock mode
                if POST_LOCK_KEEP_HAND:
                    self._cache_frame_for_hand(frame, metadata, frame_id)
                    now = time.time()
                    if (self._hand_tracker is not None
                            and now - self._last_hand_time >= self._hand_min_interval
                            and self._last_rotated_frame is not None):
                        self._last_hand_time = now
                        t_hand = time.time()
                        try:
                            self._run_hand_tracking(self._last_rotated_frame, frame_id)
                        except Exception as e:
                            import traceback
                            print(f"[P9_HAND_ERR] frame={frame_id}: {e}")
                            traceback.print_exc()
                        hand_ms = (time.time() - t_hand) * 1000
                        self._metrics["idle_hand_ms_total"] += hand_ms
                        self._metrics["idle_hand_frames"] += 1
                        self._metrics["hand_frames_total"] += 1
                        self._metrics["idle_tcp_hand_sends"] += 1
                        self._metrics["_hand_timestamps"].append(time.time())

                self._metrics["idle_frames"] += 1
                self._metrics["_idle_timestamps"].append(time.time())
                self._metrics["mode"] = "post_lock_idle"

                # Optional: diagnostic pipeline run at very low FPS
                now = time.time()
                if now - self._last_diag_time >= self._post_lock_diag_interval:
                    self._last_diag_time = now
                    print(f"[P9_DIAG] frame={frame_id} post_lock_diagnostic")

                # Phase 9.4: Clean FPS log that complements Unity's [U9_FPS]
                if self._metrics["idle_frames"] % 50 == 1:
                    avg_hand = 0.0
                    if self._metrics["idle_hand_frames"] > 0:
                        avg_hand = self._metrics["idle_hand_ms_total"] / self._metrics["idle_hand_frames"]
                    idle_fps = self._compute_rolling_fps(self._metrics["_idle_timestamps"])
                    hand_fps = self._compute_rolling_fps(self._metrics["_hand_timestamps"])
                    udp_fps = self._compute_rolling_fps(self._metrics["_udp_timestamps"])
                    print(f"[P9_FPS] mode=idle udp_input_fps={udp_fps:.1f} "
                          f"hand_hz={hand_fps:.1f} hand_ms={avg_hand:.1f} "
                          f"idle_fps={idle_fps:.1f}")

                # Phase 9.2: CRITICAL FIX — use HAND interval, not guardian interval
                # The guardian interval is 1.0s (1 FPS), which throttled hand to 1 Hz.
                # Hand tracking must run at HAND_TARGET_FPS even after lock.
                elapsed = time.time() - t_loop_start
                sleep_rem = self._hand_min_interval - elapsed
                if sleep_rem > 0:
                    time.sleep(sleep_rem)
                continue

            # ============================================================
            # Normal mode: Full pipeline
            # ============================================================

            # Phase 7C: ALWAYS cache rotated frame + depth for hand tracking
            self._cache_frame_for_hand(frame, metadata, frame_id)

            # Phase 7C: Hand tracking runs FIRST, independently of guardian
            now = time.time()
            if (self._hand_tracker is not None
                    and now - self._last_hand_time >= self._hand_min_interval
                    and self._last_rotated_frame is not None):
                self._last_hand_time = now
                t_hand = time.time()
                try:
                    self._run_hand_tracking(self._last_rotated_frame, frame_id)
                except Exception as e:
                    import traceback
                    print(f"[P7_HAND_ERR] frame={frame_id}: {e}")
                    traceback.print_exc()
                hand_ms = (time.time() - t_hand) * 1000
                self._metrics["warm_hand_ms_total"] += hand_ms
                self._metrics["warm_hand_frames"] += 1
                self._metrics["hand_frames_total"] += 1
                self._metrics["warm_tcp_hand_sends"] += 1
                self._metrics["_hand_timestamps"].append(time.time())

            # Guardian floor pipeline
            try:
                result = self._run_pipeline(frame, metadata, frame_id)
                with self._output_lock:
                    self._latest_result = result
                if self._tcp_server is not None:
                    self._tcp_server.push_ai_result(result)
                    rtype = result.get("type", "?")
                    rstate = result.get("boundary_state", "?")
                    print(f"[P7_TCP_SEND] type={rtype} frame={frame_id} state={rstate}")

                    # Phase 9: Detect lock transition
                    if rstate == "locked" and not self._post_lock_mode:
                        self._post_lock_mode = True
                        self._metrics["lock_frame_id"] = frame_id
                        self._metrics["lock_time"] = time.time()
                        self._metrics["idle_start_time"] = time.time()
                        self._metrics["warm_end_time"] = time.time()
                        print(f"[P9_MODE] guardian locked, depth=OFF seg=OFF plane=OFF hand=ON")
                        print(f"[P9_MODE] post-lock mode activated at frame={frame_id}")

                self._frame_count += 1
                self._fps_count   += 1
                self._metrics["warm_frames"] += 1
                self._metrics["warm_tcp_guardian_sends"] += 1
                self._metrics["_warm_timestamps"].append(time.time())

                # Phase 9.1: Accumulate per-stage timing from pipeline result
                pipeline_ms = (time.time() - t_loop_start) * 1000
                self._metrics["warm_pipeline_ms_total"] += pipeline_ms
                dbg_data = result.get("debug", {})
                if isinstance(dbg_data, dict):
                    self._metrics["warm_depth_ms_total"] += dbg_data.get("depth_ms", 0)
                    self._metrics["warm_seg_ms_total"] += dbg_data.get("seg_ms", 0)
                    self._metrics["warm_plane_ms_total"] += dbg_data.get("plane_ms", 0)

                # Phase 9.4: Clean FPS log that complements Unity's [U9_FPS]
                if self._metrics["warm_frames"] % 30 == 1:
                    warm_fps = self._compute_rolling_fps(self._metrics["_warm_timestamps"])
                    hand_fps = self._compute_rolling_fps(self._metrics["_hand_timestamps"])
                    udp_fps = self._compute_rolling_fps(self._metrics["_udp_timestamps"])
                    avg_depth = (self._metrics["warm_depth_ms_total"] / max(1, self._metrics["warm_frames"]))
                    avg_seg = (self._metrics["warm_seg_ms_total"] / max(1, self._metrics["warm_frames"]))
                    avg_plane = (self._metrics["warm_plane_ms_total"] / max(1, self._metrics["warm_frames"]))
                    tcp_guardian_hz = self._metrics["warm_tcp_guardian_sends"] / max(0.1, time.time() - self._metrics["warm_start_time"])
                    print(f"[P9_FPS] mode=warm udp_input_fps={udp_fps:.1f} "
                          f"pipeline_fps={warm_fps:.1f} hand_hz={hand_fps:.1f} "
                          f"tcp_send_hz={tcp_guardian_hz:.1f}")
                    print(f"[P9_TIMING] depth_ms={avg_depth:.1f} "
                          f"seg_ms={avg_seg:.1f} plane_ms={avg_plane:.1f}")

            except Exception as e:
                import traceback
                print(f"[P7_PIPELINE_ERR] frame={frame_id}: {e}")
                traceback.print_exc()
                self._send_scanning_status(frame_id)
                self._metrics["errors"] += 1

            elapsed   = time.time() - t_loop_start
            sleep_rem = self._min_interval - elapsed
            if sleep_rem > 0:
                time.sleep(sleep_rem)

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def _run_pipeline(self, frame, metadata, frame_id) -> dict:
        t_total_start = time.time()

        # --- Read pose from metadata ---
        pose      = (metadata or {}).get("camera_to_world") or (metadata or {}).get("pose")
        has_pose  = (
            pose is not None and
            isinstance(pose, list) and
            len(pose) == 16
        )

        # --- Phase 7C: Reuse cached rotated frame + depth from _cache_frame_for_hand ---
        # _cache_frame_for_hand already ran rotation + depth before this method.
        rotation_deg = INPUT_ROTATION_DEGREES
        frame     = self._last_rotated_frame  # Already rotated
        depth_map = self._last_depth_map      # Already computed
        fx = self._last_fx
        fy = self._last_fy
        cx = self._last_cx
        cy = self._last_cy
        h, w = frame.shape[:2]

        print(f"[P5_ROTATE] id={frame_id} rot={rotation_deg} "
              f"new={w}x{h} fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")

        # --- Debug dict (populated throughout pipeline) ---
        dbg = {
            "frame_id": frame_id,
            "timestamp_ms": int(time.time() * 1000),
            "image_width": w, "image_height": h,
            "has_pose": has_pose,
            "fx": round(fx, 2), "fy": round(fy, 2),
            "cx": round(cx, 2), "cy": round(cy, 2),
            "input_rotation_degrees": rotation_deg,
            # Depth
            "depth_min": 0, "depth_max": 0, "depth_mean": 0, "depth_ms": 0,
            # Segmentation
            "seg_mode": "semantic", "seg_ms": 0, "used_fallback": False,
            "semantic_floor_pixels": 0, "semantic_floor_ratio": 0,
            "semantic_confidence": 0,
            "floor_pixels_after_roi": 0, "floor_ratio_after_roi": 0,
            # Backprojection
            "backproject_points": 0,
            # Plane
            "plane_valid": False, "plane_ms": 0,
            "plane_normal_camera": None,
            "normal_cv": None, "normal_unity_cam": None, "plane_normal_world": None,
            "plane_d": 0, "plane_rmse": 0, "plane_inlier_ratio": 0,
            # Matrix order test
            "horizontal_score": 0, "score_col": 0, "score_row": 0,
            "selected_matrix_order": "",
            # Boundary
            "boundary_source": "", "boundary_width": 0, "boundary_depth": 0,
            # World check
            "boundary_world_y_avg": 0, "boundary_world_y_min": 0, "boundary_world_y_max": 0,
            # State / lock
            "boundary_state": "scanning",
            "lock_allowed": False, "lock_block_reason": "",
            "candidate_score": 0,
            "candidate_pool_count": 0,
            "candidate_pool_required": self._candidate_pool_size,
            # Rejection
            "reject_stage": "", "reject_reason": "",
            # Timing
            "total_ms": 0,
        }

        print(f"[P5_FRAME] id={frame_id} size={w}x{h} "
              f"pose={has_pose} rot={rotation_deg} "
              f"fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")

        # ---- Step 1: Depth (reuse cached) ----
        d_min, d_max, d_mean = float(np.min(depth_map)), float(np.max(depth_map)), float(np.mean(depth_map))
        dbg["depth_min"]  = round(d_min, 3)
        dbg["depth_max"]  = round(d_max, 3)
        dbg["depth_mean"] = round(d_mean, 3)
        dbg["depth_ms"]   = 0  # Already timed in _cache_frame_for_hand
        print(f"[P5_DEPTH] id={frame_id} min={d_min:.2f} max={d_max:.2f} "
              f"mean={d_mean:.2f} ms=0.0")

        # ---- Step 2: Floor segmentation ----
        t0              = time.time()
        floor_mask, seg_stats = self._floor_segmenter.segment(frame, frame_id=frame_id)
        t_seg           = (time.time() - t0) * 1000
        dbg["seg_ms"]   = round(t_seg, 1)

        # Handle early rejection (USE_FALLBACK_FLOOR_MASK=False path)
        if seg_stats.get("rejected", False):
            reason = seg_stats.get("reject_reason", "semantic_floor_too_low")
            dbg["seg_mode"]              = "semantic"
            dbg["semantic_floor_pixels"] = seg_stats["floor_pixels"]
            dbg["semantic_floor_ratio"]  = round(seg_stats["ratio"], 4)
            dbg["semantic_confidence"]   = round(seg_stats["confidence"], 4)
            dbg["reject_stage"]          = "seg"
            dbg["reject_reason"]         = reason
            print(f"[P5_REJECT] id={frame_id} stage=seg reason={reason}")
            return self._handle_invalid_floor(frame_id, reason, dbg)

        floor_pixels  = seg_stats["floor_pixels"]
        floor_ratio   = seg_stats["ratio"]
        floor_conf    = seg_stats["confidence"]
        used_fallback = seg_stats.get("used_fallback", False)
        seg_mode      = "fallback" if used_fallback else "semantic"

        dbg["seg_mode"]              = seg_mode
        dbg["used_fallback"]         = used_fallback
        dbg["semantic_floor_pixels"] = floor_pixels
        dbg["semantic_floor_ratio"]  = round(floor_ratio, 4)
        dbg["semantic_confidence"]   = round(floor_conf, 4)

        print(f"[P5_SEG] id={frame_id} mode={seg_mode} "
              f"pixels={floor_pixels} ratio={floor_ratio:.4f} conf={floor_conf:.3f}")

        # Loose scan gate
        if floor_ratio < self._floor_min_ratio:
            dbg["reject_stage"]  = "seg"
            dbg["reject_reason"] = "insufficient_floor_pixels"
            print(f"[P5_REJECT] id={frame_id} stage=seg reason=insufficient_floor_pixels")
            return self._handle_invalid_floor(frame_id, "insufficient floor pixels", dbg)

        # ---- Apply lower-image ROI ----
        roi_start_y = int(h * self._lower_image_roi)
        pixels_before = np.count_nonzero(floor_mask)
        floor_mask[:roi_start_y, :] = False
        pixels_after  = np.count_nonzero(floor_mask)
        ratio_after   = pixels_after / (h * w) if (h * w) > 0 else 0.0

        dbg["floor_pixels_after_roi"] = pixels_after
        dbg["floor_ratio_after_roi"]  = round(ratio_after, 4)
        print(f"[P5_ROI] id={frame_id} before={pixels_before} "
              f"after={pixels_after} ratio_after={ratio_after:.4f}")

        if ratio_after < self._floor_min_ratio:
            dbg["reject_stage"]  = "roi"
            dbg["reject_reason"] = "insufficient_floor_after_roi"
            print(f"[P5_REJECT] id={frame_id} stage=roi reason=insufficient_floor_after_roi")
            return self._handle_invalid_floor(frame_id, "insufficient floor after ROI", dbg)

        # ---- Step 3: Backproject floor pixels → 3D ----
        t0 = time.time()
        points_3d, bp_stats = backproject_floor_points(
            depth_map=depth_map, floor_mask=floor_mask,
            fx=fx, fy=fy, cx=cx, cy=cy,
            min_depth=self._min_depth, max_depth=self._max_depth,
            max_points=self._max_points, frame_id=frame_id,
        )
        t_bp = (time.time() - t0) * 1000
        dbg["backproject_points"] = len(points_3d)
        print(f"[P5_BACKPROJECT] id={frame_id} points={len(points_3d)} "
              f"z_min={bp_stats.get('depth_min',0):.2f} "
              f"z_max={bp_stats.get('depth_max',0):.2f}")

        if len(points_3d) < 3:
            dbg["reject_stage"]  = "backproject"
            dbg["reject_reason"] = "insufficient_3d_points"
            print(f"[P5_REJECT] id={frame_id} stage=backproject "
                  f"reason=insufficient_3d_points")
            return self._handle_invalid_floor(frame_id, "insufficient 3D points", dbg)

        # ---- Step 4: RANSAC + LSQ plane fit ----
        t0 = time.time()
        plane, plane_stats = fit_floor_plane(
            points=points_3d,
            ransac_iterations=self._ransac_iterations,
            inlier_threshold=self._ransac_threshold,
            min_inliers=self._plane_min_inliers,
            min_inlier_ratio=self._plane_min_ratio,
            max_rmse=self._plane_max_rmse,
            frame_id=frame_id,
        )
        t_plane = (time.time() - t0) * 1000
        dbg["plane_ms"] = round(t_plane, 1)

        if plane is None or not plane_stats.get("valid", False):
            reason = plane_stats.get("reason", "unknown")
            dbg["reject_stage"]  = "plane"
            dbg["reject_reason"] = f"plane_fit_failed_{reason}"
            print(f"[P5_REJECT] id={frame_id} stage=plane "
                  f"reason=plane_fit_failed ({reason})")
            return self._handle_invalid_floor(frame_id, f"plane fit failed ({reason})", dbg)

        inlier_points   = plane_stats.get("inlier_points", None)
        normal_cam_list = plane["normal"]
        dbg["plane_valid"]        = True
        dbg["plane_normal_camera"] = [round(v, 4) for v in normal_cam_list]
        dbg["plane_d"]            = round(plane["d"], 4)
        dbg["plane_rmse"]         = round(plane["rmse"], 4)
        dbg["plane_inlier_ratio"] = round(plane["inlier_ratio"], 4)
        print(f"[P5_PLANE] id={frame_id} valid=True "
              f"rmse={plane['rmse']:.4f} ir={plane['inlier_ratio']:.2f} "
              f"normal_cam=({normal_cam_list[0]:.3f},"
              f"{normal_cam_list[1]:.3f},{normal_cam_list[2]:.3f})")

        # ---- Step 4.5: Pose / horizontal-score check ----
        horizontal_score = 0.0
        normal_world     = None
        M                = None
        selected_order   = "none"

        if not has_pose:
            dbg["reject_stage"]  = "pose"
            dbg["reject_reason"] = "no_camera_to_world_pose"
            print(f"[P5_REJECT] id={frame_id} stage=pose reason=no_pose")
            return self._handle_invalid_floor(frame_id, "no camera_to_world pose", dbg)

        # ---- Normal conversion: OpenCV cam → Unity cam space ----
        normal_cv       = np.array(plane["normal"], dtype=np.float64)
        # OpenCV: X=right, Y=down, Z=forward
        # Unity:  X=right, Y=up,   Z=backward  → flip Y and Z
        normal_unity_cam = np.array([normal_cv[0], -normal_cv[1], -normal_cv[2]], dtype=np.float64)

        print(f"[P5_NORMAL_TEST] id={frame_id} "
              f"normal_cv=({normal_cv[0]:.4f},{normal_cv[1]:.4f},{normal_cv[2]:.4f})")
        print(f"[P5_NORMAL_TEST] id={frame_id} "
              f"normal_unity_cam=({normal_unity_cam[0]:.4f},"
              f"{normal_unity_cam[1]:.4f},{normal_unity_cam[2]:.4f})")
        dbg["normal_cv"]       = [round(v, 4) for v in normal_cv.tolist()]
        dbg["normal_unity_cam"] = [round(v, 4) for v in normal_unity_cam.tolist()]

        # ---- Test BOTH matrix reshape orders ----
        # Unity Matrix4x4 is column-major in memory; Python default is row-major.
        # Testing both lets us diagnose the correct convention from session_log.csv.
        pose_arr = np.array(pose, dtype=np.float64)
        M_col    = pose_arr.reshape(4, 4, order='F')  # column-major (Fortran order)
        M_row    = pose_arr.reshape(4, 4, order='C')  # row-major (C order)

        R_col = M_col[:3, :3]
        R_row = M_row[:3, :3]

        normal_world_col = _normalize(R_col @ normal_unity_cam)
        normal_world_row = _normalize(R_row @ normal_unity_cam)

        # horizontal score = |dot(normal_world, Unity_up)|
        # Unity up = (0, 1, 0) in world space
        score_col = float(abs(normal_world_col[1]))
        score_row = float(abs(normal_world_row[1]))

        # Camera up direction in world space (diagnostic)
        cam_up_col = _normalize(R_col @ np.array([0.0, 1.0, 0.0]))
        cam_up_row = _normalize(R_row @ np.array([0.0, 1.0, 0.0]))

        # Select the matrix order that gives the higher horizontal score
        if score_col >= score_row:
            selected_order   = "col"
            M                = M_col
            normal_world     = normal_world_col
            horizontal_score = score_col
        else:
            selected_order   = "row"
            M                = M_row
            normal_world     = normal_world_row
            horizontal_score = score_row

        print(f"[P5_MATRIX_TEST] id={frame_id} "
              f"score_col={score_col:.4f} score_row={score_row:.4f} "
              f"selected={selected_order}")
        print(f"[P5_NORMAL_TEST] id={frame_id} "
              f"normal_world=({normal_world[0]:.4f},"
              f"{normal_world[1]:.4f},{normal_world[2]:.4f})")
        print(f"[P5_NORMAL_TEST] id={frame_id} "
              f"cam_up_world=({cam_up_col[0]:.4f},"
              f"{cam_up_col[1]:.4f},{cam_up_col[2]:.4f}) [col order]")
        print(f"[P5_NORMAL_TEST] id={frame_id} horizontal_score={horizontal_score:.4f}")

        dbg["horizontal_score"]      = round(horizontal_score, 4)
        dbg["score_col"]             = round(score_col, 4)
        dbg["score_row"]             = round(score_row, 4)
        dbg["selected_matrix_order"] = selected_order
        dbg["plane_normal_world"]    = [round(v, 4) for v in normal_world.tolist()]

        # ---- Scan gate: reject clearly non-horizontal planes ----
        if horizontal_score < self._plane_min_horizontal_score:
            dbg["reject_stage"]  = "plane"
            dbg["reject_reason"] = (
                f"not_horizontal score={horizontal_score:.3f} "
                f"(col={score_col:.3f} row={score_row:.3f})"
            )
            print(f"[P5_REJECT] id={frame_id} stage=plane "
                  f"reason=not_horizontal score={horizontal_score:.3f} "
                  f"min={self._plane_min_horizontal_score:.2f}")
            return self._handle_invalid_floor(frame_id, "plane not horizontal", dbg)

        # ---- Step 5: Generate boundary ----
        ref_pts        = inlier_points if inlier_points is not None else points_3d
        floor_center_z = float(np.median(ref_pts[:, 2]))
        floor_center_x = float(np.median(ref_pts[:, 0]))
        actual_near_z  = max(0.3, floor_center_z - self._boundary_depth / 2.0)
        actual_far_z   = floor_center_z + self._boundary_depth / 2.0

        t0 = time.time()
        boundary, boundary_stats = generate_boundary(
            plane=plane,
            inlier_points=inlier_points,
            frame_id=frame_id,
        )
        t_boundary = (time.time() - t0) * 1000

        if boundary is None or not boundary_stats.get("valid", False):
            reason = boundary_stats.get("reason", "unknown")
            message = boundary_stats.get("message", "")
            dbg["reject_stage"]  = "boundary"
            dbg["reject_reason"] = f"boundary_gen_failed_{reason}"
            print(f"[P7_AREA_REJECT] id={frame_id} stage=boundary "
                  f"reason={reason} message={message}")
            return self._handle_invalid_floor(
                frame_id,
                message if message else f"boundary gen failed ({reason})",
                dbg,
            )

        confidence = boundary["confidence"]
        b_source   = boundary.get("source", "?")
        dbg["boundary_source"] = b_source
        dbg["boundary_width"]  = round(boundary["width"], 3)
        dbg["boundary_depth"]  = round(boundary["depth"], 3)

        # AI floor fit debug fields
        if boundary_stats.get("visible_w"):
            dbg["visible_w"]   = boundary_stats["visible_w"]
            dbg["visible_d"]   = boundary_stats["visible_d"]
            dbg["safe_w"]      = boundary_stats.get("safe_w", 0.0)
            dbg["safe_d"]      = boundary_stats.get("safe_d", 0.0)
            dbg["selected_w"]  = boundary_stats.get("selected_w", 0.0)
            dbg["selected_d"]  = boundary_stats.get("selected_d", 0.0)
            dbg["center_x"]    = boundary_stats.get("center_x", 0.0)
            dbg["center_z"]    = boundary_stats.get("center_z", 0.0)

        print(f"[P7_BOUNDARY] id={frame_id} source={b_source} "
              f"w={boundary['width']:.2f} d={boundary['depth']:.2f} "
              f"conf={confidence:.3f}")

        # ---- Step 5.5: Plane patch (1 m × 1 m debug quad in Unity cam space) ----
        plane_patch_camera = self._generate_plane_patch(
            plane, floor_center_x, floor_center_z, frame_id)

        # ---- Step 5.6: World-space Y verification ----
        boundary_world_y = self._world_space_check(
            boundary["boundary_camera"], M, frame_id)
        if boundary_world_y:
            dbg["boundary_world_y_avg"] = boundary_world_y["avg"]
            dbg["boundary_world_y_min"] = boundary_world_y["min"]
            dbg["boundary_world_y_max"] = boundary_world_y["max"]

        # ---- Step 6: Candidate scoring & lock ----
        if self._boundary_state in ("locked", "ready_to_confirm"):
            return self._locked_boundary_msg

        score = _score_candidate(
            horizontal_score  = horizontal_score,
            inlier_ratio      = plane["inlier_ratio"],
            confidence        = confidence,
            rmse              = plane["rmse"],
            floor_pixel_ratio = ratio_after,
        )
        dbg["candidate_score"] = round(score, 4)

        # Strict lock gate
        passes_lock_gate = (
            horizontal_score        >= self._lock_min_horizontal_score and
            ratio_after             >= self._lock_min_floor_ratio and
            plane["inlier_ratio"]   >= self._lock_min_inlier_ratio and
            plane["rmse"]           <= self._lock_max_rmse and
            confidence              >= self._lock_min_confidence
        )

        lock_allowed      = passes_lock_gate
        lock_block_reason = ""

        if used_fallback and not self._allow_fallback_lock:
            lock_allowed      = False
            lock_block_reason = "fallback_cannot_lock"
            print(f"[P5_REJECT] id={frame_id} stage=lock "
                  f"reason=fallback_cannot_lock")
        elif not passes_lock_gate:
            reasons = []
            if horizontal_score       < self._lock_min_horizontal_score:
                reasons.append(f"h={horizontal_score:.3f}<{self._lock_min_horizontal_score}")
            if ratio_after            < self._lock_min_floor_ratio:
                reasons.append(f"fr={ratio_after:.3f}<{self._lock_min_floor_ratio}")
            if plane["inlier_ratio"]  < self._lock_min_inlier_ratio:
                reasons.append(f"ir={plane['inlier_ratio']:.2f}<{self._lock_min_inlier_ratio}")
            if plane["rmse"]          > self._lock_max_rmse:
                reasons.append(f"rmse={plane['rmse']:.4f}>{self._lock_max_rmse}")
            if confidence             < self._lock_min_confidence:
                reasons.append(f"conf={confidence:.3f}<{self._lock_min_confidence}")
            lock_block_reason = "below_lock_gate: " + " | ".join(reasons)

        dbg["lock_allowed"]        = lock_allowed
        dbg["lock_block_reason"]   = lock_block_reason
        dbg["candidate_pool_count"] = len(self._candidate_pool)

        # Reset consecutive-invalid counter (this frame was geometrically valid)
        self._consecutive_invalid = 0

        if lock_allowed:
            self._candidate_pool.append({
                "plane":             plane,
                "boundary":          boundary,
                "normal_world":      normal_world.tolist(),
                "score":             score,
                "horizontal_score":  horizontal_score,
                "inlier_ratio":      plane["inlier_ratio"],
                "rmse":              plane["rmse"],
                "confidence":        confidence,
                "floor_ratio":       ratio_after,
                "used_fallback":     used_fallback,
                "frame_id":          frame_id,
                "plane_patch_camera": plane_patch_camera,
                "floor_candidates":  boundary_stats.get("floor_candidates", []),
                "recommended_candidate": boundary_stats.get("recommended_candidate", -1),
            })
            dbg["candidate_pool_count"] = len(self._candidate_pool)
            print(f"[P7_CANDIDATE] id={frame_id} pool="
                  f"{len(self._candidate_pool)}/{self._candidate_pool_size} "
                  f"score={score:.3f} h={horizontal_score:.3f} "
                  f"ir={plane['inlier_ratio']:.2f} rmse={plane['rmse']:.4f}")

            if len(self._candidate_pool) >= self._candidate_pool_size:
                best = max(self._candidate_pool, key=lambda c: c["score"])

                # Phase 7A: Placement mode determines final state
                if self._placement_mode == "user_confirm":
                    # Do NOT auto-lock. Send ready_to_confirm so Unity
                    # can show the Build Playing Area button.
                    final_state = "ready_to_confirm"
                    self._boundary_state      = "ready_to_confirm"
                    self._floor_confirmed     = True
                    dbg["boundary_state"]     = "ready_to_confirm"
                    dbg["locked"]             = False
                    dbg["lock_source_frame"]  = best["frame_id"]
                    dbg["placement_mode"]     = "user_confirm_required"

                    # Phase 7B: Include floor candidates in debug
                    dbg["floor_candidates_count"] = boundary_stats.get("floor_candidates", []).__len__()
                    dbg["recommended_candidate"]  = boundary_stats.get("recommended_candidate", -1)

                    self._locked_boundary_msg = self._build_message(
                        frame_id, best["boundary"], best["plane"],
                        best["horizontal_score"], ratio_after, final_state,
                        len(self._candidate_pool), self._candidate_pool_size,
                        dbg, used_fallback,
                        plane_patch_camera=best.get("plane_patch_camera"),
                        floor_candidates=best.get("floor_candidates"),
                    )
                    print(f"[P7_READY] floor_confirmed=True "
                          f"best_score={best['score']:.3f} "
                          f"state=ready_to_confirm "
                          f"pool={len(self._candidate_pool)} frame={frame_id}")
                    # Do NOT clear pool — keep sending ready_to_confirm
                    self._write_csv_row(dbg)
                    return self._locked_boundary_msg

                else:
                    # Legacy auto-lock (Phase 6 behavior)
                    self._boundary_state      = "locked"
                    dbg["boundary_state"]     = "locked"
                    dbg["locked"]             = True
                    dbg["lock_source_frame"]  = best["frame_id"]
                    self._locked_boundary_msg = self._build_message(
                        frame_id, best["boundary"], best["plane"],
                        best["horizontal_score"], ratio_after, "locked",
                        len(self._candidate_pool), self._candidate_pool_size,
                        dbg, used_fallback,
                        plane_patch_camera=best.get("plane_patch_camera"),
                    )
                    print(f"[P7_LOCK] locked=True best_score={best['score']:.3f} "
                          f"pool={len(self._candidate_pool)} frame={frame_id}")
                    self._candidate_pool = []
                    self._write_csv_row(dbg)
                    return self._locked_boundary_msg
            else:
                self._boundary_state  = "preview"
                dbg["boundary_state"] = "preview"
        else:
            print(f"[P7_CANDIDATE] id={frame_id} lock_allowed=False "
                  f"reason={lock_block_reason}")
            self._boundary_state  = "scanning" if not self._candidate_pool else "preview"
            dbg["boundary_state"] = self._boundary_state

        t_total = (time.time() - t_total_start) * 1000
        dbg["total_ms"] = round(t_total, 1)
        print(f"[P7_PERF] id={frame_id} total_ms={t_total:.1f}")

        self._write_csv_row(dbg)

        return self._build_message(
            frame_id, boundary, plane, horizontal_score,
            ratio_after, self._boundary_state,
            len(self._candidate_pool), self._candidate_pool_size,
            dbg, used_fallback,
            plane_patch_camera=plane_patch_camera,
            floor_candidates=boundary_stats.get("floor_candidates", []),
        )

    # ------------------------------------------------------------------
    # Phase 7C: Frame caching for hand tracking
    # ------------------------------------------------------------------

    def _cache_frame_for_hand(self, frame, metadata, frame_id):
        """Cache rotated frame + depth + intrinsics for hand tracking.

        Phase 9 fix: Track intrinsics version. If intrinsics change,
        invalidate cached depth/hand projection data.
        In post-lock mode, skip depth inference to save GPU.
        """
        import cv2

        fx = float(metadata.get("fx", 250.0)) if metadata else 250.0
        fy = float(metadata.get("fy", 250.0)) if metadata else 250.0
        cx = metadata.get("cx", None) if metadata else None
        cy = metadata.get("cy", None) if metadata else None

        h, w = frame.shape[:2]
        if cx is None: cx = w / 2.0
        if cy is None: cy = h / 2.0
        cx = float(cx); cy = float(cy)

        # Phase 9: Detect intrinsics change
        intrinsics_key = (round(fx, 2), round(fy, 2), round(cx, 2), round(cy, 2), w, h)
        if not hasattr(self, '_last_intrinsics_key'):
            self._last_intrinsics_key = None

        if intrinsics_key != self._last_intrinsics_key:
            if self._last_intrinsics_key is not None:
                print(f"[P9_CACHE] intrinsics changed, clearing depth/hand projection cache")
                print(f"[P9_CACHE] old={self._last_intrinsics_key} new={intrinsics_key}")
                self._last_depth_map = None
                self._intrinsics_version += 1
            self._last_intrinsics_key = intrinsics_key

        # Apply input rotation
        rotation_deg = INPUT_ROTATION_DEGREES
        rotated = frame
        if rotation_deg != 0:
            rotated, fx, fy, cx, cy = rotate_frame_and_intrinsics(
                frame, fx, fy, cx, cy, rotation_deg)

        # Phase 9: In post-lock mode, skip depth to save GPU
        # Hand tracking can still work with the last cached depth map
        if self._post_lock_mode and POST_LOCK_STOP_DEPTH:
            # Reuse last depth map (may be stale, but hand 2D still works)
            depth_map = self._last_depth_map
            if depth_map is None:
                # First frame after lock, run depth one more time
                depth_map = self._depth_model.predict(rotated)
        else:
            depth_map = self._depth_model.predict(rotated)

        # Cache for hand tracking
        self._last_rotated_frame = rotated
        self._last_depth_map = depth_map
        self._last_fx = fx
        self._last_fy = fy
        self._last_cx = cx
        self._last_cy = cy
        self._cached_intrinsics_version = self._intrinsics_version

        # Phase 9: Log cache state periodically
        if self._frame_count % 30 == 0:
            print(f"[P9_CACHE] frame cached id={frame_id} fx={fx:.1f} fy={fy:.1f} "
                  f"cx={cx:.1f} cy={cy:.1f} rot={rotation_deg} "
                  f"intrinsics_v={self._intrinsics_version}")

    # ------------------------------------------------------------------
    # Phase 7C: Hand tracking (runs independently of guardian state)
    # ------------------------------------------------------------------

    def _run_hand_tracking(self, rotated_frame, frame_id):
        """Run MediaPipe hand detection using the ROTATED frame.

        Uses cached depth map and intrinsics.
        This ensures landmark pixel coordinates align with depth pixels.
        Runs on EVERY frame regardless of guardian state (scanning,
        preview, ready_to_confirm, locked).
        """
        if self._hand_tracker is None:
            return

        if self._last_depth_map is None:
            print(f"[P7_HAND_SKIP] frame={frame_id} reason=no_depth_map")
            return

        import cv2

        h, w = rotated_frame.shape[:2]
        print(f"[P7_HAND_IN] frame={frame_id} image={w}x{h}")

        # Convert rotated frame to RGB (MediaPipe expects RGB)
        if len(rotated_frame.shape) == 2:
            rgb = cv2.cvtColor(rotated_frame, cv2.COLOR_GRAY2RGB)
        elif rotated_frame.shape[2] == 4:
            rgb = cv2.cvtColor(rotated_frame, cv2.COLOR_BGRA2RGB)
        elif rotated_frame.shape[2] == 3:
            rgb = cv2.cvtColor(rotated_frame, cv2.COLOR_BGR2RGB)
        else:
            rgb = rotated_frame

        # Ensure depth map matches frame dimensions
        depth = self._last_depth_map
        if depth.shape[:2] != rgb.shape[:2]:
            depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]),
                               interpolation=cv2.INTER_LINEAR)

        hand_result = self._hand_tracker.detect(
            rgb_frame=rgb,
            depth_map=depth,
            fx=self._last_fx,
            fy=self._last_fy,
            cx=self._last_cx,
            cy=self._last_cy,
            frame_id=frame_id,
        )

        if hand_result is not None:
            if self._tcp_server is not None:
                self._tcp_server.push_ai_result(hand_result)
            valid_3d = hand_result.get("valid_3d", True)
            print(f"[P7_HAND_SEND] type=AI_HAND_DATA frame={frame_id} "
                  f"hand={hand_result['handedness']} "
                  f"conf={hand_result['confidence']:.2f} "
                  f"pts={hand_result.get('valid_landmark_count', 0)}/21 "
                  f"valid_3d={valid_3d}")
            print(f"[P7_TCP_SEND] type=AI_HAND_DATA frame={frame_id} "
                  f"valid={hand_result.get('hand_valid', False)}")
        else:
            print(f"[P7_HAND_NONE] frame={frame_id}")

    # ------------------------------------------------------------------
    # Plane patch (small 1 m × 1 m debug quad on the detected floor)
    # ------------------------------------------------------------------

    
    def _generate_plane_patch(self, plane, center_x, center_z, frame_id):
        """Return a 1 m × 1 m quad on the plane, in Unity camera space.

        Coordinate conversion (OpenCV cam → Unity cam):
            X → x   (unchanged)
            Y → -y  (flip: down → up)
            Z → z   (unchanged: forward stays forward in both spaces)
        """
        try:
            a, b, c = plane["normal"]
            d_val   = float(plane["d"])
            if abs(b) < 1e-6:
                return None
            half    = 0.5
            pts     = []
            for x, z in [
                (center_x - half, center_z - half),
                (center_x + half, center_z - half),
                (center_x + half, center_z + half),
                (center_x - half, center_z + half),
            ]:
                y_cv = -(a * x + c * z + d_val) / b
                pts.append([
                    round(x,     4),
                    round(-y_cv, 4),   # flip Y: OpenCV down → Unity up
                    round(z,     4),   # keep Z: forward stays forward
                ])
            return pts
        except Exception as e:
            print(f"[P5_PLANE_PATCH_ERR] {e}")
            return None

    # ------------------------------------------------------------------
    # World-space Y verification
    # ------------------------------------------------------------------

    def _world_space_check(self, boundary_camera, M, frame_id):
        """Transform boundary corners to world space and report Y range."""
        try:
            R = M[:3, :3]
            t = M[:3, 3]
            ys = [float((R @ np.array(pt, dtype=np.float64) + t)[1])
                  for pt in boundary_camera]
            y_min, y_max, y_avg = min(ys), max(ys), sum(ys) / len(ys)
            print(f"[P5_WORLD_CHECK] id={frame_id} "
                  f"y_min={y_min:.3f} y_max={y_max:.3f} y_avg={y_avg:.3f}")
            return {"min": round(y_min, 3), "max": round(y_max, 3), "avg": round(y_avg, 3)}
        except Exception as e:
            print(f"[P5_WORLD_CHECK_ERR] {e}")
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _write_csv_row(self, dbg: dict):
        if not (self._csv_writer and self._csv_file):
            return
        try:
            self._csv_writer.writerow([
                dbg.get("frame_id", ""),
                dbg.get("seg_mode", ""),
                dbg.get("used_fallback", ""),
                dbg.get("semantic_floor_ratio", ""),
                dbg.get("semantic_confidence", ""),
                dbg.get("floor_ratio_after_roi", ""),
                dbg.get("depth_mean", ""),
                dbg.get("backproject_points", ""),
                dbg.get("plane_rmse", ""),
                dbg.get("plane_inlier_ratio", ""),
                dbg.get("score_col", ""),
                dbg.get("score_row", ""),
                dbg.get("selected_matrix_order", ""),
                dbg.get("horizontal_score", ""),
                dbg.get("input_rotation_degrees", ""),
                dbg.get("boundary_state", ""),
                dbg.get("lock_allowed", ""),
                dbg.get("reject_stage", ""),
                dbg.get("reject_reason", ""),
                dbg.get("total_ms", ""),
            ])
            self._csv_file.flush()
        except Exception:
            pass

    def _build_message(
        self, frame_id, boundary, plane,
        horizontal_score, floor_ratio_after,
        state, stable_count, stable_required,
        dbg=None, used_fallback=False,
        plane_patch_camera=None,
        floor_candidates=None,
    ) -> dict:
        msg = {
            "type":            "AI_GUARDIAN_DATA",
            "frame_id":        frame_id,
            "timestamp_ms":    int(time.time() * 1000),
            "floor_valid":     True,
            "confidence":      boundary["confidence"],
            "source":          "ai_floor_depth_segmentation",
            "boundary_state":  state,
            "placement_mode":  self._placement_mode,
            "stable_count":    stable_count,
            "stable_required": stable_required,
            "boundary_camera": boundary["boundary_camera"],
            "plane_camera": {
                "normal":       plane["normal"],
                "d":            plane["d"],
                "rmse":         plane["rmse"],
                "inlier_ratio": plane["inlier_ratio"],
            },
            # Full debug dict always included so Unity diagnostic UI has it
            "debug": dbg if dbg else {
                "floor_pixel_ratio": floor_ratio_after,
                "horizontal_score":  round(horizontal_score, 4),
                "boundary_width":    round(boundary.get("width", 0), 3),
                "boundary_depth":    round(boundary.get("depth", 0), 3),
                "boundary_source":   boundary.get("source", "?"),
                "used_fallback":     used_fallback,
                "lock_allowed":      not used_fallback,
            },
        }
        if plane_patch_camera:
            msg["plane_patch_camera"] = plane_patch_camera
        # Phase 7B: Include floor candidates for Unity to render 3 blue rectangles
        if floor_candidates:
            msg["floor_candidates"] = floor_candidates
        return msg

    def _handle_invalid_floor(self, frame_id, reason: str, dbg=None) -> dict:
        if self._boundary_state in ("locked", "ready_to_confirm") and self._locked_boundary_msg is not None:
            return self._locked_boundary_msg

        self._consecutive_invalid += 1
        if (self._consecutive_invalid >= self._max_consecutive_invalid
                and self._candidate_pool):
            print(f"[P7_CANDIDATE] pool reset after "
                  f"{self._consecutive_invalid} consecutive invalid frames")
            self._candidate_pool      = []
            self._consecutive_invalid = 0

        self._boundary_state = "scanning"
        if dbg:
            dbg["boundary_state"] = "scanning"
            self._write_csv_row(dbg)
        return self._make_status_message(frame_id, f"Scanning... ({reason})", dbg)

    def _make_status_message(self, frame_id, message_text: str, dbg=None) -> dict:
        msg = {
            "type":         "AI_GUARDIAN_STATUS",
            "frame_id":     frame_id,
            "timestamp_ms": int(time.time() * 1000),
            "floor_valid":  False,
            "message":      message_text,
        }
        if dbg:
            msg["debug"] = dbg
        print(f"[P5_STATUS] frame={frame_id} {message_text}")
        return msg

    def _send_scanning_status(self, frame_id):
        msg = self._make_status_message(frame_id, "Scanning floor...")
        with self._output_lock:
            self._latest_result = msg
        if self._tcp_server is not None:
            self._tcp_server.push_ai_result(msg)

    # ------------------------------------------------------------------
    # Phase 9.1: Rolling FPS computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_rolling_fps(timestamps, window_s=5.0):
        """Compute FPS from a rolling window of timestamps."""
        if len(timestamps) < 2:
            return 0.0
        now = timestamps[-1]
        cutoff = now - window_s
        # Count timestamps within window
        recent = [t for t in timestamps if t >= cutoff]
        if len(recent) < 2:
            return 0.0
        duration = recent[-1] - recent[0]
        if duration < 0.01:
            return 0.0
        return (len(recent) - 1) / duration

    # ------------------------------------------------------------------
    # Phase 9.1: Metrics export
    # ------------------------------------------------------------------

    def get_session_metrics(self) -> dict:
        """Return current session metrics dictionary with proper FPS."""
        m = {}
        now = time.time()

        # Copy non-internal metrics
        for k, v in self._metrics.items():
            if not k.startswith("_"):
                m[k] = v

        m["end_time"] = now
        m["duration_s"] = round(now - self._metrics["start_time"], 1)
        m["total_frames"] = self._metrics["warm_frames"] + self._metrics["idle_frames"]

        # WARM phase averages
        wf = max(1, self._metrics["warm_frames"])
        warm_duration = 0.0
        if self._metrics.get("warm_end_time") and self._metrics["warm_start_time"]:
            warm_duration = self._metrics["warm_end_time"] - self._metrics["warm_start_time"]
        elif self._metrics["lock_time"]:
            warm_duration = self._metrics["lock_time"] - self._metrics["warm_start_time"]
        else:
            warm_duration = now - self._metrics["warm_start_time"]

        m["warm_duration_s"] = round(warm_duration, 1)
        m["warm_pipeline_fps"] = round(self._metrics["warm_frames"] / max(0.1, warm_duration), 1)
        m["avg_warm_depth_ms"] = round(self._metrics["warm_depth_ms_total"] / wf, 1)
        m["avg_warm_seg_ms"] = round(self._metrics["warm_seg_ms_total"] / wf, 1)
        m["avg_warm_plane_ms"] = round(self._metrics["warm_plane_ms_total"] / wf, 1)
        m["avg_warm_hand_ms"] = round(self._metrics["warm_hand_ms_total"] / max(1, self._metrics["warm_hand_frames"]), 1)
        m["warm_tcp_guardian_hz"] = round(self._metrics["warm_tcp_guardian_sends"] / max(0.1, warm_duration), 1)
        m["warm_tcp_hand_hz"] = round(self._metrics["warm_tcp_hand_sends"] / max(0.1, warm_duration), 1)

        # IDLE phase averages
        idle_duration = 0.0
        if self._metrics.get("idle_start_time"):
            idle_duration = now - self._metrics["idle_start_time"]
        elif self._metrics.get("lock_time"):
            idle_duration = now - self._metrics["lock_time"]

        m["idle_duration_s"] = round(idle_duration, 1)
        if idle_duration > 0.1 and self._metrics["idle_frames"] > 0:
            m["idle_fps"] = round(self._metrics["idle_frames"] / idle_duration, 1)
        else:
            m["idle_fps"] = 0.0
        m["idle_hand_fps"] = round(
            self._metrics["idle_hand_frames"] / max(0.1, idle_duration), 1) if idle_duration > 0.1 else 0.0
        m["avg_idle_hand_ms"] = round(
            self._metrics["idle_hand_ms_total"] / max(1, self._metrics["idle_hand_frames"]), 1)
        m["idle_tcp_hand_hz"] = round(
            self._metrics["idle_tcp_hand_sends"] / max(0.1, idle_duration), 1) if idle_duration > 0.1 else 0.0

        return m

    def export_session_artifacts(self, output_dir: str = "artifacts"):
        """Write session metrics to JSON, CSV, and summary TXT."""
        import json, csv, os

        os.makedirs(output_dir, exist_ok=True)
        m = self.get_session_metrics()

        # JSON
        json_path = os.path.join(output_dir, "session_metrics.json")
        with open(json_path, "w") as f:
            json.dump(m, f, indent=2, default=str)
        print(f"[P9_EXPORT] wrote {json_path}")

        # CSV
        csv_path = os.path.join(output_dir, "session_metrics.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for k, v in sorted(m.items()):
                writer.writerow([k, v])
        print(f"[P9_EXPORT] wrote {csv_path}")

        # Summary TXT
        txt_path = os.path.join(output_dir, "session_summary.txt")
        with open(txt_path, "w") as f:
            f.write("AI Guardian — Phase 9.1 Session Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Duration: {m['duration_s']}s\n")
            f.write(f"Total Frames: {m['total_frames']}\n")
            f.write(f"Lock Frame: {m.get('lock_frame_id', 'N/A')}\n")
            f.write(f"Errors: {m.get('errors', 0)}\n")
            f.write(f"Warnings: {m.get('warnings', 0)}\n\n")

            f.write("--- WARM PHASE (depth+seg ON) ---\n")
            f.write(f"Duration: {m['warm_duration_s']}s\n")
            f.write(f"Frames: {m['warm_frames']}\n")
            f.write(f"Pipeline FPS: {m['warm_pipeline_fps']}\n")
            f.write(f"Avg Depth: {m['avg_warm_depth_ms']}ms\n")
            f.write(f"Avg Seg: {m['avg_warm_seg_ms']}ms\n")
            f.write(f"Avg Plane: {m['avg_warm_plane_ms']}ms\n")
            f.write(f"Avg Hand: {m['avg_warm_hand_ms']}ms\n")
            f.write(f"TCP Guardian Hz: {m['warm_tcp_guardian_hz']}\n")
            f.write(f"TCP Hand Hz: {m['warm_tcp_hand_hz']}\n\n")

            f.write("--- IDLE PHASE (depth+seg OFF, hand ON) ---\n")
            f.write(f"Duration: {m['idle_duration_s']}s\n")
            f.write(f"Frames: {m['idle_frames']}\n")
            f.write(f"Idle FPS: {m['idle_fps']}\n")
            f.write(f"Hand FPS: {m.get('idle_hand_fps', 0)}\n")
            f.write(f"Avg Hand: {m['avg_idle_hand_ms']}ms\n")
            f.write(f"TCP Hand Hz: {m['idle_tcp_hand_hz']}\n")

        print(f"[P9_EXPORT] wrote {txt_path}")
        print(f"[P9_EXPORT] session metrics exported: "
              f"warm_fps={m['warm_pipeline_fps']} idle_fps={m['idle_fps']} "
              f"hand_frames={m.get('hand_frames_total', 0)}")

        return m