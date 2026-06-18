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
    # Phase 2F: headset lock tolerance
    HEADSET_LOCK_MIN_HORIZONTAL_SCORE, HEADSET_LOCK_MIN_CONFIDENCE,
    # Phase 2K: Unity-space coordinate sanity gate (req 7)
    UNITY_LOCK_MAX_Y_SPREAD_M,
    # Phase 7A
    HAND_TRACKING_ENABLED, HAND_TRACKING_TARGET_FPS,
    HAND_MIN_DETECTION_CONF, HAND_MIN_TRACKING_CONF, HAND_MAX_NUM_HANDS,
    PLACEMENT_MODE,
    # Phase 9: Post-lock optimization
    POST_LOCK_ENABLED, POST_LOCK_DIAGNOSTIC_FPS,
    POST_LOCK_STOP_DEPTH, POST_LOCK_STOP_SEGMENTATION,
    POST_LOCK_STOP_PLANE_FIT, POST_LOCK_KEEP_HAND, POST_LOCK_HAND_DEPTH_FPS,
    POST_LOCK_HAND_MIN_HZ,
    # Phase 10.6: Hybrid (SegFormer + geometry) floor detection
    FLOOR_DETECTION_MODE, GEO_FLOOR_ROI_TOP_FRAC,
    GEO_FLOOR_INLIER_THRESH_M, GEO_FLOOR_MIN_CAM_HORIZ,
    GEO_FLOOR_SEMANTIC_BAND_MULT,
    HYBRID_SEM_FALLBACK_RATIO, HYBRID_SEM_FALLBACK_CONF,
    HYBRID_SEM_HIGH_CONF, HYBRID_SEM_HIGH_BAND_MULT,
    # Phase 14: Multi-frame world-space floor accumulation
    FLOOR_ACCUM_ENABLED, FLOOR_ACCUM_CELL_M, FLOOR_ACCUM_MAX_SIZE_M,
    FLOOR_ACCUM_MIN_CELL_HITS, FLOOR_ACCUM_PLANE_BAND_M,
    FLOOR_ACCUM_MIN_FRAMES, FLOOR_ACCUM_MIN_CELLS, FLOOR_ACCUM_MIN_AREA_M2,
    FLOOR_ACCUM_STABILITY_WINDOW, FLOOR_ACCUM_CENTER_TOL_M,
    FLOOR_ACCUM_YAW_TOL_DEG, FLOOR_ACCUM_FLOORY_TOL_M, FLOOR_ACCUM_SIZE_TOL_M,
    FLOOR_ACCUM_MAX_CELLS, FLOOR_ACCUM_RESET_AFTER_INVALID,
    FLOOR_ACCUM_AREA_GROWTH_FRAMES, FLOOR_ACCUM_AREA_GROWTH_EPS_M2,
    # Phase 2L: rolling + manual lock
    FLOOR_ACCUM_ROLLING_LOCK, FLOOR_ACCUM_LOCK_MAX_WAIT_FRAMES,
    FLOOR_ACCUM_MANUAL_LOCK_ENABLED,
    # Phase 2Q: largest-area auto-lock
    FLOOR_ACCUM_AUTO_LOCK_MIN_AREA_M2,
    # Phase 2M: Python-side optical-axis correction
    APPLY_OPTICAL_AXIS_CORRECTION, OPTICAL_AXIS_CORRECTION_SIGN,
)

from models.depth_model       import DepthModelBase, create_depth_model
from models.floor_segmenter   import FloorSegmenter
from geometry.backproject     import backproject_floor_points
from geometry.plane_fit       import fit_floor_plane
from geometry.boundary_generator import generate_boundary
from geometry.floor_from_depth import detect_floor_from_depth, _morph_clean
from geometry.rotation_utils  import rotate_frame_and_intrinsics
from geometry.floor_accumulator import FloorAccumulator


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

        # Phase 10.6: Hybrid (SegFormer + geometry) floor detection (from config)
        self._floor_detection_mode    = FLOOR_DETECTION_MODE
        self._geo_roi_top_frac        = GEO_FLOOR_ROI_TOP_FRAC
        self._geo_inlier_thresh_m     = GEO_FLOOR_INLIER_THRESH_M
        self._geo_min_cam_horiz       = GEO_FLOOR_MIN_CAM_HORIZ
        self._geo_semantic_band_mult  = GEO_FLOOR_SEMANTIC_BAND_MULT
        self._hybrid_sem_fb_ratio     = HYBRID_SEM_FALLBACK_RATIO
        self._hybrid_sem_fb_conf      = HYBRID_SEM_FALLBACK_CONF
        self._hybrid_sem_high_conf    = HYBRID_SEM_HIGH_CONF
        self._hybrid_sem_high_band    = HYBRID_SEM_HIGH_BAND_MULT
        self._geo_debug_dir           = os.path.join(os.path.dirname(os.path.dirname(__file__)), "debug")
        self._geo_call_count          = 0

        # State
        self._boundary_state       = "scanning"
        self._locked_boundary_msg  = None
        self._candidate_pool       = []
        self._candidate_pool_size  = CANDIDATE_POOL_SIZE
        self._consecutive_invalid  = 0
        self._max_consecutive_invalid = MAX_CONSECUTIVE_INVALID_BEFORE_POOL_RESET

        # Phase 14: world-space multi-frame floor accumulator. Replaces the
        # single-best per-frame candidate-pool lock with a cubic built from ALL
        # trusted floor points seen during scanning (≤3×3 m), locked only once
        # center/rotation/floorY/width/depth are stable.
        self._floor_accum_enabled  = FLOOR_ACCUM_ENABLED
        self._accum_min_frames     = FLOOR_ACCUM_MIN_FRAMES
        self._accum_max_size       = FLOOR_ACCUM_MAX_SIZE_M
        self._accum_reset_after_invalid = FLOOR_ACCUM_RESET_AFTER_INVALID
        self._floor_accum = FloorAccumulator(
            cell_size_m      = FLOOR_ACCUM_CELL_M,
            max_size_m       = FLOOR_ACCUM_MAX_SIZE_M,
            min_cell_hits    = FLOOR_ACCUM_MIN_CELL_HITS,
            plane_band_m     = FLOOR_ACCUM_PLANE_BAND_M,
            min_frames       = FLOOR_ACCUM_MIN_FRAMES,
            min_cells        = FLOOR_ACCUM_MIN_CELLS,
            min_area_m2      = FLOOR_ACCUM_MIN_AREA_M2,
            stability_window = FLOOR_ACCUM_STABILITY_WINDOW,
            center_tol_m     = FLOOR_ACCUM_CENTER_TOL_M,
            yaw_tol_deg      = FLOOR_ACCUM_YAW_TOL_DEG,
            floory_tol_m     = FLOOR_ACCUM_FLOORY_TOL_M,
            size_tol_m       = FLOOR_ACCUM_SIZE_TOL_M,
            max_cells        = FLOOR_ACCUM_MAX_CELLS,
            area_growth_frames = FLOOR_ACCUM_AREA_GROWTH_FRAMES,
            area_growth_eps_m2 = FLOOR_ACCUM_AREA_GROWTH_EPS_M2,
            rolling_lock       = FLOOR_ACCUM_ROLLING_LOCK,
            lock_max_wait_frames = FLOOR_ACCUM_LOCK_MAX_WAIT_FRAMES,
            auto_lock_min_area_m2 = FLOOR_ACCUM_AUTO_LOCK_MIN_AREA_M2,
        )
        self._manual_lock_enabled = FLOOR_ACCUM_MANUAL_LOCK_ENABLED

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
        self._last_rotation_deg    = INPUT_ROTATION_DEGREES  # Phase 1.5: resolved per-frame
        self._last_rotation_src    = "config"                # "packet" | "config"
        # Phase 2F: floor-mask orientation sanity (floor vs ceiling). Diagnostic only —
        # computed from the existing mask, never alters SegFormer/RANSAC.
        self._last_orient_ok       = False
        self._last_centroid_y      = 0.0
        self._last_bottom_ratio    = 0.0
        self._last_horizontal_score = 0.0   # Phase 2H: sent on every msg for the geometry probe
        self._optical_corrected     = False  # Phase 2M: optical-axis correction applied this frame
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
        self._last_depth_refresh_time = 0.0   # Phase 3: post-lock depth throttle

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
        # Phase 10.2 (J): optional Unity→Python lock-event listener. Set via
        # set_unity_lock_listener(). Used only for session metrics — never
        # touches the safety pipeline.
        self._unity_lock_listener = None

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

    def set_unity_lock_listener(self, listener):
        """Phase 10.2 (J): hold a reference to the UnityLockListener so
        export_session_artifacts() can pull lock_frame_id / lock unity-time
        into session metrics. Safe to call with None — getter is null-checked.
        """
        self._unity_lock_listener = listener
        print(f"[P10_LOCK_LISTENER] linked={listener is not None}")

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

        # Phase 10.6: SegFormer is needed for "hybrid" (fusion) and "semantic"
        # (legacy) modes. Only the pure-geometry diagnostic mode skips it.
        if self._floor_detection_mode == "geometry_first":
            print("[P5_PIPELINE] floor_mode=geometry_first — SegFormer NOT loaded (diagnostic)")
            self._floor_segmenter = None
        else:
            print(f"[P5_PIPELINE] floor_mode={self._floor_detection_mode} — loading SegFormer")
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

                # Phase 3 [HAND_SPEEDUP]: drop to post-lock (floor models OFF,
                # MediaPipe ON at full rate) as soon as the lock happens — via ANY
                # of these triggers, so a lost ACK packet can't keep the hand at
                # ~1 Hz. Once Unity enters the lobby the camera shows the VR room,
                # not the real floor, so our accumulator can NEVER lock on its own
                # there — hence the network + floor-lost fallbacks below.
                if not self._post_lock_mode:
                    trigger = None
                    if self._unity_lock_listener is not None:
                        if self._unity_lock_listener.get_last_lock() is not None:
                            trigger = "unity_ack"
                        elif self._unity_lock_listener.lobby_ready:
                            trigger = "unity_lobby_ready"
                    # Python-only fallback (no network): the floor was scanned and
                    # the accumulator built a rectangle, then the floor vanished for
                    # several frames (Unity switched to the lobby view) → the lock
                    # already happened, so free the GPU for the hand.
                    if (trigger is None
                            and self._floor_accum.frame_count > 0
                            and self._consecutive_invalid >= 5):
                        trigger = "floor_lost_fallback"
                    if trigger is not None:
                        self._post_lock_mode = True
                        if self._metrics.get("lock_frame_id") is None:
                            ack = (self._unity_lock_listener.get_last_lock()
                                   if self._unity_lock_listener is not None else None)
                            self._metrics["lock_frame_id"] = (
                                ack.get("frame_id", frame_id) if ack else frame_id)
                        self._metrics["lock_time"] = time.time()
                        self._metrics["idle_start_time"] = time.time()
                        self._metrics["warm_end_time"] = time.time()
                        self._metrics["lock_source"] = trigger
                        print(f"[P9_MODE] {trigger} → post-lock: "
                              f"depth=OFF seg=OFF plane=OFF hand=ON@{HAND_TRACKING_TARGET_FPS:.0f}Hz "
                              f"(GPU freed for usable hand ray)")
                        print(f"[P9_MODE] UNITY ACK lock → hand=ON@{HAND_TRACKING_TARGET_FPS:.0f}Hz")

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
    # Phase 10.6: Hybrid floor detection  (SegFormer + depth/geometry)
    # ------------------------------------------------------------------

    def _detect_floor_hybrid(self, frame, depth_map, fx, fy, cx, cy, frame_id):
        """
        Fuse SegFormer semantic floor with the dominant horizontal depth-plane.

          1. SegFormer raw inference  → semantic floor mask (NO rejection gate).
          2. Depth-plane detection    → robust backbone, with the semantic mask
             unioned in where it agrees with the plane (recovers detail, boosts
             confidence). This is the normal path and works on glossy tile where
             SegFormer alone fails.
          3. Fallback: if geometry finds NO horizontal plane but SegFormer is
             confident, use the semantic floor mask so a depth-degenerate frame
             that SegFormer nailed is not thrown away.

        Returns (floor_mask, stats) with the legacy seg_stats interface plus
        hybrid extras (semantic_ratio, semantic_confidence, semantic_agreement,
        mode in {"hybrid","semantic_fallback","geometry"}).
        """
        self._geo_call_count += 1
        h, w = depth_map.shape[:2]

        # --- 1. SegFormer raw (no rejection) ---
        # Phase 14: keep the per-pixel floor PROBABILITY map (was discarded) so
        # the fusion can trust HIGH-confidence floor pixels strongly.
        sem_mask = None
        sem_prob = None
        sem_conf = 0.0
        sem_ratio = 0.0
        if self._floor_segmenter is not None:
            try:
                sem_mask, sem_prob, sem_conf = self._floor_segmenter._run_inference(frame)
                sem_ratio = float(sem_mask.mean()) if sem_mask is not None else 0.0
            except Exception as e:
                print(f"[HYBRID] id={frame_id} SegFormer inference failed: {e}")
                sem_mask = None
                sem_prob = None

        # --- 2. Geometry plane + semantic union (+ high-confidence semantic) ---
        floor_mask, stats = detect_floor_from_depth(
            depth_map=depth_map,
            fx=fx, fy=fy, cx=cx, cy=cy,
            roi_top_frac=self._geo_roi_top_frac,
            min_depth=self._min_depth,
            max_depth=self._max_depth,
            inlier_threshold=self._geo_inlier_thresh_m,
            min_camera_horizontal=self._geo_min_cam_horiz,
            semantic_floor_mask=sem_mask,
            semantic_band_mult=self._geo_semantic_band_mult,
            semantic_prob=sem_prob,
            semantic_high_conf=self._hybrid_sem_high_conf,
            semantic_high_band_mult=self._hybrid_sem_high_band,
            frame_id=frame_id,
        )

        # --- 3. Semantic fallback when geometry found no horizontal plane ---
        if (stats.get("rejected")
                and sem_mask is not None
                and sem_ratio >= self._hybrid_sem_fb_ratio
                and sem_conf >= self._hybrid_sem_fb_conf):
            roi = np.zeros((h, w), dtype=bool)
            roi[int(h * self._geo_roi_top_frac):, :] = True
            fb_mask = _morph_clean(sem_mask.astype(bool) & roi, kernel=9)
            fb_pixels = int(fb_mask.sum())
            print(f"[HYBRID] id={frame_id} geometry rejected "
                  f"({stats.get('reject_reason')}) → SEMANTIC FALLBACK "
                  f"sem_ratio={sem_ratio:.3f} conf={sem_conf:.3f} pixels={fb_pixels}")
            floor_mask = fb_mask
            stats = {
                "floor_pixels": fb_pixels,
                "ratio": fb_pixels / (h * w) if (h * w) > 0 else 0.0,
                "confidence": round(float(sem_conf), 3),
                "used_fallback": True,
                "rejected": False,
                "reject_reason": None,
                "plane": None, "plane_normal": None,
                "inlier_ratio": 0.0, "rmse": 0.0, "camera_horizontal": 0.0,
                "semantic_agreement": 1.0,
                "mode": "semantic_fallback",
            }

        # attach semantic diagnostics for logging / CSV
        stats["semantic_ratio"] = round(sem_ratio, 4)
        stats["semantic_confidence"] = round(float(sem_conf), 4)

        if SAVE_DEBUG_IMAGES and (self._geo_call_count % DEBUG_IMAGE_EVERY_N == 1):
            self._save_geo_debug_overlay(frame, floor_mask, stats, frame_id, sem_mask)

        return floor_mask, stats

    # ------------------------------------------------------------------
    # Phase 10.6: Geometry-first floor detection (diagnostic, no SegFormer)
    # ------------------------------------------------------------------

    def _detect_floor_geometry_first(self, frame, depth_map, fx, fy, cx, cy, frame_id):
        """
        Detect the floor as the dominant horizontal plane in the depth map,
        bypassing SegFormer's unreliable class label. Returns a (floor_mask,
        stats) pair whose stats are interface-compatible with the legacy
        segmenter (keys: floor_pixels, ratio, confidence, used_fallback,
        rejected, reject_reason, plus geometry extras: plane, plane_normal,
        inlier_ratio, rmse, camera_horizontal, mode).
        """
        self._geo_call_count += 1
        floor_mask, stats = detect_floor_from_depth(
            depth_map=depth_map,
            fx=fx, fy=fy, cx=cx, cy=cy,
            roi_top_frac=self._geo_roi_top_frac,
            min_depth=self._min_depth,
            max_depth=self._max_depth,
            inlier_threshold=self._geo_inlier_thresh_m,
            min_camera_horizontal=self._geo_min_cam_horiz,
            frame_id=frame_id,
        )

        if SAVE_DEBUG_IMAGES and (self._geo_call_count % DEBUG_IMAGE_EVERY_N == 1):
            self._save_geo_debug_overlay(frame, floor_mask, stats, frame_id)

        return floor_mask, stats

    def _save_geo_debug_overlay(self, frame, floor_mask, stats, frame_id, semantic_mask=None):
        """
        Save an overlay so the floor is inspectable:
          GREEN = the full DETECTED floor — the geometry plane floor UNION the
                  SegFormer floor inside the ROI. This includes the far floor
                  that Depth Anything curved off the plane (previously painted
                  red), so green matches the real scanned floor.
          RED   = genuinely rejected NON-floor inside the evaluated lower region
                  (wall base / sofa / objects) — never accepted floor.
        """
        try:
            import cv2
            os.makedirs(self._geo_debug_dir, exist_ok=True)
            ts  = int(time.time())
            ov  = frame.copy()
            roi = np.zeros(frame.shape[:2], dtype=bool)
            roi[int(frame.shape[0] * self._geo_roi_top_frac):, :] = True
            green = floor_mask.astype(bool)
            if semantic_mask is not None:
                green = green | (semantic_mask.astype(bool) & roi)
            red = roi & ~green                       # rejected non-floor in ROI only
            if red.any():
                ov[red] = (
                    ov[red].astype(np.int32) * [0.3, 0.3, 1.0]
                ).clip(0, 255).astype(np.uint8)
            if green.any():
                ov[green] = (
                    ov[green].astype(np.int32) * [0.3, 1.0, 0.3]
                ).clip(0, 255).astype(np.uint8)
            mode = stats.get("mode", "geometry")
            cv2.putText(ov, f"{mode.upper()} id={frame_id} ratio={stats['ratio']*100:.0f}% "
                            f"conf={stats['confidence']:.2f} "
                            f"sem_agree={stats.get('semantic_agreement',0):.2f}",
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            if stats.get("rejected"):
                cv2.putText(ov, f"REJECT {stats.get('reject_reason','')}",
                            (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.imwrite(os.path.join(self._geo_debug_dir,
                        f"frame_{frame_id}_{ts}_geo.jpg"), ov)
        except Exception as e:
            print(f"[GEO_FLOOR_DEBUG_ERR] {e}")

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def _run_pipeline(self, frame, metadata, frame_id) -> dict:
        t_total_start = time.time()

        # Phase 2H: reset per-frame geometry-probe signals so an early reject reports
        # a flat-plane score of 0 for this rotation (not a stale value).
        self._last_horizontal_score = 0.0
        self._last_orient_ok        = False

        # --- Read pose from metadata ---
        pose      = (metadata or {}).get("camera_to_world") or (metadata or {}).get("pose")
        has_pose  = (
            pose is not None and
            isinstance(pose, list) and
            len(pose) == 16
        )

        # --- Phase 7C: Reuse cached rotated frame + depth from _cache_frame_for_hand ---
        # _cache_frame_for_hand already ran rotation + depth before this method.
        rotation_deg = self._last_rotation_deg  # Phase 1.5: resolved in _cache_frame_for_hand
        frame     = self._last_rotated_frame  # Already rotated
        depth_map = self._last_depth_map      # Already computed
        fx = self._last_fx
        fy = self._last_fy
        cx = self._last_cx
        cy = self._last_cy
        h, w = frame.shape[:2]

        # Phase 1.5: rot now tracks device orientation (src=packet) when Unity tags
        # it, else the config default (src=config). 'upright' is the post-rotation
        # frame orientation SegFormer actually sees.
        print(f"[P5_ROTATE] id={frame_id} rot={rotation_deg} src={self._last_rotation_src} "
              f"upright={'landscape' if w > h else 'portrait'} "
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

        # ---- Step 2: Floor detection (hybrid / geometry-first / legacy semantic) ----
        t0              = time.time()
        if self._floor_detection_mode == "hybrid":
            floor_mask, seg_stats = self._detect_floor_hybrid(
                frame, depth_map, fx, fy, cx, cy, frame_id)
        elif self._floor_detection_mode == "geometry_first":
            floor_mask, seg_stats = self._detect_floor_geometry_first(
                frame, depth_map, fx, fy, cx, cy, frame_id)
        else:
            floor_mask, seg_stats = self._floor_segmenter.segment(frame, frame_id=frame_id)
        t_seg           = (time.time() - t0) * 1000
        dbg["seg_ms"]   = round(t_seg, 1)
        dbg["seg_mode"] = seg_stats.get("mode", "semantic")

        # Handle early rejection (USE_FALLBACK_FLOOR_MASK=False path)
        if seg_stats.get("rejected", False):
            reason = seg_stats.get("reject_reason", "semantic_floor_too_low")
            dbg["seg_mode"]              = seg_stats.get("mode", "semantic")
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
        seg_mode      = seg_stats.get("mode", "fallback" if used_fallback else "semantic")

        dbg["seg_mode"]              = seg_mode
        dbg["used_fallback"]         = used_fallback
        dbg["semantic_floor_pixels"] = floor_pixels
        dbg["semantic_floor_ratio"]  = round(floor_ratio, 4)
        dbg["semantic_confidence"]   = round(floor_conf, 4)

        print(f"[P5_SEG] id={frame_id} mode={seg_mode} "
              f"pixels={floor_pixels} ratio={floor_ratio:.4f} conf={floor_conf:.3f}")
        # Phase 1 required logs.
        print(f"[SEG_FLOOR] pixels={floor_pixels} ratio={floor_ratio:.4f} conf={floor_conf:.3f}")

        # ---- Phase 2F [H_ROT_SANITY]: floor vs ceiling. After the (rotated) image is
        # upright, the real floor mask sits in the LOWER part of the frame; a wrong
        # rotation makes SegFormer mask the ceiling/wall (upper part). This is a pure
        # diagnostic on the existing mask + a flag sent to Unity's rotation probe;
        # it does NOT change segmentation/geometry. ----
        try:
            _fm = floor_mask
            _ys, _xs = np.nonzero(_fm)
            _H = _fm.shape[0]
            if len(_ys) > 0:
                _cy = float(_ys.mean()) / max(1, _H)                 # 0=top, 1=bottom
                _br = float((_ys > _H / 2).sum()) / float(len(_ys))  # bottom-half fraction
                _ma = float(len(_ys)) / float(_fm.shape[0] * _fm.shape[1])
            else:
                _cy, _br, _ma = 0.0, 0.0, 0.0
        except Exception:
            _cy, _br, _ma = 0.0, 0.0, 0.0
        self._last_centroid_y   = _cy
        self._last_bottom_ratio = _br
        self._last_orient_ok    = (_cy >= 0.45 and _br >= 0.40)
        _verdict = "ACCEPT" if self._last_orient_ok else "REJECT"
        _reason  = "floor_lower" if self._last_orient_ok else "floor_on_top_or_wall"
        print(f"[H_ROT_SANITY] rot={rotation_deg} floorCentroidY={_cy:.2f} "
              f"bottomRatio={_br:.2f} maskArea={_ma:.3f} verdict={_verdict} reason={_reason}")
        print(f"[FLOOR_FUSION] semantic={seg_stats.get('semantic_agreement', 0.0)} "
              f"geometry={seg_mode} agreement={seg_stats.get('semantic_agreement', 0.0)} "
              f"high_conf_px={seg_stats.get('semantic_high_pixels', 0)} final={seg_mode}")

        # Loose scan gate
        if floor_ratio < self._floor_min_ratio:
            dbg["reject_stage"]  = "seg"
            dbg["reject_reason"] = "insufficient_floor_pixels"
            print(f"[P5_REJECT] id={frame_id} stage=seg reason=insufficient_floor_pixels")
            print(f"[FLOOR_REJECT] reason=low_conf stage=seg ratio={floor_ratio:.4f}")
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
        print(f"[DEPTH_BACKPROJECT] points={len(points_3d)}")

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
            print(f"[RANSAC_PLANE] valid=False reason={reason}")
            print(f"[FLOOR_REJECT] reason=bad_plane stage=plane detail={reason}")
            return self._handle_invalid_floor(frame_id, f"plane fit failed ({reason})", dbg)

        inlier_points   = plane_stats.get("inlier_points", None)
        normal_cam_list = plane["normal"]
        _n_inliers = len(inlier_points) if inlier_points is not None else 0
        print(f"[RANSAC_PLANE] valid=True inliers={_n_inliers} "
              f"rmse={plane['rmse']:.4f} inlier_ratio={plane['inlier_ratio']:.3f} "
              f"normal=({normal_cam_list[0]:.3f},{normal_cam_list[1]:.3f},{normal_cam_list[2]:.3f})")
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

        # ---- Normal conversion + matrix-order resolution (UNIFIED) ----
        # CONTRACT: Unity serialises camera_to_world COLUMN-MAJOR (see Unity
        # Assets/Scripts/Protocol/AIFramePacket.cs:25 "16 floats, column-major"
        # and SetCameraToWorld at :65) and RE-APPLIES that same matrix to the
        # boundary it receives (AIGuardianReceiver.ConvertToWorldWithPose:1239
        # `framePose.rotation * camPoint`). Therefore M_col (reshape order='F')
        # IS the true Unity pose. We compute BOTH reshape orders, but score them
        # in the SAME camera-space convention the geometry is built in, then drive
        # the gate AND all world geometry from ONE transform — so the plane we
        # APPROVE is the exact plane we BUILD (no "approve with row, build with
        # col" divergence, no col(forced)).
        normal_cv = np.array(plane["normal"], dtype=np.float64)

        # OpenCV cam (X=right, Y=down, Z=forward) -> Unity cam (X=right, Y=UP,
        # Z=forward): flip Y ONLY. This is exactly what Unity re-applies to
        # boundary_camera and what boundary_generator / floor_accumulator use.
        # The OLD gate flipped Y AND Z (a DIFFERENT camera space than the geometry
        # was built in) — that mismatch is why score_row spuriously beat score_col
        # on real floors and made the gate disagree with the world normal.
        normal_cam_pts = np.array([normal_cv[0], -normal_cv[1], normal_cv[2]], dtype=np.float64)

        pose_arr  = np.array(pose, dtype=np.float64)
        M_col_raw = pose_arr.reshape(4, 4, order='F')   # column-major = TRUE Unity pose
        M_row     = pose_arr.reshape(4, 4, order='C')   # row-major (diagnostic only)

        # Phase 2M (req 6): optionally fold Unity's optical-axis conversion
        # correction into the pose IN PYTHON. The image was rotated for SegFormer
        # but camera_to_world is the un-rotated pose; Unity normally fixes this by
        # rotating the received boundary about the optical axis (corrRot) before
        # the pose. Doing the SAME here (world = R · Rz(corr) · camPoint + t) makes
        # Python's world == the true world, so the accumulator no longer smears the
        # floor across frames in landscape. When applied we set optical_axis_corrected
        # on the outgoing boundary and Unity SKIPS its corrRot (no double-correction).
        # corr_deg is 0 in portrait → portrait is byte-for-byte unchanged.
        corr_deg = 0
        if metadata is not None:
            try:
                corr_deg = int(metadata.get("conversion_correction_degrees", 0))
            except (TypeError, ValueError):
                corr_deg = 0
        self._optical_corrected = False
        if APPLY_OPTICAL_AXIS_CORRECTION and (corr_deg % 360) != 0:
            Rz4   = self._optical_axis_R4(OPTICAL_AXIS_CORRECTION_SIGN * corr_deg)
            M_col = M_col_raw @ Rz4
            self._optical_corrected = True
            print(f"[OPTICAL_AXIS_CORRECTION] id={frame_id} applied corr_deg={corr_deg} "
                  f"sign={OPTICAL_AXIS_CORRECTION_SIGN} (Unity skips corrRot)")
        else:
            M_col = M_col_raw

        R_col = M_col[:3, :3]
        R_row = M_row[:3, :3]

        def _up_normal(R):
            n = _normalize(R @ normal_cam_pts)
            return (-n if n[1] < 0.0 else n)
        nw_col = _up_normal(R_col)            # floor normal in world via TRUE pose
        nw_row = _up_normal(R_row)            # via transpose (diagnostic)
        score_col = float(abs(nw_col[1]))     # |normal . world_up| — true flatness
        score_row = float(abs(nw_row[1]))

        # ONE transform for the gate AND the geometry: the TRUE column-major pose.
        # (Switching geometry to 'row' when score_row wins — the literal request —
        # would be WRONG: row is the transpose, not the pose Unity re-applies, so
        # it builds a rotated world the device then re-rotates onto the wall. The
        # real cause of "row wins" was the wrong-Z gate convention, now fixed, so
        # col is correct AND honest. The Unity-space y_spread gate below is the
        # safety net if this assumption is ever violated.)
        selected_order    = "col"
        M                 = M_col
        normal_world      = nw_col
        horizontal_score  = score_col
        M_geom            = M_col             # geometry uses the SAME transform
        normal_world_geom = normal_world      # gate normal == geometry normal

        dbg["normal_cv"]             = [round(v, 4) for v in normal_cv.tolist()]
        dbg["normal_unity_cam"]      = [round(v, 4) for v in normal_cam_pts.tolist()]
        dbg["score_col"]             = round(score_col, 4)
        dbg["score_row"]             = round(score_row, 4)
        dbg["horizontal_score"]      = round(horizontal_score, 4)
        dbg["selected_matrix_order"] = selected_order
        dbg["plane_normal_world"]    = [round(v, 4) for v in normal_world.tolist()]
        dbg["normal_world_geom"]     = [round(v, 4) for v in normal_world_geom.tolist()]
        self._last_horizontal_score  = float(horizontal_score)  # geometry rotation probe

        _row_note = ("row_would_win_SUSPECT" if score_row > score_col + 0.05
                     else "col_dominant")
        print(f"[ROT_PAIR_SELECTED] id={frame_id} order={selected_order} "
              f"score_col={score_col:.4f} score_row={score_row:.4f} "
              f"horizontal_score={horizontal_score:.4f} "
              f"normal_world=({normal_world[0]:.3f},{normal_world[1]:.3f},{normal_world[2]:.3f}) "
              f"note={_row_note}")
        print(f"[PY_GEOM_POSE] order=col normal_world_geom=({normal_world_geom[0]:.3f},"
              f"{normal_world_geom[1]:.3f},{normal_world_geom[2]:.3f}) "
              f"selected_for_gate={selected_order} (gate==geometry)")

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
            print(f"[FLOOR_REJECT] reason=wall_or_vertical "
                  f"horizontal_score={horizontal_score:.3f} "
                  f"normal_world=({normal_world[0]:.3f},{normal_world[1]:.3f},{normal_world[2]:.3f})")
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
            print(f"[FLOOR_REJECT] reason=too_small stage=boundary detail={reason}")
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

        # Phase 2F: headset lock tolerance. In the VR headset the phone looks DOWN at
        # the floor at a steep angle, so a real flat floor scores ~0.50 horizontal —
        # just under the 0.55 handheld lock gate, so it never locks. Relax the
        # horizontal/confidence lock thresholds ONLY for headset frames (headset_mode
        # is sent on the frame packet). NormalAR is unchanged. 0.45 still sits well
        # above walls/ceiling (<0.40, already rejected at the scan gate).
        _headset_mode = bool(metadata.get("headset_mode", False)) if metadata else False
        _lock_min_h = (HEADSET_LOCK_MIN_HORIZONTAL_SCORE if _headset_mode
                       else self._lock_min_horizontal_score)
        _lock_min_conf = (HEADSET_LOCK_MIN_CONFIDENCE if _headset_mode
                          else self._lock_min_confidence)
        if _headset_mode:
            print(f"[H_LOCK_GATE] mode=headset hThresh={_lock_min_h:.2f} "
                  f"h={horizontal_score:.3f} confThresh={_lock_min_conf:.2f} "
                  f"conf={confidence:.3f} "
                  f"verdict={'PASS' if (horizontal_score >= _lock_min_h and confidence >= _lock_min_conf) else 'FAIL'}")

        # Strict lock gate
        passes_lock_gate = (
            horizontal_score        >= _lock_min_h and
            ratio_after             >= self._lock_min_floor_ratio and
            plane["inlier_ratio"]   >= self._lock_min_inlier_ratio and
            plane["rmse"]           <= self._lock_max_rmse and
            confidence              >= _lock_min_conf
        )

        lock_allowed      = passes_lock_gate
        lock_block_reason = ""

        if used_fallback and not self._allow_fallback_lock:
            lock_allowed      = False
            lock_block_reason = "fallback_cannot_lock"
            # Phase 10.1: spec log — every rejected candidate gets a reason field
            # so the operator can grep [LOCK_CANDIDATE_REJECT] to understand why
            # the pool isn't converging.
            print(f"[LOCK_CANDIDATE_REJECT] frame={frame_id} reason=fallback_cannot_lock")
        elif not passes_lock_gate:
            reasons = []
            if horizontal_score       < _lock_min_h:
                reasons.append(f"h={horizontal_score:.3f}<{_lock_min_h}")
            if ratio_after            < self._lock_min_floor_ratio:
                reasons.append(f"fr={ratio_after:.3f}<{self._lock_min_floor_ratio}")
            if plane["inlier_ratio"]  < self._lock_min_inlier_ratio:
                reasons.append(f"ir={plane['inlier_ratio']:.2f}<{self._lock_min_inlier_ratio}")
            if plane["rmse"]          > self._lock_max_rmse:
                reasons.append(f"rmse={plane['rmse']:.4f}>{self._lock_max_rmse}")
            if confidence             < _lock_min_conf:
                reasons.append(f"conf={confidence:.3f}<{_lock_min_conf}")
            lock_block_reason = "below_lock_gate: " + " | ".join(reasons)
            # Phase 10.1: spec log per rejected candidate.
            print(f"[LOCK_CANDIDATE_REJECT] frame={frame_id} reason=below_lock_gate "
                  f"details=({' '.join(reasons)})")

        dbg["lock_allowed"]        = lock_allowed
        dbg["lock_block_reason"]   = lock_block_reason

        # Reset consecutive-invalid counter (this frame was geometrically valid)
        self._consecutive_invalid = 0

        # ==================================================================
        # Phase 14: WORLD-space multi-frame floor accumulation.
        #
        # Every frame that clears the per-frame lock gate is a "trusted" frame.
        # Its RANSAC inlier floor points are accumulated, in WORLD coordinates,
        # into a persistent occupancy grid (req 2-5). The locked cubic is built
        # from that accumulated area (req 6), capped at 3×3 m (req 7), and is
        # only emitted once center/rotation/floorY/width/depth are stable
        # (req 8). After lock the rectangle is frozen (req 9-10).
        # ==================================================================
        if lock_allowed and inlier_points is not None and len(inlier_points) >= 10:
            acc_rect = self._floor_accum.add_frame(
                inlier_points_cv = inlier_points,
                M                = M_geom,          # TRUE column-major pose
                plane            = plane,
                normal_world     = normal_world_geom,
                confidence       = confidence,
                frame_id         = frame_id,
            )
        else:
            acc_rect = self._floor_accum.current_rect()

        stab = self._floor_accum.stability()
        dbg["candidate_pool_count"]    = self._floor_accum.frame_count
        dbg["candidate_pool_required"] = self._accum_min_frames
        dbg["accum_frames"]   = self._floor_accum.frame_count
        dbg["accum_cells"]    = self._floor_accum.cell_count
        dbg["accum_stable"]   = bool(stab.get("stable", False))
        dbg["accum_d_center"] = round(stab.get("d_center", -1.0), 3)
        dbg["accum_d_yaw"]    = round(stab.get("d_yaw", -1.0), 2)
        dbg["accum_d_w"]      = round(stab.get("d_w", -1.0), 3)
        dbg["accum_d_d"]      = round(stab.get("d_d", -1.0), 3)
        dbg["accum_d_floory"] = round(stab.get("d_floory", -1.0), 3)
        if acc_rect is not None:
            dbg["accum_width"]   = round(acc_rect["width"], 3)
            dbg["accum_depth"]   = round(acc_rect["depth"], 3)
            dbg["accum_area"]    = round(acc_rect["area"], 3)
            dbg["accum_floor_y"] = round(acc_rect["floor_y"], 3)
            dbg["accum_yaw"]     = round(acc_rect["yaw_deg"], 2)

        print(f"[FLOOR_ACCUMULATOR] id={frame_id} accepted={lock_allowed} "
              f"frames={self._floor_accum.frame_count}/{self._accum_min_frames} "
              f"cells={self._floor_accum.cell_count} "
              f"area={dbg.get('accum_area', 0.0)} "
              f"stable={stab.get('stable', False)} "
              f"d_center={stab.get('d_center', -1.0):.3f} d_yaw={stab.get('d_yaw', -1.0):.1f} "
              f"d_w={stab.get('d_w', -1.0):.3f} d_d={stab.get('d_d', -1.0):.3f} "
              f"d_floory={stab.get('d_floory', -1.0):.3f}")

        # Area-growth gate trace: keep the LARGEST floor; only lock once the area
        # has stopped growing for FLOOR_ACCUM_AREA_GROWTH_FRAMES accepted frames.
        if lock_allowed and acc_rect is not None:
            print(f"[AREA_GROWTH] currentArea={self._floor_accum.current_area:.2f} "
                  f"bestArea={self._floor_accum.best_area:.2f} "
                  f"stableFrames={self._floor_accum.growth_stable_frames}/"
                  f"{self._floor_accum.area_growth_frames}")
            if (self._floor_accum.growth_stable_frames < self._floor_accum.area_growth_frames
                    and stab.get("stable", False)):
                # geometry is steady but the floor is still expanding — hold the
                # lock and keep scanning for a larger area.
                print(f"[LOCK_WAIT_LARGER_AREA] area still growing "
                      f"current={self._floor_accum.current_area:.2f} "
                      f"best={self._floor_accum.best_area:.2f} "
                      f"plateau={self._floor_accum.growth_stable_frames}/"
                      f"{self._floor_accum.area_growth_frames}")

        # ================================================================
        # Phase 1 [FLAT_FLOOR_CUBE]: the cube is built ONLY from the flat,
        # world-space accumulated rectangle (every corner sits at floorY). We
        # NEVER fall back to the per-frame camera-space boundary — that rectangle
        # is laid out in the tilted CAMERA X/Z plane, so on a hand-held (tilted)
        # phone it stands up the wall. That is exactly the reported failure:
        # boundary world_y spanning -0.07 .. 1.75 m and the cube climbing the
        # wall. Until enough trusted floor has accumulated to form a rectangle,
        # we stay in 'scanning' and send NO cube at all.
        # ================================================================
        if acc_rect is None:
            if not lock_allowed:
                print(f"[P7_CANDIDATE] id={frame_id} lock_allowed=False "
                      f"reason={lock_block_reason}")
            self._boundary_state  = "scanning"
            dbg["boundary_state"] = "scanning"
            t_total = (time.time() - t_total_start) * 1000
            dbg["total_ms"] = round(t_total, 1)
            self._write_csv_row(dbg)
            return self._make_status_message(
                frame_id, "Scanning floor... building trusted area", dbg)

        # We have a trusted, flat, world-space rectangle.
        acc_cx, acc_cz = acc_rect["center"]
        print(f"[FLOOR_CANDIDATE] w={acc_rect['width']:.2f} d={acc_rect['depth']:.2f} "
              f"area={acc_rect['area']:.2f} center=({acc_cx:.2f},{acc_cz:.2f}) "
              f"yaw={acc_rect['yaw_deg']:.1f} floorY={acc_rect['floor_y']:.3f}")

        send_boundary = {
            "boundary_camera": self._world_corners_to_camera(acc_rect["corners_world"], M_geom),
            "boundary_world":  [[round(float(v), 4) for v in c] for c in acc_rect["corners_world"]],
            "width":           round(acc_rect["width"], 3),
            "depth":           round(acc_rect["depth"], 3),
            "confidence":      round(acc_rect.get("confidence", confidence), 3),
            "source":          "accumulated_floor",
        }
        _sb_c = send_boundary["boundary_camera"]
        print(f"[BOUNDARY_JSON] id={frame_id} state=preview source=accumulated_floor "
              f"w={send_boundary['width']:.2f} d={send_boundary['depth']:.2f} "
              f"order=col corner0_cam={_sb_c[0] if _sb_c else None} "
              f"corner0_world={send_boundary['boundary_world'][0]}")

        # ---- Lock decision (req 3,4,7,8) ----
        # AUTO: rolling accumulator says placement is steady + area plateaued/enough
        #       confident frames (floor_accumulator.ready_to_lock).
        # MANUAL: the user pressed "Lock Now" (UNITY_REQUEST_LOCK) and the current
        #       preview clears the safe minimums (can_lock_now).
        # BOTH still require the per-frame lock gate (lock_allowed) AND the
        # Unity-space coordinate sanity check — manual skips the WAIT, never SAFETY.
        manual_pending = (self._manual_lock_enabled
                          and self._unity_lock_listener is not None
                          and self._unity_lock_listener.manual_lock_pending)
        auto_ready   = self._floor_accum.ready_to_lock()
        manual_ready = manual_pending and self._floor_accum.can_lock_now()
        if manual_pending:
            print(f"[MANUAL_LOCK] id={frame_id} pending lock_allowed={lock_allowed} "
                  f"can_lock_now={self._floor_accum.can_lock_now()} "
                  f"frames={self._floor_accum.frame_count} cells={self._floor_accum.cell_count} "
                  f"area={acc_rect['area']:.2f} (will lock as soon as safe)")
        _lock_trigger = "manual" if (manual_ready and not auto_ready) else "auto"
        if (lock_allowed and (auto_ready or manual_ready)
                and self._coord_sanity_ok(acc_rect, M_geom, horizontal_score, frame_id, dbg)):
            print(f"[LOCK_TRIGGER] id={frame_id} source={_lock_trigger}")
            if manual_ready and self._unity_lock_listener is not None:
                self._unity_lock_listener.consume_manual_lock()   # clear latch on success
            frozen = self._floor_accum.lock()
            cam_corners = self._world_corners_to_camera(frozen["corners_world"], M_geom)
            locked_boundary = {
                "boundary_camera": cam_corners,
                "boundary_world":  [[round(float(v), 4) for v in c] for c in frozen["corners_world"]],
                "width":           round(frozen["width"], 3),
                "depth":           round(frozen["depth"], 3),
                "confidence":      round(frozen.get("confidence", confidence), 3),
                "source":          "accumulated_floor_locked",
            }

            # Phase 1 contract trace — the EXACT world corners we send. p0..p3
            # must all share ~the same Y (flat on the floor); Unity must echo
            # these in [UNITY_LOCK_RECEIVED].
            _wc = frozen["corners_world"]
            _ys = [c[1] for c in _wc]
            print(f"[PY_LOCK_CORNERS] "
                  f"p0=({_wc[0][0]:.3f},{_wc[0][1]:.3f},{_wc[0][2]:.3f}) "
                  f"p1=({_wc[1][0]:.3f},{_wc[1][1]:.3f},{_wc[1][2]:.3f}) "
                  f"p2=({_wc[2][0]:.3f},{_wc[2][1]:.3f},{_wc[2][2]:.3f}) "
                  f"p3=({_wc[3][0]:.3f},{_wc[3][1]:.3f},{_wc[3][2]:.3f}) "
                  f"floorY={frozen['floor_y']:.3f} y_spread={max(_ys)-min(_ys):.3f} "
                  f"width={frozen['width']:.3f} depth={frozen['depth']:.3f} "
                  f"yaw={frozen['yaw_deg']:.1f}")
            print(f"[PY_SENT_CAMERA_CORNERS] "
                  f"c0={cam_corners[0]} c1={cam_corners[1]} "
                  f"c2={cam_corners[2]} c3={cam_corners[3]} (pose=M_col)")

            # Placement mode determines whether we auto-lock or hand Unity a
            # ready_to_confirm (user presses Build Playing Area).
            final_state = "ready_to_confirm" if self._placement_mode == "user_confirm" else "locked"
            self._boundary_state     = final_state
            self._floor_confirmed    = True
            dbg["boundary_state"]    = final_state
            dbg["locked"]            = (final_state == "locked")
            dbg["lock_source_frame"] = frame_id
            dbg["lock_source"]       = "world_accumulation"
            frz_cx, frz_cz = frozen["center"]

            # Record OUR lock frame in metrics so the session summary shows a real
            # Lock Frame even if the Unity→Python ACK is lost — the Unity ACK, when
            # it arrives, overwrites this with unity_accepted.
            if self._metrics.get("lock_frame_id") is None:
                self._metrics["lock_frame_id"] = frame_id
                self._metrics["lock_time"] = time.time()
                self._metrics["lock_source"] = "python_accumulator"

            self._locked_boundary_msg = self._build_message(
                frame_id, locked_boundary, plane,
                horizontal_score, ratio_after, final_state,
                self._floor_accum.frame_count, self._accum_min_frames,
                dbg, used_fallback,
                plane_patch_camera=plane_patch_camera,
                floor_candidates=None,
            )
            print(f"[STABLE_LOCK] center=({frz_cx:.2f},{frz_cz:.2f}) "
                  f"size=({frozen['width']:.2f}x{frozen['depth']:.2f}) "
                  f"floorY={frozen['floor_y']:.3f} yaw={frozen['yaw_deg']:.1f} "
                  f"reason=stable_largest_floor "
                  f"d_center={stab.get('d_center', -1.0):.3f} d_yaw={stab.get('d_yaw', -1.0):.1f} "
                  f"d_w={stab.get('d_w', -1.0):.3f} d_d={stab.get('d_d', -1.0):.3f} "
                  f"d_floorY={stab.get('d_floory', -1.0):.3f}")
            print(f"[FLOOR_ACCUM_LOCK] state={final_state} "
                  f"w={frozen['width']:.2f} d={frozen['depth']:.2f} "
                  f"area={frozen['area']:.2f} floorY={frozen['floor_y']:.3f} "
                  f"yaw={frozen['yaw_deg']:.1f} frames={self._floor_accum.frame_count} "
                  f"cells={frozen['cell_count']} frame={frame_id}")
            print(f"[LOCK_AREA_SELECTED] width={frozen['width']:.2f} depth={frozen['depth']:.2f} "
                  f"area={frozen['area']:.2f} cap={self._accum_max_size:.1f}x{self._accum_max_size:.1f} "
                  f"source=world_accumulation")
            print(f"[LOCKED_CUBE] id={frame_id} state={final_state} source=accumulated_floor "
                  f"w={frozen['width']:.3f} d={frozen['depth']:.3f} "
                  f"floorY={frozen['floor_y']:.3f} yaw={frozen['yaw_deg']:.1f} "
                  f"y_spread={max(_ys)-min(_ys):.3f} cap={self._accum_max_size:.1f} "
                  f"frames={self._floor_accum.frame_count} order=col")
            print(f"[BOUNDARY_JSON] id={frame_id} state={final_state} "
                  f"source=accumulated_floor_locked "
                  f"w={locked_boundary['width']:.2f} d={locked_boundary['depth']:.2f} "
                  f"order=col corner0_cam={cam_corners[0] if cam_corners else None} "
                  f"corner0_world={locked_boundary['boundary_world'][0]}")
            self._write_csv_row(dbg)
            return self._locked_boundary_msg

        # ---- Not locked yet: preview the accumulating FLAT cube (yellow) ----
        if not lock_allowed:
            print(f"[P7_CANDIDATE] id={frame_id} lock_allowed=False "
                  f"reason={lock_block_reason}")
        self._boundary_state  = "preview"
        dbg["boundary_state"] = "preview"

        # [SCAN_PROGRESS] — why we're still yellow, and what's left before green.
        _sp = self._scan_progress()
        _still_growing = self._floor_accum.growth_stable_frames < self._floor_accum.area_growth_frames
        print(f"[SCAN_PROGRESS] id={frame_id} {_sp['percent']:.0f}% "
              f"area={_sp['area']:.2f}/{FLOOR_ACCUM_AUTO_LOCK_MIN_AREA_M2:.1f}m2(autolock) "
              f"best={_sp['best_area']:.2f} frames={_sp['frames']}/{_sp['frames_req']} "
              f"cells={_sp['cells']}/{_sp['cells_req']} place_stable={_sp['place_stable']} "
              f"growing={_still_growing} manual_ok={_sp['manual_ok']} "
              f"verdict={'GREEN_WHEN_STABLE' if _sp['area'] >= FLOOR_ACCUM_AUTO_LOCK_MIN_AREA_M2 else 'GROW_MORE'}")

        t_total = (time.time() - t_total_start) * 1000
        dbg["total_ms"] = round(t_total, 1)
        print(f"[P7_PERF] id={frame_id} total_ms={t_total:.1f}")

        self._write_csv_row(dbg)

        return self._build_message(
            frame_id, send_boundary, plane, horizontal_score,
            ratio_after, self._boundary_state,
            self._floor_accum.frame_count, self._accum_min_frames,
            dbg, used_fallback,
            plane_patch_camera=plane_patch_camera,
            floor_candidates=None,
        )

    # ------------------------------------------------------------------
    # Phase 7C: Frame caching for hand tracking
    # ------------------------------------------------------------------

    def _resolve_input_rotation(self, metadata):
        """Phase 1.5: clockwise degrees to make the incoming sensor frame upright
        for SegFormer, for the current device orientation. Prefers the value Unity
        tagged on the packet (``input_rotation_degrees``); falls back to the fixed
        config default when the field is absent or invalid. This is orientation
        preprocessing only — the floor-detection algorithm is never changed.
        Returns (degrees, source) where source is "packet" or "config".
        """
        rot = metadata.get("input_rotation_degrees", None) if metadata else None
        try:
            rot = int(rot)
        except (TypeError, ValueError):
            rot = None
        if rot in (0, 90, 180, 270):
            return rot, "packet"
        return INPUT_ROTATION_DEGREES, "config"

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

        # Apply input rotation. Phase 1.5: prefer the per-frame rotation Unity
        # tagged for the current device orientation (portrait vs landscape); fall
        # back to the fixed config default so older clients behave exactly as
        # before. Orientation preprocessing only — floor logic is unchanged.
        rotation_deg, self._last_rotation_src = self._resolve_input_rotation(metadata)
        self._last_rotation_deg = rotation_deg
        rotated = frame
        if rotation_deg != 0:
            rotated, fx, fy, cx, cy = rotate_frame_and_intrinsics(
                frame, fx, fy, cx, cy, rotation_deg)

        # Phase 3 [HAND_FPS]: post-lock, depth is the bottleneck (it ran every
        # hand frame → ~3 FPS). Either fully stop it (POST_LOCK_STOP_DEPTH) or —
        # the default — THROTTLE it to POST_LOCK_HAND_DEPTH_FPS so MediaPipe runs
        # fast on the in-between frames while the fingertip 3D still refreshes a
        # few times a second. The Unity side holds the last valid 3D for 0.7 s,
        # so the ray/glove stay smooth between depth refreshes.
        if self._post_lock_mode and POST_LOCK_STOP_DEPTH:
            # Phase 3 [HAND_FPS]: depth FULLY OFF post-lock — the whole GPU goes to
            # MediaPipe (>=15 Hz hand). depth_map=None signals hand_tracker.detect
            # to reconstruct the 3D hand from MediaPipe's own landmarks (no
            # DepthAnything call), so the glove/ray stay valid AND the hand is fast.
            depth_map = None
        elif self._post_lock_mode:
            now = time.time()
            base_interval = 1.0 / max(0.5, POST_LOCK_HAND_DEPTH_FPS)
            # Phase 3 [HAND_SPEEDUP]: adaptive throttle. Depth and MediaPipe share
            # one GPU, so every depth.predict steals time from the hand. If the
            # measured hand rate has fallen below the floor, refresh depth half as
            # often so MediaPipe recovers real-time (the Unity side keeps the last
            # valid 3D for ~0.7 s, so the ray/glove stay alive between refreshes).
            hand_hz = self._compute_rolling_fps(self._metrics["_hand_timestamps"])
            interval = base_interval
            throttled = False
            if 1.0 < hand_hz < POST_LOCK_HAND_MIN_HZ:
                interval = base_interval * 2.0
                throttled = True
            if (self._last_depth_map is None
                    or now - self._last_depth_refresh_time >= interval):
                depth_map = self._depth_model.predict(rotated)
                self._last_depth_refresh_time = now
                if self._metrics["idle_frames"] % 50 == 0:
                    print(f"[P3_DEPTH_THROTTLE] hand_hz={hand_hz:.1f} "
                          f"min_hz={POST_LOCK_HAND_MIN_HZ:.0f} "
                          f"depth_interval_ms={interval*1000:.0f} throttled={throttled}")
            else:
                depth_map = self._last_depth_map     # reuse — skip depth this frame (fast)
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
        preview, ready_to_confirm, locked) UNLESS Unity has notified us that
        the hand pipeline should stop (Phase 10.6 — HAND_PIPELINE_OFF event
        fires when the patient presses the Toshfa button and enters the gym).
        """
        # Phase 10.6: stop running MediaPipe Hand if Unity asked us to.
        # The Toshfa Pose session on the laptop is the only CV source needed
        # from that point. Saves GPU + battery; avoids competing inference.
        if (self._unity_lock_listener is not None
                and self._unity_lock_listener.hand_pipeline_off):
            return

        if self._hand_tracker is None:
            return

        # Phase 3: depth may be None post-lock (DepthAnything disabled). That is
        # NOT a skip — hand_tracker.detect reconstructs the 3D hand from MediaPipe.

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

        # Ensure depth map matches frame dimensions.
        # Phase 3: depth may be None post-lock (DepthAnything disabled). Only
        # resize when we actually have a map; detect() reconstructs from MediaPipe
        # when depth is None.
        depth = self._last_depth_map
        if depth is not None and depth.shape[:2] != rgb.shape[:2]:
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

    @staticmethod
    def _optical_axis_R4(deg):
        """4×4 rotation about the camera optical axis (Unity-cam +Z), matching
        Unity's Quaternion.AngleAxis(deg, Vector3.forward) applied to a cam point."""
        a = float(deg) * np.pi / 180.0
        c, s = float(np.cos(a)), float(np.sin(a))
        R = np.eye(4, dtype=np.float64)
        R[0, 0] = c; R[0, 1] = -s
        R[1, 0] = s; R[1, 1] = c
        return R

    def _scan_progress(self) -> dict:
        """Req 5: scan-progress summary for the Unity UI. percent is the min of the
        three independent requirements (frames, cells, area) so the bar only reads
        100% when an auto-lock is actually achievable; place_stable / manual_ok let
        the UI show "hold steady" vs "Lock Now ready"."""
        acc = self._floor_accum
        rect = acc.current_rect()
        area = float(rect["area"]) if rect else 0.0
        f_req = max(1, self._accum_min_frames)
        c_req = max(1, acc._min_cells) if hasattr(acc, "_min_cells") else 150
        a_req = acc._min_area if hasattr(acc, "_min_area") else 1.0
        p_frames = min(1.0, acc.frame_count / f_req)
        p_cells  = min(1.0, acc.cell_count / c_req)
        p_area   = min(1.0, area / a_req) if a_req > 0 else 1.0
        percent  = round(100.0 * min(p_frames, p_cells, p_area), 1)
        try:
            place_stable = bool(acc.stability().get("place_stable", False))
        except Exception:
            place_stable = False
        return {
            "percent":       percent,
            "frames":        acc.frame_count,
            "frames_req":    f_req,
            "cells":         acc.cell_count,
            "cells_req":     c_req,
            "area":          round(area, 2),
            "area_req":      round(float(a_req), 2),
            "place_stable":  place_stable,
            "growth_plateau": acc.growth_stable_frames,
            "growth_req":    acc.area_growth_frames,
            "manual_ok":     bool(self._manual_lock_enabled and acc.can_lock_now()),
            "best_area":     round(acc.best_area, 2),
        }

    def _coord_sanity_ok(self, rect, M_geom, horizontal_score, frame_id, dbg):
        """[COORD_SANITY] req 7: server-side last-line check, evaluated as part of
        the lock CONDITION (never forces a lock, can only block one).

        The accumulated rectangle is flat by construction (all four corners sit at
        floor_y), so the y_spread here is a STRUCTURAL guard. The geometrically
        meaningful checks are:
          * floor sits BELOW the camera by a human-room amount (rejects a
            ceiling/wall captured under a wrong image rotation), using the pose
            translation as the camera world Y, and
          * world 'up' is actually up (horizontal_score — same number the gate
            approved on, now that gate == geometry).
        The AUTHORITATIVE max(y)-min(y) corner spread is re-checked Unity-side in
        AIGuardianReceiver.ValidateCoordinateSanity AFTER the device re-applies its
        true pose — that is the only place the real device transform is known."""
        try:
            cam_y    = float(np.asarray(M_geom, dtype=np.float64).reshape(4, 4)[1, 3])
            floor_y  = float(rect["floor_y"])
            ys       = [float(c[1]) for c in rect["corners_world"]]
            y_spread = max(ys) - min(ys)
            below    = cam_y - floor_y
            flat_ok  = y_spread <= UNITY_LOCK_MAX_Y_SPREAD_M
            floor_ok = 0.30 <= below <= 2.50
            up_ok    = horizontal_score >= self._lock_min_horizontal_score
            sane     = flat_ok and floor_ok and up_ok
            verdict  = ("PASS" if sane else
                        "REJECT:y_spread" if not flat_ok else
                        "REJECT:floor_not_below_cam" if not floor_ok else
                        "REJECT:not_horizontal")
            print(f"[COORD_SANITY] id={frame_id} y_spread={y_spread:.3f} "
                  f"maxSpread={UNITY_LOCK_MAX_Y_SPREAD_M:.2f} floorY={floor_y:.3f} "
                  f"camY={cam_y:.3f} belowCam={below:.2f} h={horizontal_score:.3f} "
                  f"verdict={verdict}")
            if dbg is not None:
                dbg["coord_y_spread"]  = round(y_spread, 3)
                dbg["coord_below_cam"] = round(below, 3)
                dbg["coord_sane"]      = bool(sane)
                if not sane:
                    dbg["lock_block_reason"] = f"coord_sanity:{verdict}"
            if not sane:
                print(f"[LOCK_CANDIDATE_REJECT] frame={frame_id} reason=coord_sanity {verdict}")
            return sane
        except Exception as e:
            print(f"[COORD_SANITY_ERR] {e}")
            return False   # fail-safe: never lock if we cannot verify

    def _world_corners_to_camera(self, corners_world, M):
        """Express WORLD corner points in THIS frame's Unity camera space via
        M⁻¹ (M maps Unity-cam → world). Unity then re-applies the frame-matched
        camera_to_world pose to recover the exact world rectangle, so the
        accumulated cubic needs no Unity-side changes. Falls back to the raw
        world points if the pose is singular (never expected for a rigid pose)."""
        try:
            M_inv = np.linalg.inv(np.asarray(M, dtype=np.float64).reshape(4, 4))
            out = []
            for c in corners_world:
                w = np.array([float(c[0]), float(c[1]), float(c[2]), 1.0], dtype=np.float64)
                cam = M_inv @ w
                out.append([round(float(cam[0]), 4), round(float(cam[1]), 4), round(float(cam[2]), 4)])
            return out
        except Exception as e:
            print(f"[ACCUM_CAM_TRANSFORM_ERR] {e}")
            return [[round(float(c[0]), 4), round(float(c[1]), 4), round(float(c[2]), 4)]
                    for c in corners_world]

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
            # Phase 2L (req 5): scan-progress for the Unity UI — how close the
            # accumulator is to an auto-lock, plus whether a manual lock is allowed.
            "scan_progress":   self._scan_progress(),
            # Phase 2M (req 6): when true, Python already applied the optical-axis
            # correction, so Unity must NOT re-apply corrRot to boundary_camera.
            "optical_axis_corrected": bool(getattr(self, "_optical_corrected", False)),
            # Phase 2F: floor-vs-ceiling sanity for Unity's rotation probe (diagnostic).
            "floor_orientation_ok": bool(getattr(self, "_last_orient_ok", False)),
            "floor_centroid_y":     round(float(getattr(self, "_last_centroid_y", 0.0)), 3),
            "floor_bottom_ratio":   round(float(getattr(self, "_last_bottom_ratio", 0.0)), 3),
            # Phase 2H: plane flatness for the GEOMETRY rotation probe.
            "horizontal_score":     round(float(getattr(self, "_last_horizontal_score", 0.0)), 4),
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
        # Req 9-10: once locked, NEVER recompute or update the boundary —
        # always replay the frozen locked message.
        if self._boundary_state in ("locked", "ready_to_confirm") and self._locked_boundary_msg is not None:
            return self._locked_boundary_msg

        self._consecutive_invalid += 1

        # Phase 14: the world accumulator persists through brief dropouts (a few
        # bad frames must NOT wipe the floor cloud — that is the whole point of
        # multi-frame accumulation). Only a sustained loss of tracking
        # (FLOOR_ACCUM_RESET_AFTER_INVALID consecutive invalid frames) resets it,
        # so that walking to a different spot starts a fresh scan.
        if (self._consecutive_invalid >= self._accum_reset_after_invalid
                and not self._floor_accum.locked
                and self._floor_accum.frame_count > 0):
            print(f"[FLOOR_ACCUM_RESET] reason=lost_tracking "
                  f"consecutive_invalid={self._consecutive_invalid} "
                  f"discarded_frames={self._floor_accum.frame_count} "
                  f"cells={self._floor_accum.cell_count}")
            self._floor_accum.reset()
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
            # Phase 2H: report plane flatness + floor orientation even on rejected
            # frames so the Unity GEOMETRY rotation probe can score every rotation.
            "horizontal_score":     round(float(getattr(self, "_last_horizontal_score", 0.0)), 4),
            "floor_orientation_ok": bool(getattr(self, "_last_orient_ok", False)),
            "floor_centroid_y":     round(float(getattr(self, "_last_centroid_y", 0.0)), 3),
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

        # Phase 10.2 (J): pull lock_frame_id from the Unity-side notification
        # so the summary reflects the lock Unity actually accepted (not just
        # what Python emitted). Falls back to existing value if listener never
        # received an event.
        if self._unity_lock_listener is not None:
            try:
                last = self._unity_lock_listener.get_last_lock()
                if last is not None:
                    m["lock_frame_id"]   = last.get("frame_id", m.get("lock_frame_id"))
                    m["lock_unity_time"] = last.get("unity_time", None)
                    m["lock_python_time"] = last.get("python_time", None)
                    m["lock_width"]      = last.get("width", None)
                    m["lock_depth"]      = last.get("depth", None)
                    m["lock_floorY"]     = last.get("floorY", None)
                    m["lock_source"]     = "unity_accepted"
                    print(f"[P10_LOCK_METRIC] frame={m['lock_frame_id']} "
                          f"size=({m['lock_width']}x{m['lock_depth']}) "
                          f"floorY={m['lock_floorY']}")
                else:
                    m["lock_source"] = "python_only_unity_never_acked"
            except Exception as _e:
                m["lock_source"] = f"lock_listener_err"

        # Phase 2 spec — explicit summary lock status. lock_frame_id is set either
        # by OUR accumulator lock (python_accumulator) or by the Unity ACK
        # (unity_accepted); only None if no lock happened this session.
        if m.get("lock_frame_id") is not None:
            print(f"[SUMMARY_LOCK_OK] lock_frame_id={m['lock_frame_id']} "
                  f"source={m.get('lock_source', 'unknown')}")
        else:
            print(f"[SUMMARY_LOCK_NONE] no lock recorded this session")

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
            f.write(f"Lock Source: {m.get('lock_source', 'N/A')}\n")
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