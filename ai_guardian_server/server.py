#!/usr/bin/env python3
# ===========================================================================
# server.py — AI Guardian Python Server (Phase 9)
# Main entry point. Starts UDP receiver, TCP server, and AI pipeline.
# Phase 9: Post-lock optimization + metrics export on shutdown.
# ===========================================================================

import signal
import sys
import time

from config import (
    UDP_HOST, UDP_PORT, TCP_HOST, TCP_PORT, MAX_UDP_PACKET,
    DUMMY_SEND_INTERVAL,
    # Depth
    DEPTH_ENABLED, DEPTH_TARGET_FPS, DEPTH_MODEL_NAME,
    DEPTH_ALLOW_DUMMY, DEPTH_USE_METRIC_INDOOR, DEPTH_TRY_DEPTH_PRO,
    # Phase 5
    BOUNDARY_MODE, ALLOW_DUMMY_GUARDIAN,
    SEG_MODEL_NAME, PIPELINE_TARGET_FPS,
    MIN_DEPTH_M, MAX_DEPTH_M, MAX_BACKPROJECT_POINTS,
    RANSAC_ITERATIONS, RANSAC_INLIER_THRESHOLD_M,
    PLANE_MIN_INLIERS, PLANE_MIN_INLIER_RATIO, PLANE_MAX_RMSE_M,
    BOUNDARY_WIDTH_M, BOUNDARY_DEPTH_M,
    BOUNDARY_NEAR_Z_M, BOUNDARY_FAR_Z_M, BOUNDARY_SAFETY_MARGIN_M,
    FLOOR_MIN_PIXEL_RATIO,
    LOWER_IMAGE_ROI, PLANE_MIN_HORIZONTAL_SCORE,
    # Lock gate
    LOCK_MIN_HORIZONTAL_SCORE, LOCK_MIN_FLOOR_RATIO,
    LOCK_MIN_INLIER_RATIO, LOCK_MAX_RMSE, LOCK_MIN_CONFIDENCE,
    # Fallback lock policy
    ALLOW_FALLBACK_LOCK,
    # Candidate pool
    CANDIDATE_POOL_SIZE, MAX_CONSECUTIVE_INVALID_BEFORE_POOL_RESET,
    # Legacy stability
    STABLE_REQUIRED, STABLE_MAX_ANGLE_DIFF, STABLE_MAX_DRIFT_M,
    STABLE_MIN_CONFIDENCE, STABLE_MAX_RMSE, STABLE_MIN_INLIER_RATIO,
    # Phase 6 Guardian fixed size
    GUARDIAN_FIXED_SIZE_ENABLED, GUARDIAN_WIDTH_M, GUARDIAN_DEPTH_M,
    GUARDIAN_CENTER_X_M, GUARDIAN_CENTER_Z_M, GUARDIAN_FLOOR_LIFT_M,
    # Phase 7B
    HAND_TRACKING_ENABLED, HAND_TRACKING_TARGET_FPS,
    HAND_MIN_DETECTION_CONF, HAND_MIN_TRACKING_CONF, HAND_MAX_NUM_HANDS,
    PLACEMENT_MODE,
    FLOOR_CANDIDATE_MIN_W, FLOOR_CANDIDATE_MIN_D,
    FLOOR_CANDIDATE_MAX_W, FLOOR_CANDIDATE_MAX_D,
    # Phase 9: Post-lock
    POST_LOCK_ENABLED, POST_LOCK_DIAGNOSTIC_FPS,
    POST_LOCK_STOP_DEPTH, POST_LOCK_STOP_SEGMENTATION,
)
from networking.udp_receiver import UDPReceiver
from networking.tcp_server   import TCPServer


def main():
    print("=" * 60)
    print("[P7_SERVER] starting AI Guardian Python server (Phase 7A)")
    print(f"[P7_SERVER] UDP: {UDP_HOST}:{UDP_PORT}")
    print(f"[P7_SERVER] TCP: {TCP_HOST}:{TCP_PORT}")
    print(f"[P7_SERVER] Depth: model={DEPTH_MODEL_NAME}")
    print(f"[P7_SERVER] Boundary: mode={BOUNDARY_MODE}")
    print(f"[P7_SERVER] Pipeline: target_fps={PIPELINE_TARGET_FPS}")
    print(f"[P7_SERVER] Guardian fixed size: enabled={GUARDIAN_FIXED_SIZE_ENABLED} "
          f"w={GUARDIAN_WIDTH_M} d={GUARDIAN_DEPTH_M}")
    print(f"[P7_SERVER] Placement: mode={PLACEMENT_MODE}")
    print(f"[P7_SERVER] Floor candidates: min=({FLOOR_CANDIDATE_MIN_W},{FLOOR_CANDIDATE_MIN_D}) "
          f"max=({FLOOR_CANDIDATE_MAX_W},{FLOOR_CANDIDATE_MAX_D})")
    print(f"[P7_SERVER] Hand tracking: enabled={HAND_TRACKING_ENABLED} "
          f"fps={HAND_TRACKING_TARGET_FPS} "
          f"det_conf={HAND_MIN_DETECTION_CONF} "
          f"track_conf={HAND_MIN_TRACKING_CONF}")
    print(f"[P7_SERVER] Scan gate: h>={PLANE_MIN_HORIZONTAL_SCORE} "
          f"floor>={FLOOR_MIN_PIXEL_RATIO}")
    print(f"[P7_SERVER] Lock gate: h>={LOCK_MIN_HORIZONTAL_SCORE} "
          f"floor>={LOCK_MIN_FLOOR_RATIO} "
          f"ir>={LOCK_MIN_INLIER_RATIO} "
          f"rmse<={LOCK_MAX_RMSE} "
          f"conf>={LOCK_MIN_CONFIDENCE}")
    print(f"[P7_SERVER] ALLOW_FALLBACK_LOCK={ALLOW_FALLBACK_LOCK} "
          f"CANDIDATE_POOL_SIZE={CANDIDATE_POOL_SIZE}")
    print(f"[P9_SERVER] Post-lock: enabled={POST_LOCK_ENABLED} "
          f"diag_fps={POST_LOCK_DIAGNOSTIC_FPS} "
          f"stop_depth={POST_LOCK_STOP_DEPTH} "
          f"stop_seg={POST_LOCK_STOP_SEGMENTATION}")
    print("=" * 60)

    udp_receiver = UDPReceiver(
        host=UDP_HOST, port=UDP_PORT, max_packet_size=MAX_UDP_PACKET)

    tcp_server = TCPServer(
        host=TCP_HOST, port=TCP_PORT,
        send_interval=DUMMY_SEND_INTERVAL,
        boundary_mode=BOUNDARY_MODE,
        allow_dummy_guardian=ALLOW_DUMMY_GUARDIAN,
    )

    pipeline_worker = None

    if BOUNDARY_MODE == "ai_floor" and DEPTH_ENABLED:
        from models.ai_pipeline_worker import AIPipelineWorker

        pipeline_worker = AIPipelineWorker(
            # Depth
            depth_model_name=DEPTH_MODEL_NAME,
            depth_allow_dummy=DEPTH_ALLOW_DUMMY,
            depth_use_metric_indoor=DEPTH_USE_METRIC_INDOOR,
            depth_try_depth_pro=DEPTH_TRY_DEPTH_PRO,
            # Segmentation
            seg_model_name=SEG_MODEL_NAME,
            # Pipeline
            target_fps=PIPELINE_TARGET_FPS,
            # Geometry
            min_depth=MIN_DEPTH_M,
            max_depth=MAX_DEPTH_M,
            max_points=MAX_BACKPROJECT_POINTS,
            ransac_iterations=RANSAC_ITERATIONS,
            ransac_threshold=RANSAC_INLIER_THRESHOLD_M,
            plane_min_inliers=PLANE_MIN_INLIERS,
            plane_min_ratio=PLANE_MIN_INLIER_RATIO,
            plane_max_rmse=PLANE_MAX_RMSE_M,
            # Boundary
            boundary_width=BOUNDARY_WIDTH_M,
            boundary_depth=BOUNDARY_DEPTH_M,
            boundary_near_z=BOUNDARY_NEAR_Z_M,
            boundary_far_z=BOUNDARY_FAR_Z_M,
            boundary_safety_margin=BOUNDARY_SAFETY_MARGIN_M,
            floor_min_ratio=FLOOR_MIN_PIXEL_RATIO,
            # Scan gate
            lower_image_roi=LOWER_IMAGE_ROI,
            plane_min_horizontal_score=PLANE_MIN_HORIZONTAL_SCORE,
            # Lock gate (strict)
            lock_min_horizontal_score=LOCK_MIN_HORIZONTAL_SCORE,
            lock_min_floor_ratio=LOCK_MIN_FLOOR_RATIO,
            lock_min_inlier_ratio=LOCK_MIN_INLIER_RATIO,
            lock_max_rmse=LOCK_MAX_RMSE,
            lock_min_confidence=LOCK_MIN_CONFIDENCE,
            # Legacy stability
            stable_required=STABLE_REQUIRED,
            stable_max_angle_diff=STABLE_MAX_ANGLE_DIFF,
            stable_max_drift_m=STABLE_MAX_DRIFT_M,
            stable_min_confidence=STABLE_MIN_CONFIDENCE,
            stable_max_rmse=STABLE_MAX_RMSE,
            stable_min_inlier_ratio=STABLE_MIN_INLIER_RATIO,
        )
    elif BOUNDARY_MODE == "dummy":
        print("[P7_SERVER] boundary mode=dummy")
    else:
        print(f"[P7_SERVER] depth disabled or unknown boundary mode")

    def shutdown(signum, frame):
        print("\n[P9_SERVER] shutting down...")
        # Phase 9: Export session metrics before stopping
        if pipeline_worker is not None:
            try:
                metrics = pipeline_worker.export_session_artifacts()
                print(f"[P9_EXPORT] session metrics exported: "
                      f"warm_fps={metrics.get('warm_fps', 0)} "
                      f"idle_fps={metrics.get('idle_fps', 0)} "
                      f"hand_frames={metrics.get('hand_frames', 0)}")
            except Exception as e:
                print(f"[P9_EXPORT_ERR] failed to export metrics: {e}")
            pipeline_worker.stop()
        udp_receiver.stop()
        tcp_server.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    udp_receiver.start()
    tcp_server.start()

    if pipeline_worker is not None:
        try:
            pipeline_worker.set_tcp_server(tcp_server)
            pipeline_worker.start()
            udp_receiver.set_depth_worker(pipeline_worker)
            print("[P7_SERVER] AI pipeline connected: UDP -> Pipeline -> TCP")
        except RuntimeError as e:
            print(f"\n[P7_SERVER] FATAL: {e}")
            print("[P7_SERVER] server running WITHOUT AI boundary.")
            pipeline_worker = None

    print("[P7_SERVER] all services running. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        shutdown(None, None)


if __name__ == "__main__":
    main()