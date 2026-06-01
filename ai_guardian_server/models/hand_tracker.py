# ===========================================================================
# models/hand_tracker.py — Phase 7C
# MediaPipe hand landmark detection + depth-based 3D ray computation.
#
# Phase 7C changes:
#   - Uses MediaPipe Tasks API (mediapipe>=0.10.14) instead of legacy
#     mp.solutions.hands which was removed in 0.10.14+.
#   - Full diagnostic logging at every step.
#   - 2D-only fallback: sends AI_HAND_DATA even when depth fails.
#   - valid_2d / valid_3d flags for Unity to handle gracefully.
#   - Ray from camera-forward fallback when depth is invalid.
#
# Pipeline:
#   1. MediaPipe HandLandmarker detects 21 2D landmarks per hand.
#   2. All 21 landmarks are backprojected to 3D using depth map.
#   3. Ray computed from index finger chain (Option C blended direction).
#   4. Convert to Unity camera space: [x, -y, z].
#   5. Send as AI_HAND_DATA packet with all 21 camera-space landmarks.
#
# Coordinate conversion (same as floor pipeline):
#   OpenCV camera: X=right, Y=down,  Z=forward
#   Unity camera:  X=right, Y=up,    Z=forward
#   Mapping: X → x, Y → -y, Z → z
# ===========================================================================

import os
import time
import numpy as np


# Phase 9.21: counters for periodic NONE diagnostics. Class-level state below.

# MediaPipe landmark indices
_LM_WRIST       = 0
_LM_THUMB_CMC   = 1
_LM_THUMB_MCP   = 2
_LM_THUMB_IP    = 3
_LM_THUMB_TIP   = 4
_LM_INDEX_MCP   = 5
_LM_INDEX_PIP   = 6
_LM_INDEX_DIP   = 7
_LM_INDEX_TIP   = 8
_LM_MIDDLE_MCP  = 9
_LM_MIDDLE_PIP  = 10
_LM_MIDDLE_DIP  = 11
_LM_MIDDLE_TIP  = 12
_LM_RING_MCP    = 13
_LM_RING_PIP    = 14
_LM_RING_DIP    = 15
_LM_RING_TIP    = 16
_LM_PINKY_MCP   = 17
_LM_PINKY_PIP   = 18
_LM_PINKY_DIP   = 19
_LM_PINKY_TIP   = 20


class HandTracker:
    """
    MediaPipe Tasks-based hand landmark tracker.
    Detects one hand and produces a 3D ray from the index finger,
    plus all 21 3D landmarks for holographic glove rendering.

    Uses mediapipe.tasks.python.vision.HandLandmarker (Tasks API),
    which requires a .task model file downloaded from Google.

    Phase 7C: Always returns a result when MediaPipe detects a 2D hand,
    even if depth is invalid (uses fallback camera-forward ray).
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        max_num_hands: int = 1,
        model_path: str = None,
    ):
        self._enabled = False
        self._detector = None
        # Phase 9.21: diagnostic counters + debug dump state for P7_HAND_NONE.
        self._none_streak = 0
        self._total_frames = 0
        self._total_detected = 0
        self._last_none_dump_time = 0.0
        self._debug_dump_dir = None  # initialized lazily on first dump

        # Resolve model path
        if model_path is None:
            # Look in models/ directory relative to this file
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, "hand_landmarker.task")

        if not os.path.exists(model_path):
            print(f"[P7_HAND_INIT] model file not found: {model_path}")
            print("[P7_HAND] Download from: https://storage.googleapis.com/mediapipe-models/"
                  "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
            self._enabled = False
            return

        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            base_options = mp_python.BaseOptions(
                model_asset_path=model_path,
            )

            options = mp_vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.IMAGE,
                num_hands=max_num_hands,
                min_hand_detection_confidence=min_detection_confidence,
                min_hand_presence_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )

            self._detector = mp_vision.HandLandmarker.create_from_options(options)
            self._mp = mp  # keep reference for mp.Image
            self._enabled = True

            print(f"[P7_HAND_INIT] api=MediaPipeTasks "
                  f"model={os.path.basename(model_path)} "
                  f"det={min_detection_confidence} "
                  f"track={min_tracking_confidence} "
                  f"max_hands={max_num_hands}")

        except Exception as e:
            import traceback
            print(f"[P7_HAND_INIT] FAILED: {e}")
            traceback.print_exc()
            self._enabled = False

    @property
    def is_available(self) -> bool:
        return self._enabled and self._detector is not None

    def detect(
        self,
        rgb_frame: np.ndarray,
        depth_map: np.ndarray,
        fx: float, fy: float,
        cx: float, cy: float,
        frame_id: str = "?",
    ) -> dict:
        """
        Detect hand landmarks and compute a 3D ray in Unity camera space.

        Phase 7C: Always returns a result when MediaPipe detects a 2D hand.
        If depth is invalid for key landmarks, sends 2D-only result with
        valid_3d=False and a fallback camera-forward ray.

        Args:
            rgb_frame:  HxW RGB uint8 image (already rotated if needed).
            depth_map:  HxW float32 depth in meters (same shape as rgb_frame).
            fx, fy:     Focal lengths in pixels (matching rotated frame).
            cx, cy:     Principal point in pixels (matching rotated frame).
            frame_id:   For logging.

        Returns:
            dict with hand data, or None if no hand detected at all.
        """
        if not self.is_available:
            return None

        h, w = rgb_frame.shape[:2]
        self._total_frames += 1

        # Phase 9.21: input-frame sanity. MediaPipe HandLandmarker silently returns
        # empty results when the input is the wrong dtype/shape/range. Probe and log.
        if rgb_frame.dtype != np.uint8 or rgb_frame.ndim != 3 or rgb_frame.shape[2] != 3:
            print(f"[P7_HAND_INPUT_BAD] frame={frame_id} dtype={rgb_frame.dtype} "
                  f"shape={rgb_frame.shape} expected=uint8/H,W,3 — fix upstream conversion")

        try:
            # Tasks API requires mp.Image
            mp_image = self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB,
                data=rgb_frame.copy(),  # copy to ensure contiguous
            )
            results = self._detector.detect(mp_image)
        except Exception as e:
            print(f"[P7_HAND_ERR] frame={frame_id}: {e}")
            return None

        if not results.hand_landmarks or len(results.hand_landmarks) == 0:
            # ----------------------------------------------------------------
            # Phase 9.21: rich P7_HAND_NONE diagnostics so we can root-cause
            # the empty-stream problem instead of guessing. We log:
            #   - input image stats (mean, std) to catch all-black / clipped frames
            #   - detection-rate over the session so the user sees if it ever worked
            #   - periodic debug image dump of what MediaPipe is actually seeing,
            #     so the user can confirm rotation/color order is correct
            # ----------------------------------------------------------------
            self._none_streak += 1

            mean_v = float(rgb_frame.mean()) if rgb_frame.size > 0 else -1.0
            std_v  = float(rgb_frame.std())  if rgb_frame.size > 0 else -1.0
            rate   = (self._total_detected / max(1, self._total_frames)) * 100.0

            if self._none_streak == 1 or self._none_streak % 30 == 0:
                # First miss in a streak, then every ~1s at 30 FPS.
                print(f"[P7_HAND_NONE] frame={frame_id} streak={self._none_streak} "
                      f"img={w}x{h} mean={mean_v:.1f} std={std_v:.1f} "
                      f"hit_rate={rate:.1f}% "
                      f"total_detected={self._total_detected}/{self._total_frames}")

            # Save a debug RGB image at most once every 3 seconds so we can
            # eyeball the actual MediaPipe input. Look for: rotation wrong,
            # color inverted, hand out of frame, image too dark / too bright.
            now = time.time()
            if now - self._last_none_dump_time >= 3.0:
                self._last_none_dump_time = now
                self._dump_debug_image(rgb_frame, frame_id, "none")

            # Phase 9.21: rotation probe — if we've missed for a long time, the
            # most common cause is the upstream rotation being wrong for the
            # phone's current orientation. Try the three alternate rotations
            # once every 60 frames during a long streak, log which one
            # actually finds a hand. We don't return that hand (the rotation
            # would be wrong for downstream depth/projection), we only log so
            # the user can fix INPUT_ROTATION_DEGREES.
            if self._none_streak > 0 and self._none_streak % 60 == 0:
                self._probe_rotations(rgb_frame, frame_id)

            return None

        # Reset streak when a hand IS detected.
        if self._none_streak > 0:
            print(f"[P7_HAND_RECOVERED] after streak={self._none_streak} frames")
            self._none_streak = 0
        self._total_detected += 1

        # Take first hand — Tasks API returns list of NormalizedLandmark lists
        hand_landmarks = results.hand_landmarks[0]  # list of 21 NormalizedLandmark
        handedness_list = results.handedness
        hand_label = "Unknown"
        hand_score = 0.0
        if handedness_list and len(handedness_list) > 0:
            # Tasks API: handedness is list of Category lists
            hand_label = handedness_list[0][0].category_name
            hand_score = handedness_list[0][0].score

        # Log 2D detection
        wrist_lm = hand_landmarks[_LM_WRIST]
        index_tip_lm = hand_landmarks[_LM_INDEX_TIP]
        print(f"[P7_HAND_2D] frame={frame_id} hand={hand_label} "
              f"conf={hand_score:.2f} "
              f"wrist=({wrist_lm.x:.3f},{wrist_lm.y:.3f}) "
              f"index_tip=({index_tip_lm.x:.3f},{index_tip_lm.y:.3f})")

        # ------------------------------------------------------------------
        # Backproject ALL 21 landmarks to 3D (Unity camera space)
        # ------------------------------------------------------------------
        landmarks_camera = []   # All 21 landmarks in Unity camera space
        landmarks_2d = []       # All 21 landmarks normalized 2D
        valid_count = 0

        # Phase 3: two ways to get 3D landmarks in Unity camera space.
        #  (a) depth available (pre-lock): backproject each 2D landmark using the
        #      mono-depth map, with a robust depth-slab clamp so silhouette noise
        #      can't fling one joint onto a far wall (the old "huge / random"
        #      glove). A real hand spans only ~12 cm in depth, so we take the
        #      median sampled depth as a reference and clamp every joint into a
        #      slab around it (no-op when depths are already coherent → genuine
        #      finger-pointing geometry, and the ray direction, are preserved).
        #  (b) depth disabled (post-lock / lobby — POST_LOCK_STOP_DEPTH): DepthAnything
        #      is OFF to free the GPU for MediaPipe (>=15 Hz hand). We reconstruct a
        #      clean, coherent 3D hand from MediaPipe's OWN normalized (x,y,z)
        #      landmarks + camera intrinsics. No depth model needed; the shape
        #      comes straight from MediaPipe so it is stable, never random.
        DEPTH_SLAB_M = 0.18   # half-thickness of the plausible hand depth band
        use_depth = depth_map is not None
        clamped_count = 0

        if use_depth:
            px_coords = []
            raw_depths = []
            for lm in hand_landmarks:
                px_x = max(0, min(w - 1, int(lm.x * w)))
                px_y = max(0, min(h - 1, int(lm.y * h)))
                px_coords.append((px_x, px_y))
                # Use larger patch for depth sampling (7x7 median)
                raw_depths.append(self._sample_depth(depth_map, px_x, px_y, radius=3))

            valid_d = [d for d in raw_depths if d > 0.05]
            ref_d = float(np.median(valid_d)) if valid_d else 0.0

            for i, lm in enumerate(hand_landmarks):
                landmarks_2d.append([round(lm.x, 4), round(lm.y, 4), round(lm.z, 4)])

                px_x, px_y = px_coords[i]
                d = raw_depths[i]

                if d <= 0.05:
                    # Invalid depth — use None placeholder
                    landmarks_camera.append(None)
                else:
                    if ref_d > 0.05:
                        cd = min(max(d, ref_d - DEPTH_SLAB_M), ref_d + DEPTH_SLAB_M)
                        if cd != d:
                            clamped_count += 1
                        d = cd
                    pt_cv = self._backproject(px_x, px_y, d, fx, fy, cx, cy)
                    # Convert to Unity camera space: [x, -y, z]
                    pt_unity = [
                        round(float(pt_cv[0]), 4),
                        round(float(-pt_cv[1]), 4),
                        round(float(pt_cv[2]), 4),
                    ]
                    landmarks_camera.append(pt_unity)
                    valid_count += 1

            print(f"[P7_HAND_3D] frame={frame_id} valid={valid_count}/21 "
                  f"src=depth ref_depth={ref_d:.2f} clamped={clamped_count}")
        else:
            # ----- MediaPipe-only 3D reconstruction (no depth model) -----
            # SIZE the hand correctly by estimating its distance from the camera:
            # a real wrist->middle-MCP span is ~0.09 m, so inverting the pinhole
            # model with that bone's pixel length gives a plausible base depth.
            REAL_PALM_M = 0.09
            w0 = hand_landmarks[_LM_WRIST]
            m9 = hand_landmarks[_LM_MIDDLE_MCP]
            palm_px = float(np.hypot((m9.x - w0.x) * w, (m9.y - w0.y) * h))
            base_depth = REAL_PALM_M * float(fy) / max(palm_px, 1.0)
            base_depth = min(max(base_depth, 0.25), 1.20)   # arm's-length band
            # MediaPipe z is wrist-relative, ~same scale as normalized x; convert
            # to metres and add as MODEST depth relief so the hand isn't flat.
            z_scale = base_depth
            for lm in hand_landmarks:
                landmarks_2d.append([round(lm.x, 4), round(lm.y, 4), round(lm.z, 4)])
                px_x = lm.x * w
                px_y = lm.y * h
                d = max(0.1, base_depth + float(lm.z) * z_scale)
                pt_cv = self._backproject(px_x, px_y, d, fx, fy, cx, cy)
                pt_unity = [
                    round(float(pt_cv[0]), 4),
                    round(float(-pt_cv[1]), 4),
                    round(float(pt_cv[2]), 4),
                ]
                landmarks_camera.append(pt_unity)
                valid_count += 1

            print(f"[P7_HAND_3D] frame={frame_id} valid={valid_count}/21 "
                  f"src=mediapipe base_depth={base_depth:.2f}")

        # ------------------------------------------------------------------
        # Ray computation — Option C (blended direction, most stable)
        # Phase 7C: If depth fails, use fallback camera-forward ray
        # ------------------------------------------------------------------
        tip_3d = landmarks_camera[_LM_INDEX_TIP]
        pip_3d = landmarks_camera[_LM_INDEX_PIP]
        mcp_3d = landmarks_camera[_LM_INDEX_MCP]

        valid_3d = True
        ray_origin = np.array([0.0, 0.0, 0.0])
        ray_dir = np.array([0.0, 0.0, 1.0])  # default: camera forward

        if tip_3d is not None and pip_3d is not None and mcp_3d is not None:
            tip_v = np.array(tip_3d, dtype=np.float64)
            pip_v = np.array(pip_3d, dtype=np.float64)
            mcp_v = np.array(mcp_3d, dtype=np.float64)

            # Option C: blended direction for stability
            dir_tip_pip = tip_v - pip_v
            dir_pip_mcp = pip_v - mcp_v

            len_tp = np.linalg.norm(dir_tip_pip)
            len_pm = np.linalg.norm(dir_pip_mcp)

            if len_tp >= 1e-6 and len_pm >= 1e-6:
                dir_tp_norm = dir_tip_pip / len_tp
                dir_pm_norm = dir_pip_mcp / len_pm

                # Blend: 65% tip-pip + 35% pip-mcp for stability
                ray_dir = 0.65 * dir_tp_norm + 0.35 * dir_pm_norm
                ray_len = np.linalg.norm(ray_dir)
                if ray_len >= 1e-6:
                    ray_dir = ray_dir / ray_len
                else:
                    ray_dir = np.array([0.0, 0.0, 1.0])
                    valid_3d = False

                ray_origin = tip_v  # Ray starts at index fingertip
            else:
                valid_3d = False
        else:
            valid_3d = False
            # Fallback: compute approximate ray from 2D landmarks
            idx_tip_nx = index_tip_lm.x
            idx_tip_ny = index_tip_lm.y
            # Convert normalized 2D to camera ray direction
            px_x_tip = idx_tip_nx * w
            px_y_tip = idx_tip_ny * h
            dir_x = (px_x_tip - cx) / fx
            dir_y = -(px_y_tip - cy) / fy  # flip Y for Unity
            dir_z = 1.0
            ray_dir = np.array([dir_x, dir_y, dir_z], dtype=np.float64)
            ray_dir = ray_dir / np.linalg.norm(ray_dir)
            # Fallback origin: approximate from 2D at mean depth
            mean_depth = float(np.mean(depth_map[depth_map > 0.05])) if np.any(depth_map > 0.05) else 2.0
            ray_origin = ray_dir * mean_depth * 0.8  # rough estimate

            print(f"[P7_HAND_3D] frame={frame_id} fallback_ray "
                  f"depth_invalid tip={tip_3d is not None} "
                  f"pip={pip_3d is not None} mcp={mcp_3d is not None}")

        # Depth values for debug
        tip_px = (max(0, min(w - 1, int(hand_landmarks[_LM_INDEX_TIP].x * w))),
                  max(0, min(h - 1, int(hand_landmarks[_LM_INDEX_TIP].y * h))))
        pip_px = (max(0, min(w - 1, int(hand_landmarks[_LM_INDEX_PIP].x * w))),
                  max(0, min(h - 1, int(hand_landmarks[_LM_INDEX_PIP].y * h))))
        mcp_px = (max(0, min(w - 1, int(hand_landmarks[_LM_INDEX_MCP].x * w))),
                  max(0, min(h - 1, int(hand_landmarks[_LM_INDEX_MCP].y * h))))

        tip_depth = self._sample_depth(depth_map, tip_px[0], tip_px[1], radius=3)
        pip_depth = self._sample_depth(depth_map, pip_px[0], pip_px[1], radius=3)
        mcp_depth = self._sample_depth(depth_map, mcp_px[0], mcp_px[1], radius=3)

        # ------------------------------------------------------------------
        # Build serializable landmarks list (None → null for JSON)
        # ------------------------------------------------------------------
        landmarks_camera_json = []
        for pt in landmarks_camera:
            if pt is None:
                landmarks_camera_json.append(None)
            else:
                landmarks_camera_json.append(pt)

        # ------------------------------------------------------------------
        # Build result packet
        # ------------------------------------------------------------------
        result = {
            "type": "AI_HAND_DATA",
            "frame_id": frame_id,
            "timestamp_ms": int(time.time() * 1000),
            "hand_valid": True,
            "valid_2d": True,
            "valid_3d": valid_3d,
            "handedness": hand_label,
            "confidence": round(float(hand_score), 3),
            "landmarks_2d": landmarks_2d,
            "landmarks_camera": landmarks_camera_json,
            "index_tip_camera": [round(v, 4) for v in (tip_3d if tip_3d else [0, 0, 0])],
            "index_pip_camera": [round(v, 4) for v in (pip_3d if pip_3d else [0, 0, 0])],
            "index_mcp_camera": [round(v, 4) for v in (mcp_3d if mcp_3d else [0, 0, 0])],
            "ray_origin_camera": [round(v, 4) for v in ray_origin.tolist()],
            "ray_dir_camera": [round(v, 4) for v in ray_dir.tolist()],
            "tip_depth_m": round(float(tip_depth), 3),
            "mcp_depth_m": round(float(mcp_depth), 3),
            "valid_landmark_count": valid_count,
            "debug": {
                "depth_tip": round(float(tip_depth), 3),
                "depth_pip": round(float(pip_depth), 3),
                "depth_mcp": round(float(mcp_depth), 3),
                "selected_points": valid_count,
                "valid_3d": valid_3d,
            },
        }

        print(f"[P7_HAND] frame={frame_id} valid=True handedness={hand_label} "
              f"conf={hand_score:.2f} pts={valid_count}/21 valid_3d={valid_3d}")
        print(f"[P7_RAY] origin=({ray_origin[0]:.3f},{ray_origin[1]:.3f},{ray_origin[2]:.3f}) "
              f"dir=({ray_dir[0]:.3f},{ray_dir[1]:.3f},{ray_dir[2]:.3f}) "
              f"tip_depth={tip_depth:.3f} valid_3d={valid_3d}")

        return result

    def close(self):
        """Release MediaPipe resources."""
        if self._detector is not None:
            try:
                self._detector.close()
            except Exception:
                pass
            self._detector = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _probe_rotations(self, rgb_frame, frame_id):
        """Run MediaPipe on the current frame rotated by 0/90/180/270 and log
        which (if any) finds a hand. Diagnostic only — does NOT affect the
        normal pipeline. Use the result to fix INPUT_ROTATION_DEGREES in config.py.
        """
        try:
            import cv2
            verdicts = []
            for ang, code in (
                (0,   None),
                (90,  cv2.ROTATE_90_CLOCKWISE),
                (180, cv2.ROTATE_180),
                (270, cv2.ROTATE_90_COUNTERCLOCKWISE),
            ):
                probe = rgb_frame if code is None else cv2.rotate(rgb_frame, code)
                mp_image = self._mp.Image(
                    image_format=self._mp.ImageFormat.SRGB,
                    data=probe.copy(),
                )
                r = self._detector.detect(mp_image)
                found = bool(r.hand_landmarks) and len(r.hand_landmarks) > 0
                verdicts.append(f"{ang}={'HIT' if found else 'miss'}")
            print(f"[P7_HAND_ROT_PROBE] frame={frame_id} " +
                  " ".join(verdicts) +
                  " — if one rotation HITs, set INPUT_ROTATION_DEGREES in config.py to match")
        except Exception as e:
            print(f"[P7_HAND_ROT_PROBE_ERR] {e}")

    def _dump_debug_image(self, rgb_frame, frame_id, tag):
        """Phase 9.21: write the RGB frame MediaPipe sees so the user can verify
        rotation/colour/exposure. Files land under debug/hand_dumps/. Throttled
        by the caller — this method just does the write.
        """
        try:
            import cv2  # local import — avoid hard dep at module load
            if self._debug_dump_dir is None:
                base = os.path.dirname(os.path.abspath(__file__))
                d = os.path.join(base, "..", "debug", "hand_dumps")
                d = os.path.abspath(d)
                os.makedirs(d, exist_ok=True)
                self._debug_dump_dir = d
                print(f"[P7_HAND_DUMP_DIR] {self._debug_dump_dir}")
            # MediaPipe expects RGB; OpenCV imwrite needs BGR.
            bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            fn = os.path.join(self._debug_dump_dir, f"hand_{tag}_{frame_id}.jpg")
            cv2.imwrite(fn, bgr)
            print(f"[P7_HAND_DUMP] saved {fn} reason={tag}")
        except Exception as e:
            print(f"[P7_HAND_DUMP_ERR] {e}")

    @staticmethod
    def _sample_depth(depth_map: np.ndarray, px: int, py: int, radius: int = 3) -> float:
        """Sample depth with a median filter for robustness.
        Default radius=3 → 7x7 window."""
        if depth_map is None:
            return 0.0   # Phase 3: depth disabled post-lock — caller reconstructs from MediaPipe
        h, w = depth_map.shape[:2]
        y0 = max(0, py - radius)
        y1 = min(h, py + radius + 1)
        x0 = max(0, px - radius)
        x1 = min(w, px + radius + 1)
        patch = depth_map[y0:y1, x0:x1]
        valid = patch[patch > 0.05]
        if len(valid) == 0:
            return 0.0
        return float(np.median(valid))

    @staticmethod
    def _backproject(px: int, py: int, depth: float,
                     fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
        """Backproject pixel + depth to OpenCV camera 3D point."""
        x = (px - cx) * depth / fx
        y = (py - cy) * depth / fy
        z = depth
        return np.array([x, y, z], dtype=np.float64)
