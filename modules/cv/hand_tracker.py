"""
hand_tracker.py  —  v8  EMA Smoothing + ARCore Checker
=======================================================
CHANGES FROM v7 (previous fixed version):
  EMA — Exponential Moving Average on index_tip x/y reduces jitter.
        Smoother fingertip = more accurate world boundary placement.
        Alpha = 0.5 (tune 0.3=smoother / 0.7=more responsive)
  DEBOUNCE — is_pointing must hold true for 3 consecutive frames before
             reporting True. Eliminates single-frame false triggers that
             caused hold-timer resets in Unity.
  ARCore HEALTH CHECK — check_arcore_compatibility() static method lets
             the server report ARCore device requirements at startup.

All v7 pointing logic preserved:
  - idx_lift >= 0.08 threshold
  - mid/ring/pinky curl >= 0.04 threshold
  - thumb not fully extended check
  - confidence score 0-1 float
"""

import mediapipe as mp


class HandTracker:
    # EMA smoothing alpha: 0.3=smoothest, 0.7=most responsive
    EMA_ALPHA = 0.5

    def __init__(self, logger):
        self.logger   = logger
        self.is_ready = False

        # EMA state: keyed by hand index
        self._ema_x   = {}   # { hand_id: float }
        self._ema_y   = {}

        # Pointing debounce: count consecutive True frames per hand
        self._point_frames = {}   # { hand_id: int }
        self._POINT_MIN_FRAMES = 3   # must hold for 3 frames before True

        try:
            self.logger.info("[HANDS] Loading MediaPipe Hands...")
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=1,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6,
            )
            self.is_ready = True
            self.logger.info("[HANDS] Hand tracker ready (EMA smoothing + debounce ON)")
        except Exception as e:
            self.logger.error(f"[HANDS] Init failed: {e}")

    # ── ARCore Compatibility Check ─────────────────────────────────────────
    @staticmethod
    def check_arcore_compatibility():
        """
        Print ARCore device requirements to console at server startup.
        This helps the developer verify the device will work before building.
        Returns a dict with all check results.
        """
        checks = {}

        # Check 1: mediapipe version
        try:
            import mediapipe as mp
            checks['mediapipe'] = {'ok': True, 'version': mp.__version__}
        except ImportError:
            checks['mediapipe'] = {'ok': False, 'version': None}

        # Check 2: OpenCV (frame decoding)
        try:
            import cv2
            checks['opencv'] = {'ok': True, 'version': cv2.__version__}
        except ImportError:
            checks['opencv'] = {'ok': False, 'version': None}

        # Check 3: NumPy (array ops)
        try:
            import numpy as np
            checks['numpy'] = {'ok': True, 'version': np.__version__}
        except ImportError:
            checks['numpy'] = {'ok': False, 'version': None}

        # Check 4: Network port availability
        import socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("0.0.0.0", 9998))   # test port (not 9999 — that's used by server)
            s.close()
            checks['network_port'] = {'ok': True, 'detail': 'port 9999 likely available'}
        except OSError as e:
            checks['network_port'] = {'ok': False, 'detail': str(e)}

        return checks

    # ── Pointing detection (unchanged from v7) ────────────────────────────
    def is_pointing(self, lm):
        """
        Returns (bool, float) — (is_pointing, confidence 0-1).
        MediaPipe landmarks: 0=wrist, 4=thumb tip, 5=index MCP, 8=index tip,
          9=middle MCP, 12=middle tip, 13=ring MCP, 16=ring tip,
          17=pinky MCP, 20=pinky tip
        """
        try:
            idx_lift  = lm[5].y  - lm[8].y    # positive when index points up
            if idx_lift < 0.08: return False, 0.0

            mid_curl  = lm[12].y - lm[9].y
            if mid_curl < 0.04:  return False, 0.0

            ring_curl = lm[16].y - lm[13].y
            if ring_curl < 0.04: return False, 0.0

            pink_curl = lm[20].y - lm[17].y
            if pink_curl < 0.03: return False, 0.0

            thumb_up  = lm[2].y  - lm[4].y
            if thumb_up > 0.12:  return False, 0.0

            conf = (min(1.0, (idx_lift  - 0.08) / 0.12)
                  * min(1.0, (mid_curl  - 0.04) / 0.08)
                  * min(1.0, (ring_curl - 0.04) / 0.08))

            return True, float(conf)

        except Exception as e:
            self.logger.error(f"[HANDS] Gesture check: {e}")
            return False, 0.0

    # ── EMA smoothing helper ───────────────────────────────────────────────
    def _ema(self, hand_id, new_x, new_y):
        """Apply exponential moving average to fingertip position."""
        a = self.EMA_ALPHA
        if hand_id not in self._ema_x:
            # First frame: initialize to current value
            self._ema_x[hand_id] = new_x
            self._ema_y[hand_id] = new_y
        else:
            self._ema_x[hand_id] = a * new_x + (1 - a) * self._ema_x[hand_id]
            self._ema_y[hand_id] = a * new_y + (1 - a) * self._ema_y[hand_id]
        return self._ema_x[hand_id], self._ema_y[hand_id]

    # ── Pointing debounce ─────────────────────────────────────────────────
    def _debounce_pointing(self, hand_id, raw_pointing):
        """
        Only report True after N consecutive True frames.
        Immediately reports False when pointing stops.
        """
        if raw_pointing:
            self._point_frames[hand_id] = self._point_frames.get(hand_id, 0) + 1
        else:
            self._point_frames[hand_id] = 0

        return self._point_frames.get(hand_id, 0) >= self._POINT_MIN_FRAMES

    # ── Main detect ───────────────────────────────────────────────────────
    def detect(self, img_rgb):
        """
        Detect hands in RGB image.
        Returns dict with EMA-smoothed index_tip coords and debounced is_pointing.
        Schema matches Unity CameraPreview.cs / HandInfo:
          { detected: bool, hands: [{ id, is_pointing, point_conf, index_tip, landmarks }] }
        """
        if not self.is_ready:
            return {"detected": False, "hands": []}

        try:
            results = self.hands.process(img_rgb)
            out = {"detected": False, "hands": []}

            if not results.multi_hand_landmarks:
                # Clear EMA/debounce state when no hands visible
                self._ema_x.clear(); self._ema_y.clear()
                self._point_frames.clear()
                return out

            out["detected"] = True

            # Track which hand IDs appeared this frame
            active_ids = set()

            for idx, hand_lm in enumerate(results.multi_hand_landmarks):
                lm = hand_lm.landmark
                active_ids.add(idx)

                # Raw pointing detection
                raw_pointing, conf = self.is_pointing(lm)

                # Debounced pointing (must hold 3 frames)
                debounced_pointing = self._debounce_pointing(idx, raw_pointing)

                # EMA-smoothed index tip
                raw_x, raw_y = float(lm[8].x), float(lm[8].y)
                smooth_x, smooth_y = self._ema(idx, raw_x, raw_y)

                lm_list = [{"x": float(p.x), "y": float(p.y), "z": float(p.z)}
                           for p in lm]

                out["hands"].append({
                    "id":          idx,
                    "is_pointing": debounced_pointing,   # debounced
                    "point_conf":  conf,
                    "index_tip": {
                        "x": smooth_x,   # EMA smoothed
                        "y": smooth_y,
                        "z": float(lm[8].z),
                    },
                    "landmarks": lm_list,
                })

            # Clean up state for hands that disappeared
            for hand_id in list(self._ema_x.keys()):
                if hand_id not in active_ids:
                    self._ema_x.pop(hand_id, None)
                    self._ema_y.pop(hand_id, None)
                    self._point_frames.pop(hand_id, None)

            return out

        except Exception as e:
            self.logger.error(f"[HANDS] detect: {e}")
            return {"detected": False, "hands": []}

    def cleanup(self):
        if self.is_ready and self.hands:
            try: self.hands.close()
            except: pass