"""
hand_tracker.py  —  v9  Tasks API + EMA Smoothing
====================================================
REWRITE for MediaPipe 0.10.32+ which dropped mp.solutions.
Uses the new Tasks API (mp.tasks.python.vision.HandLandmarker).

Preserved from v8:
  - EMA smoothing on index_tip (alpha=0.5)
  - Pointing debounce (3 consecutive frames)
  - Pointing gesture detection (idx_lift, curl checks)
  - Same output schema for server.py compatibility
"""

import os
import urllib.request


# Model URL for auto-download
_HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)


def _ensure_model(path, url, logger):
    """Download model file if it doesn't exist."""
    if os.path.exists(path):
        return path
    logger.info(f"[HANDS] Downloading model → {path} ...")
    try:
        urllib.request.urlretrieve(url, path)
        logger.info(f"[HANDS] ✓ Model downloaded ({os.path.getsize(path) / 1e6:.1f} MB)")
    except Exception as e:
        logger.error(f"[HANDS] ✗ Model download failed: {e}")
        raise
    return path


class HandTracker:
    EMA_ALPHA = 0.5

    def __init__(self, logger):
        self.logger   = logger
        self.is_ready = False

        # EMA state
        self._ema_x = {}
        self._ema_y = {}

        # Pointing debounce
        self._point_frames = {}
        self._POINT_MIN_FRAMES = 1  # Bug 4 fix: let Unity own stability gating

        try:
            self.logger.info("[HANDS] Loading MediaPipe Hands (Tasks API)...")

            import mediapipe as mp
            from mediapipe.tasks.python import vision, BaseOptions

            # Auto-download model
            model_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(model_dir, "hand_landmarker.task")
            _ensure_model(model_path, _HAND_MODEL_URL, logger)

            # Create hand landmarker
            options = vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                num_hands=2,
                min_hand_detection_confidence=0.3,   # Bug 6 fix: was 0.6
                min_tracking_confidence=0.4,          # Bug 6 fix: was 0.6
                min_hand_presence_confidence=0.3,     # Bug 6 fix: was 0.6
                running_mode=vision.RunningMode.IMAGE,
            )
            self._landmarker = vision.HandLandmarker.create_from_options(options)
            self._mp = mp   # keep ref for mp.Image

            self.is_ready = True
            self.logger.info(f"[HANDS] ✓ Hand tracker ready (Tasks API v{mp.__version__})")

        except Exception as e:
            self.logger.error(f"[HANDS] Init failed: {e}")

    # ── ARCore Compatibility Check ──────────────────────────────────────
    @staticmethod
    def check_arcore_compatibility():
        checks = {}
        try:
            import mediapipe as mp
            checks['mediapipe'] = {'ok': True, 'version': mp.__version__}
        except ImportError:
            checks['mediapipe'] = {'ok': False, 'version': None}
        try:
            import cv2
            checks['opencv'] = {'ok': True, 'version': cv2.__version__}
        except ImportError:
            checks['opencv'] = {'ok': False, 'version': None}
        try:
            import numpy as np
            checks['numpy'] = {'ok': True, 'version': np.__version__}
        except ImportError:
            checks['numpy'] = {'ok': False, 'version': None}
        import socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("0.0.0.0", 9999))
            s.close()
            checks['network_port'] = {'ok': True, 'detail': 'port 9999 available'}
        except OSError as e:
            checks['network_port'] = {'ok': False, 'detail': str(e)}
        return checks

    # ── Pointing detection (v10 — orientation-independent) ─────────────
    def _dist(self, a, b):
        """Euclidean distance between two landmarks."""
        return ((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2) ** 0.5

    def _finger_extended(self, lm, mcp, pip, tip):
        """Check if finger is extended: tip far from MCP relative to pip-to-mcp distance."""
        mcp_pip = self._dist(lm[mcp], lm[pip])
        mcp_tip = self._dist(lm[mcp], lm[tip])
        if mcp_pip < 1e-6: return False, 0.0
        ratio = mcp_tip / mcp_pip  # extended ≈ 2.0+, curled ≈ 0.5-1.0
        return ratio > 1.5, ratio

    def _finger_curled(self, lm, mcp, pip, tip):
        """Check if finger is curled: tip close to MCP."""
        mcp_pip = self._dist(lm[mcp], lm[pip])
        mcp_tip = self._dist(lm[mcp], lm[tip])
        if mcp_pip < 1e-6: return True, 0.0
        ratio = mcp_tip / mcp_pip  # curled ≈ 0.5-1.2, extended ≈ 2.0+
        # Bug 8 fix: was 1.6 — rejected natural relaxed hands (ratios 1.6-1.8)
        return ratio < 2.0, ratio

    _gesture_log_counter = 0

    def is_pointing(self, lm):
        """
        Orientation-independent pointing detection.
        Two modes:
          MODE A: Index extended + 2+ others curled (classic pointing)
          MODE B: Index DOMINANT — index extended much more than others (relaxed hand)
        Works when pointing up, down, sideways, at floor, etc.
        """
        try:
            # Index finger: MCP=5, PIP=6, TIP=8
            idx_ext, idx_ratio = self._finger_extended(lm, 5, 6, 8)

            # Middle: MCP=9, PIP=10, TIP=12
            mid_curled, mid_ratio = self._finger_curled(lm, 9, 10, 12)

            # Ring: MCP=13, PIP=14, TIP=16
            ring_curled, ring_ratio = self._finger_curled(lm, 13, 14, 16)

            # Pinky: MCP=17, PIP=18, TIP=20
            pink_curled, pink_ratio = self._finger_curled(lm, 17, 18, 20)

            # Debug log every 30 frames
            self._gesture_log_counter += 1
            if self._gesture_log_counter % 30 == 0:
                self.logger.info(
                    f"[GESTURE] idx_ext={idx_ext}({idx_ratio:.2f}) "
                    f"mid_curl={mid_curled}({mid_ratio:.2f}) "
                    f"ring_curl={ring_curled}({ring_ratio:.2f}) "
                    f"pink_curl={pink_curled}({pink_ratio:.2f})")

            # Must have index extended
            if not idx_ext:
                return False, 0.0

            # ── MODE A: Classic pointing (index extended + others curled) ──
            curled_count = sum([mid_curled, ring_curled, pink_curled])
            if curled_count >= 2:
                conf = min(1.0, (idx_ratio - 1.5) / 1.0)
                return True, float(conf)

            # ── MODE B: Dominant index (Bug 8 fix) ──
            # User has relaxed open hand but index is clearly the most extended finger.
            # This catches "pointing at floor" with natural hand position.
            # Index must be significantly more extended than ALL other fingers.
            other_max = max(mid_ratio, ring_ratio, pink_ratio)
            if idx_ratio > 1.8 and idx_ratio > other_max + 0.1:
                conf = min(1.0, (idx_ratio - other_max) / 1.0)
                return True, float(conf)

            return False, 0.0

        except Exception as e:
            self.logger.error(f"[HANDS] Gesture check: {e}")
            return False, 0.0

    def is_fist(self, lm):
        """All 4 fingers curled = fist. Used as STOP drawing gesture."""
        try:
            idx_curled, _ = self._finger_curled(lm, 5, 6, 8)
            mid_curled, _ = self._finger_curled(lm, 9, 10, 12)
            ring_curled, _ = self._finger_curled(lm, 13, 14, 16)
            pink_curled, _ = self._finger_curled(lm, 17, 18, 20)
            is_fist = idx_curled and mid_curled and ring_curled and pink_curled
            return is_fist, 1.0 if is_fist else 0.0
        except:
            return False, 0.0

    # ── EMA smoothing ───────────────────────────────────────────────────
    def _ema(self, hand_id, new_x, new_y):
        a = self.EMA_ALPHA
        if hand_id not in self._ema_x:
            self._ema_x[hand_id] = new_x
            self._ema_y[hand_id] = new_y
        else:
            self._ema_x[hand_id] = a * new_x + (1 - a) * self._ema_x[hand_id]
            self._ema_y[hand_id] = a * new_y + (1 - a) * self._ema_y[hand_id]
        return self._ema_x[hand_id], self._ema_y[hand_id]

    # ── Pointing debounce ───────────────────────────────────────────────
    def _debounce_pointing(self, hand_id, raw_pointing):
        if raw_pointing:
            self._point_frames[hand_id] = self._point_frames.get(hand_id, 0) + 1
        else:
            # Soft decay instead of hard reset — 1 dropped frame doesn't kill stability
            self._point_frames[hand_id] = max(0, self._point_frames.get(hand_id, 0) - 1)
        return self._point_frames.get(hand_id, 0) >= self._POINT_MIN_FRAMES

    # ── Main detect (Tasks API) ─────────────────────────────────────────
    def detect(self, img_rgb, pad_sx=1.0, pad_sy=1.0):
        """
        Detect hands in RGB numpy array.
        pad_sx/pad_sy: scale factors from square-padding (coords are in padded space).
        Returns same schema as v8 for server.py compatibility.

        P1 FIX NOTE: img_rgb is now 160×160 (was 320×320) after the server-side
        resize. pad_sx/pad_sy still encode the original 240/320 = 0.75 ratio,
        which is correct — they correct landmarks from padded-space back to the
        original 240×320 portrait frame coordinate space.
        """
        if not self.is_ready:
            return {"detected": False, "hands": []}

        try:
            import numpy as np
            mp = self._mp

            # Convert numpy array to MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img_rgb))

            # Detect
            result = self._landmarker.detect(mp_image)

            out = {"detected": False, "hands": []}

            if not result.hand_landmarks:
                # Bug 3 fix: soft decay instead of hard clear
                for k in list(self._point_frames.keys()):
                    self._point_frames[k] = max(0, self._point_frames[k] - 1)
                    if self._point_frames[k] == 0:
                        self._point_frames.pop(k)
                # EMA can be cleared — no hand to smooth
                self._ema_x.clear()
                self._ema_y.clear()
                return out

            out["detected"] = True
            active_ids = set()

            for idx, hand_lm in enumerate(result.hand_landmarks):
                active_ids.add(idx)

                # Raw pointing detection
                raw_pointing, conf = self.is_pointing(hand_lm)

                # Fist detection (STOP gesture)
                raw_fist, _ = self.is_fist(hand_lm)

                # Debounced pointing
                debounced_pointing = self._debounce_pointing(idx, raw_pointing)

                # EMA-smoothed index tip — correct for padding
                raw_x = float(hand_lm[8].x) / pad_sx if pad_sx > 0 else float(hand_lm[8].x)
                raw_y = float(hand_lm[8].y) / pad_sy if pad_sy > 0 else float(hand_lm[8].y)
                # Clamp to [0, 1]
                raw_x = min(1.0, max(0.0, raw_x))
                raw_y = min(1.0, max(0.0, raw_y))
                smooth_x, smooth_y = self._ema(idx, raw_x, raw_y)

                # Correct all landmarks for padding too
                lm_list = [{"x": min(1.0, float(p.x) / pad_sx),
                            "y": min(1.0, float(p.y) / pad_sy),
                            "z": float(p.z)}
                           for p in hand_lm]

                out["hands"].append({
                    "id":          idx,
                    "is_pointing": debounced_pointing,
                    "is_fist":     raw_fist,
                    "point_conf":  conf,
                    "index_tip": {
                        "x": smooth_x,
                        "y": smooth_y,
                        "z": float(hand_lm[8].z),
                    },
                    "landmarks": lm_list,
                })

            # Cleanup disappeared hands
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
        if self.is_ready and hasattr(self, '_landmarker'):
            try:
                self._landmarker.close()
            except:
                pass