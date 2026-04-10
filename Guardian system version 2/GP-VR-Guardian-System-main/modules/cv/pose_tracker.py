"""
Pose Tracker Module — v2 Tasks API
====================================
Rewritten for MediaPipe 0.10.32+ which dropped mp.solutions.
Uses mp.tasks.python.vision.PoseLandmarker.
Auto-downloads the model file on first run.
"""

import os
import urllib.request


_POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)


def _ensure_model(path, url, logger):
    """Download model file if it doesn't exist."""
    if os.path.exists(path):
        return path
    logger.info(f"[POSE] Downloading model → {path} ...")
    try:
        urllib.request.urlretrieve(url, path)
        logger.info(f"[POSE] ✓ Model downloaded ({os.path.getsize(path) / 1e6:.1f} MB)")
    except Exception as e:
        logger.error(f"[POSE] ✗ Model download failed: {e}")
        raise
    return path


class PoseTracker:
    def __init__(self, logger):
        self.logger = logger
        self.is_ready = False

        try:
            self.logger.info("[POSE] Loading MediaPipe Pose (Tasks API)...")

            import mediapipe as mp
            from mediapipe.tasks.python import vision, BaseOptions

            # Auto-download model
            model_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(model_dir, "pose_landmarker_lite.task")
            _ensure_model(model_path, _POSE_MODEL_URL, logger)

            options = vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                min_pose_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                running_mode=vision.RunningMode.IMAGE,
            )
            self._landmarker = vision.PoseLandmarker.create_from_options(options)
            self._mp = mp

            self.is_ready = True
            self.logger.info(f"[POSE] ✓ Pose tracking ready (Tasks API v{mp.__version__})")

        except ImportError as e:
            self.logger.error(f"[POSE] MediaPipe import failed: {e}")
            self.logger.warning("[POSE] ✗ Pose tracking DISABLED")
            self.is_ready = False
        except Exception as e:
            self.logger.error(f"[POSE] Initialization failed: {e}")
            self.is_ready = False

    def detect(self, img_rgb):
        """Detect pose in RGB image. Returns 33 body landmarks if detected."""
        if not self.is_ready:
            return {"detected": False, "landmarks": []}

        try:
            import numpy as np
            mp = self._mp

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img_rgb))
            result = self._landmarker.detect(mp_image)

            pose_data = {"detected": False, "landmarks": []}

            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                pose_data["detected"] = True
                for lm in result.pose_landmarks[0]:
                    pose_data["landmarks"].append({
                        "x": float(lm.x),
                        "y": float(lm.y),
                        "z": float(lm.z),
                        "visibility": float(lm.visibility) if hasattr(lm, 'visibility') else 0.0,
                    })

            return pose_data

        except Exception as e:
            self.logger.error(f"[POSE] Detection failed: {e}")
            return {"detected": False, "landmarks": []}

    def cleanup(self):
        if self.is_ready and hasattr(self, '_landmarker'):
            try:
                self._landmarker.close()
                self.logger.info("[POSE] Cleaned up")
            except:
                pass