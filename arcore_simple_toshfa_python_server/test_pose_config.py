"""Phase 2+3 tests for pose-pipeline enablement and webcam-failure safety.

Run from the GP-VR-Guardian-System folder:
    .venv\\Scripts\\python.exe -m unittest test_pose_config -v

These validate the config/env contract and that a missing webcam cannot crash the
capture thread -- without Unity, a webcam, or the MediaPipe model.
"""

import importlib
import os
import queue
import unittest
from unittest import mock


class TestPoseEnablement(unittest.TestCase):
    """POSE_ENABLED defaults on and is overridable via GUARDIAN_POSE_ENABLED."""

    def tearDown(self):
        # Always restore config to its env-default state for later tests/modules.
        import config
        os.environ.pop("GUARDIAN_POSE_ENABLED", None)
        importlib.reload(config)

    def test_pose_enabled_default_true(self):
        import config
        os.environ.pop("GUARDIAN_POSE_ENABLED", None)
        importlib.reload(config)
        self.assertTrue(config.POSE_ENABLED)

    def test_pose_disabled_by_env_zero(self):
        import config
        with mock.patch.dict(os.environ, {"GUARDIAN_POSE_ENABLED": "0"}):
            importlib.reload(config)
            self.assertFalse(config.POSE_ENABLED)

    def test_pose_enabled_by_env_one(self):
        import config
        with mock.patch.dict(os.environ, {"GUARDIAN_POSE_ENABLED": "1"}):
            importlib.reload(config)
            self.assertTrue(config.POSE_ENABLED)

    def test_camera_index_override(self):
        import config
        with mock.patch.dict(os.environ, {"GUARDIAN_POSE_CAMERA_INDEX": "1"}):
            importlib.reload(config)
            self.assertEqual(config.POSE_CAMERA_INDEX, 1)

    def test_env_flag_parsing(self):
        import config
        with mock.patch.dict(os.environ, {"X_FLAG": "0"}):
            self.assertFalse(config._env_flag("X_FLAG", True))
        with mock.patch.dict(os.environ, {"X_FLAG": "yes"}):
            self.assertTrue(config._env_flag("X_FLAG", False))
        os.environ.pop("X_FLAG", None)
        self.assertTrue(config._env_flag("X_FLAG", True))   # falls back to default


class TestWebcamFailureSafe(unittest.TestCase):
    """A webcam that cannot open must not raise out of the capture thread."""

    def test_capture_returns_cleanly_when_camera_unavailable(self):
        import cv2
        import pose_tracking

        fake_cap = mock.Mock()
        fake_cap.isOpened.return_value = False  # simulate no/unavailable webcam

        with mock.patch.object(cv2, "VideoCapture", return_value=fake_cap), \
             mock.patch.object(pose_tracking.time, "sleep", return_value=None):
            # Must return (not raise/hang) when the device cannot be opened.
            pose_tracking.run_pose_capture(queue.Queue(maxsize=1), lambda: "client")

        fake_cap.release.assert_called()  # device handle released on the failure path


if __name__ == "__main__":
    unittest.main(verbosity=2)
