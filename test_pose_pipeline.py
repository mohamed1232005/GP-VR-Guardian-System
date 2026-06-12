"""Test suite for the body-pose pipeline and its wire protocol.

Run from the GP-VR-Guardian-System folder:
    .venv\\Scripts\\python.exe -m unittest test_pose_pipeline -v

These tests validate everything that can be checked without Unity or a webcam:
- the TCP wire format Unity's BodyPoseReceiver depends on (>IH header, type 0x15,
  JSON keys, 33x4 landmark rows),
- the Python-side data shaping (_build_body_pose),
- LandmarkSmoother reuse for 33 pose joints,
- PoseDetector behaviour with a mocked MediaPipe landmarker,
- the drop-oldest capture queue,
- a reference implementation of the Unity rep/form math, proving the algorithm.
"""

import json
import queue
import struct
import unittest
from unittest import mock

import numpy as np

import pose_tracking
import transport
from transport import _pack_message, MSG_BODY_POSE, _TCP_HEADER_FMT, _TCP_HEADER_SIZE


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #

class _FakeLandmark:
    def __init__(self, x, y, z, visibility):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


class _FakeResult:
    def __init__(self, pose_landmarks):
        self.pose_landmarks = pose_landmarks


def _make_landmarks(n=33, vis=0.9):
    return [_FakeLandmark(i / 100.0, i / 50.0, i / 200.0, vis) for i in range(n)]


# --------------------------------------------------------------------------- #
# Wire protocol
# --------------------------------------------------------------------------- #

class TestWireProtocol(unittest.TestCase):
    def test_body_pose_type_id(self):
        self.assertEqual(MSG_BODY_POSE, 0x15)
        self.assertEqual(transport._TYPE_TO_ID["BODY_POSE"], 0x15)

    def test_pack_and_decode_roundtrip(self):
        """A packed BODY_POSE must decode exactly the way the test client / Unity reads it."""
        payload = {
            "type": "BODY_POSE",
            "landmarks": [[0.1, 0.2, -0.3, 0.95] for _ in range(33)],
            "frame_timestamp_ms": 123456,
            "tracked": True,
        }
        packed = _pack_message(MSG_BODY_POSE, payload)

        # Header is big-endian uint32 length + uint16 type id.
        length, type_id = struct.unpack(_TCP_HEADER_FMT, packed[:_TCP_HEADER_SIZE])
        body = packed[_TCP_HEADER_SIZE:]

        self.assertEqual(type_id, 0x15)
        self.assertEqual(length, len(body))

        decoded = json.loads(body.decode("utf-8"))
        self.assertEqual(decoded["type"], "BODY_POSE")
        self.assertTrue(decoded["tracked"])
        self.assertEqual(decoded["frame_timestamp_ms"], 123456)
        self.assertEqual(len(decoded["landmarks"]), 33)
        self.assertTrue(all(len(row) == 4 for row in decoded["landmarks"]))

    def test_result_dispatcher_routes_body_pose(self):
        """response['type'] == 'BODY_POSE' must map to 0x15 (not the WARNING fallback)."""
        type_id = transport._TYPE_TO_ID.get({"type": "BODY_POSE"}["type"], transport.MSG_WARNING)
        self.assertEqual(type_id, transport.MSG_BODY_POSE)


# --------------------------------------------------------------------------- #
# _build_body_pose
# --------------------------------------------------------------------------- #

class TestBuildBodyPose(unittest.TestCase):
    def test_tracked_shape(self):
        xyz = np.arange(33 * 3, dtype=np.float32).reshape(33, 3)
        vis = np.full(33, 0.9, dtype=np.float32)

        out = pose_tracking._build_body_pose(xyz, vis, 555)

        self.assertEqual(out["type"], "BODY_POSE")
        self.assertTrue(out["tracked"])
        self.assertEqual(out["frame_timestamp_ms"], 555)
        self.assertEqual(len(out["landmarks"]), 33)
        for i, row in enumerate(out["landmarks"]):
            self.assertEqual(len(row), 4)  # [x, y, z, visibility]
            self.assertAlmostEqual(row[3], 0.9, places=5)
            self.assertTrue(all(isinstance(v, float) for v in row))

    def test_untracked(self):
        out = pose_tracking._build_body_pose(None, None, 777)
        self.assertFalse(out["tracked"])
        self.assertEqual(out["landmarks"], [])
        self.assertEqual(out["frame_timestamp_ms"], 777)

    def test_json_serialisable(self):
        xyz = np.zeros((33, 3), dtype=np.float32)
        vis = np.zeros(33, dtype=np.float32)
        out = pose_tracking._build_body_pose(xyz, vis, 1)
        json.dumps(out)  # must not raise (no numpy types leak through)


# --------------------------------------------------------------------------- #
# LandmarkSmoother reuse for 33 joints
# --------------------------------------------------------------------------- #

class TestLandmarkSmootherReuse(unittest.TestCase):
    def test_smooths_33_landmarks(self):
        sm = pose_tracking.LandmarkSmoother(alpha=0.5)

        first = sm.smooth(_make_landmarks(33))
        self.assertEqual(first.shape, (33, 3))

        # Second frame at different positions -> EMA blends toward the new values.
        moved = [_FakeLandmark(1.0, 1.0, 1.0, 0.9) for _ in range(33)]
        second = sm.smooth(moved)
        self.assertEqual(second.shape, (33, 3))
        # With alpha 0.5 the result is halfway between first and 1.0.
        expected = 0.5 * 1.0 + 0.5 * first[0][0]
        self.assertAlmostEqual(second[0][0], expected, places=5)

    def test_reset(self):
        sm = pose_tracking.LandmarkSmoother(alpha=0.5)
        sm.smooth(_make_landmarks(33))
        sm.reset()
        # After reset the next frame passes through unchanged (no prior state).
        out = sm.smooth(_make_landmarks(33))
        self.assertEqual(out.shape, (33, 3))


# --------------------------------------------------------------------------- #
# PoseDetector with a mocked MediaPipe landmarker
# --------------------------------------------------------------------------- #

class TestPoseDetector(unittest.TestCase):
    def _make_detector(self, fake_landmarker):
        with mock.patch("os.path.exists", return_value=True), \
             mock.patch.object(
                 pose_tracking.mp_vision.PoseLandmarker,
                 "create_from_options",
                 return_value=fake_landmarker):
            return pose_tracking.PoseDetector("dummy.task")

    def test_tracked_frame(self):
        fake = mock.Mock()
        fake.detect_for_video.return_value = _FakeResult([_make_landmarks(33)])
        det = self._make_detector(fake)

        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        out = det.process_frame(rgb, 100)

        self.assertEqual(out["type"], "BODY_POSE")
        self.assertTrue(out["tracked"])
        self.assertEqual(len(out["landmarks"]), 33)
        fake.detect_for_video.assert_called_once()

    def test_untracked_frame(self):
        fake = mock.Mock()
        fake.detect_for_video.return_value = _FakeResult([])  # no pose
        det = self._make_detector(fake)

        out = det.process_frame(np.zeros((8, 8, 3), dtype=np.uint8), 100)
        self.assertFalse(out["tracked"])
        self.assertEqual(out["landmarks"], [])

    def test_monotonic_timestamp_guard(self):
        fake = mock.Mock()
        fake.detect_for_video.return_value = _FakeResult([_make_landmarks(33)])
        det = self._make_detector(fake)

        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        first = det.process_frame(rgb, 100)
        # Feed a NON-increasing timestamp; the detector must bump it forward.
        second = det.process_frame(rgb, 50)
        self.assertGreater(second["frame_timestamp_ms"], first["frame_timestamp_ms"])


# --------------------------------------------------------------------------- #
# Capture queue drop-oldest behaviour
# --------------------------------------------------------------------------- #

class TestCaptureQueue(unittest.TestCase):
    def test_put_latest_keeps_only_newest(self):
        q = queue.Queue(maxsize=1)
        pose_tracking._put_latest_pose(q, {"timestamp_ms": 1})
        pose_tracking._put_latest_pose(q, {"timestamp_ms": 2})

        self.assertEqual(q.qsize(), 1)
        item = q.get_nowait()
        self.assertEqual(item["timestamp_ms"], 2)  # oldest dropped


# --------------------------------------------------------------------------- #
# Reference implementation of the Unity rep/form math (BodyPoseReceiver mirror)
# --------------------------------------------------------------------------- #

class _RepCounterReference:
    """Mirror of BodyPoseReceiver's reach hysteresis state machine and spine angle.

    Kept deliberately tiny and identical in shape to the C# so the algorithm itself
    can be validated here even though the C# cannot run in this environment.
    """

    def __init__(self, reach_high=1.25, reach_low=0.85, spine_max_deg=25.0):
        self.reach_high = reach_high
        self.reach_low = reach_low
        self.spine_max_deg = spine_max_deg
        self.rep_count = 0
        self._state = "returned"

    @staticmethod
    def spine_angle_deg(shoulder_mid, hip_mid):
        import math
        tx = shoulder_mid[0] - hip_mid[0]
        ty = shoulder_mid[1] - hip_mid[1]
        return math.degrees(math.atan2(abs(tx), abs(ty)))

    def feed(self, reach_ratio):
        if self._state == "returned":
            if reach_ratio >= self.reach_high:
                self._state = "extended"
        elif self._state == "extended":
            if reach_ratio <= self.reach_low:
                self.rep_count += 1
                self._state = "returned"
        return self.rep_count


class TestRepFormReference(unittest.TestCase):
    def test_three_full_cycles_count_three_reps(self):
        rc = _RepCounterReference()
        # Three extend->return cycles, with sub-threshold noise that must not double-count.
        sequence = [0.5, 1.3, 1.4, 0.9, 0.7,   # rep 1
                    0.6, 1.5, 1.0, 0.8,         # rep 2
                    0.5, 1.26, 0.84]            # rep 3
        for r in sequence:
            rc.feed(r)
        self.assertEqual(rc.rep_count, 3)

    def test_hysteresis_prevents_double_count(self):
        rc = _RepCounterReference()
        # Oscillating between the thresholds (never returning below reach_low) = no new reps.
        for r in [1.3, 1.1, 1.3, 1.1, 1.3]:
            rc.feed(r)
        self.assertEqual(rc.rep_count, 0)

    def test_upright_vs_leaning_spine_angle(self):
        rc = _RepCounterReference()
        # Image-space: nearly vertical torso -> small angle (good form).
        upright = rc.spine_angle_deg((0.5, 0.2), (0.5, 0.6))
        self.assertLess(upright, rc.spine_max_deg)
        # Strong forward lean -> large angle (warning).
        leaning = rc.spine_angle_deg((0.5, 0.3), (0.2, 0.4))
        self.assertGreater(leaning, rc.spine_max_deg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
