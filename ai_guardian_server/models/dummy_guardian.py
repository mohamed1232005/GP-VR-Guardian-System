# ===========================================================================
# models/dummy_guardian.py — Phase 3: Dummy Guardian Boundary Generator
# Generates a fake guardian boundary in camera coordinates.
# Used to prove the spatial rendering loop before real AI models.
# ===========================================================================

import time


class DummyGuardian:
    """Generates dummy AI_GUARDIAN_DATA messages for testing."""

    def __init__(self):
        self._frame_counter = 0

    def generate(self) -> dict:
        """
        Generate a dummy AI_GUARDIAN_DATA message.
        
        The boundary is a rectangle in camera-space coordinates:
          - Y = 1.2 (floor ~1.2m below camera)
          - Z = 2.0 to 4.0 (2m to 4m in front of camera)
          - X = -1.0 to 1.0 (1m left to 1m right)
        
        Returns:
            dict: A message dict ready to be JSON-serialized.
        """
        self._frame_counter += 1

        message = {
            "type": "AI_GUARDIAN_DATA",
            "frame_id": self._frame_counter,
            "timestamp_ms": int(time.time() * 1000),
            "floor_valid": True,
            "confidence": 1.0,
            "boundary_camera": [
                [-1.0, 1.2, 2.0],
                [ 1.0, 1.2, 2.0],
                [ 1.0, 1.2, 4.0],
                [-1.0, 1.2, 4.0]
            ]
        }

        print(f"[P3_DUMMY] sending AI_GUARDIAN_DATA frame={self._frame_counter}")

        return message
