# ===========================================================================
# debug/fps_meter.py — FPS Counter Utility
# Tracks frames-per-second for any named counter.
# ===========================================================================

import time


class FPSMeter:
    """Simple FPS counter that logs periodically."""

    def __init__(self, name: str, log_interval: float = 5.0):
        self.name = name
        self.log_interval = log_interval

        self._count = 0
        self._last_log_time = time.time()
        self._start_time = time.time()

    def tick(self):
        """Call once per frame/event to count it."""
        self._count += 1
        now = time.time()
        elapsed = now - self._last_log_time

        if elapsed >= self.log_interval:
            fps = self._count / elapsed
            print(f"[FPS] {self.name} fps={fps:.1f}")
            self._count = 0
            self._last_log_time = now

    def get_fps(self) -> float:
        """Get current estimated FPS without logging."""
        now = time.time()
        elapsed = now - self._last_log_time
        if elapsed > 0:
            return self._count / elapsed
        return 0.0
