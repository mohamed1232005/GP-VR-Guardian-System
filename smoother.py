import time
from collections import Counter, deque

import numpy as np

from config import EMA_ALPHA, VOTE_THRESHOLD, VOTE_WINDOW


class LandmarkSmoother:
    def __init__(self, alpha: float = EMA_ALPHA):
        self.alpha = float(alpha)
        self._last = None

    def smooth(self, landmarks) -> np.ndarray:
        current = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

        if self._last is None or self._last.shape != current.shape:
            self._last = current
            return current

        self._last = self.alpha * current + (1.0 - self.alpha) * self._last
        return self._last

    def reset(self) -> None:
        self._last = None


class GestureVoteBuffer:
    def __init__(self, window: int = VOTE_WINDOW, threshold: int = VOTE_THRESHOLD):
        self.window = int(window)
        self.threshold = int(threshold)
        self._items = deque(maxlen=self.window)

    def push(self, gesture: str):
        gesture = gesture or "None"
        self._items.append(gesture)

        counts = Counter(self._items)
        winner, count = counts.most_common(1)[0]

        if winner != "None" and count >= self.threshold:
            return winner

        return None

    def reset(self) -> None:
        self._items.clear()


class GestureCooldown:
    def __init__(self):
        self._last_fired = {}

    def is_allowed(self, gesture: str) -> bool:
        cooldown = self._cooldown_seconds(gesture)
        last = self._last_fired.get(gesture, -9999.0)
        return (time.monotonic() - last) >= cooldown

    def mark_fired(self, gesture: str) -> None:
        self._last_fired[gesture] = time.monotonic()

    def reset(self) -> None:
        self._last_fired.clear()

    @staticmethod
    def _cooldown_seconds(gesture: str) -> float:
        from config import COOLDOWN_FIST, COOLDOWN_OPEN_PALM, COOLDOWN_PINCH

        if gesture == "Pinch":
            return COOLDOWN_PINCH
        if gesture in ("Closed_Fist", "Fist"):
            return COOLDOWN_FIST
        if gesture == "Open_Palm":
            return COOLDOWN_OPEN_PALM
        return 0.5