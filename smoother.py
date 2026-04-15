"""Temporal smoothing utilities: LandmarkSmoother, GestureVoteBuffer, GestureCooldown."""

import time
from collections import deque

import numpy as np

from config import (
    EMA_ALPHA, VOTE_WINDOW, VOTE_THRESHOLD,
    COOLDOWN_PINCH, COOLDOWN_FIST, COOLDOWN_OPEN_PALM,
)


class LandmarkSmoother:
    """Exponential moving average over 21 hand landmarks (x, y, z)."""

    def __init__(self, alpha: float = EMA_ALPHA):
        self.alpha = alpha
        self.prev  = None  # np.array (21, 3)

    def smooth(self, landmarks) -> np.ndarray:
        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
        if self.prev is None:
            self.prev = coords
        else:
            self.prev = self.alpha * coords + (1.0 - self.alpha) * self.prev
        return self.prev  # shape (21, 3)

    def reset(self):
        self.prev = None


class GestureVoteBuffer:
    """Majority-vote buffer — gesture must appear threshold/window times to confirm."""

    def __init__(self, window: int = VOTE_WINDOW, threshold: int = VOTE_THRESHOLD):
        self.threshold = threshold
        self.history   = deque(maxlen=window)

    def push(self, gesture: str) -> str | None:
        self.history.append(gesture)
        counts: dict[str, int] = {}
        for g in self.history:
            counts[g] = counts.get(g, 0) + 1
        best = max(counts, key=counts.get)
        return best if counts[best] >= self.threshold else None

    def reset(self):
        self.history.clear()


class GestureCooldown:
    """Per-gesture cooldown to prevent double-firing."""

    _COOLDOWNS = {
        "Pinch"      : COOLDOWN_PINCH,
        "Closed_Fist": COOLDOWN_FIST,
        "Open_Palm"  : COOLDOWN_OPEN_PALM,
        "POINT"      : 0.0,
        "NONE"       : 0.0,
    }

    def __init__(self):
        self.last_fired: dict[str, float] = {}

    def is_allowed(self, gesture: str) -> bool:
        cd    = self._COOLDOWNS.get(gesture, 0.0)
        since = time.monotonic() - self.last_fired.get(gesture, 0.0)
        return since >= cd

    def mark_fired(self, gesture: str):
        self.last_fired[gesture] = time.monotonic()

    def reset(self):
        self.last_fired.clear()