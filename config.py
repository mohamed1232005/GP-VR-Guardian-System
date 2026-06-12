"""Runtime configuration for the active Mode 2/System A Python service.

Python is only responsible for receiving Unity camera frames, detecting hand
landmarks/gestures, and sending HAND_DATA back to Unity.

Selected constants can be overridden via environment variables prefixed
GUARDIAN_ (e.g. GUARDIAN_TCP_PORT=9100). Invalid values fall back to the
defaults below with a warning, so a bad env var can never crash startup.
"""

import logging
import os

_TRUE_FLAG_VALUES = ("1", "true", "yes", "on")
_FALSE_FLAG_VALUES = ("0", "false", "no", "off")


def _env_int(name: str, default: int) -> int:
    """Read an int from the environment; fall back to default on bad input."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw.strip())
    except ValueError:
        print(f"[CONFIG] invalid int {name}={raw!r}; using default {default}", flush=True)
        return default


def _env_float(name: str, default: float) -> float:
    """Read a float from the environment; fall back to default on bad input."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw.strip())
    except ValueError:
        print(f"[CONFIG] invalid float {name}={raw!r}; using default {default}", flush=True)
        return default


def _env_flag(name: str, default: bool) -> bool:
    """Read a boolean flag from the environment; fall back to default on bad input."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    value = raw.strip().lower()
    if value in _TRUE_FLAG_VALUES:
        return True
    if value in _FALSE_FLAG_VALUES:
        return False
    print(f"[CONFIG] invalid flag {name}={raw!r}; using default {default}", flush=True)
    return default


# Debug logging. Keep False in production. Per-frame logging to the Windows
# console is synchronous and stalls the CV/pose worker threads, adding latency
# and frame-pacing jitter. Startup and error messages always log.
DEBUG_LOG = _env_flag("GUARDIAN_DEBUG_LOG", False)

# Shared logger hierarchy root. server.py attaches the console + rotating file
# handlers at startup; until then records simply fall through (tests/imports
# stay quiet except for WARNING+ via logging's last-resort handler).
_LOGGER = logging.getLogger("guardian")


def dlog(*args, **kwargs):
    """Per-frame debug log. No-op unless DEBUG_LOG is True.

    Routed through logging (logger.debug) so debug traffic lands in the same
    console + rotating-file handlers as everything else. Print-style kwargs
    (e.g. flush) are accepted for backward compatibility and ignored.
    """
    if DEBUG_LOG:
        _LOGGER.debug(" ".join(str(arg) for arg in args))


# Network ports.
TCP_CONTROL_PORT = _env_int("GUARDIAN_TCP_PORT", 9000)
UDP_FRAME_PORT = _env_int("GUARDIAN_UDP_PORT", 9001)

# Hard limits on inbound payloads (memory-exhaustion protection).
# A TCP header declaring more than MAX_TCP_MESSAGE_BYTES closes that client;
# a UDP frame declaring more than MAX_JPEG_BYTES is dropped.
MAX_TCP_MESSAGE_BYTES = _env_int("GUARDIAN_MAX_TCP_MESSAGE_BYTES", 1_048_576)
MAX_JPEG_BYTES = _env_int("GUARDIAN_MAX_JPEG_BYTES", 500_000)

# Liveness heartbeat (MSG_HEARTBEAT 0x16, payload "{}") sent to the connected
# Unity client. Unity treats it as silent liveness traffic.
HEARTBEAT_INTERVAL_SECONDS = _env_float("GUARDIAN_HEARTBEAT_INTERVAL_SECONDS", 2.0)

# Delay before a crashed CV/pose worker is rebuilt and restarted.
WORKER_RESTART_DELAY_SECONDS = _env_float("GUARDIAN_WORKER_RESTART_DELAY_SECONDS", 1.0)

# Rotating server log file (written beside server.py by server.setup_logging).
LOG_FILE_NAME = os.environ.get("GUARDIAN_LOG_FILE", "guardian_server.log")
LOG_FILE_MAX_BYTES = _env_int("GUARDIAN_LOG_FILE_MAX_BYTES", 1_048_576)
LOG_FILE_BACKUP_COUNT = _env_int("GUARDIAN_LOG_FILE_BACKUP_COUNT", 3)

# Latest-frame queue size. Keep this at 1 for low-latency tracking.
FRAME_QUEUE_MAXSIZE = 1

# Landmark smoothing and gesture confirmation.
EMA_ALPHA = 0.45
VOTE_WINDOW = 5
VOTE_THRESHOLD = 3

# MediaPipe confidence thresholds.
HAND_DETECT_CONF = 0.55
HAND_TRACK_CONF = 0.50
HAND_PRESENCE_CONF = 0.55

# Custom landmark gesture thresholds.
POINT_EXTENDED_MIN = 0.62
POINT_CURLED_MAX = 0.52
PINCH_MAX_RATIO = 0.46
PINCH_INDEX_MIN_RATIO = 0.22
OPEN_EXTEND_MARGIN = 0.08
FIST_TIP_MCP_MAX_RATIO = 0.72

# Body-pose pipeline (laptop webcam -> MediaPipe Pose).
# This pipeline is fully independent of the hand pipeline: its own capture
# thread, queue, worker, and detector. Python stays stateless CV and emits
# only raw landmarks; Unity owns all rehab interpretation (form/reps/holds).
#
# DISABLED ON PURPOSE: the "second camera" (laptop webcam / BlazePose) is turned
# off for now. With this False, server.py never opens the webcam and never starts
# the pose capture/worker threads, so the hand pipeline gets the full CPU. The
# Unity-side BodyPoseReceiver simply receives no BODY_POSE messages and stays idle.
# To re-enable the body-pose feature later, set this back to True (no other change
# needed) and make sure pose_landmarker.task sits beside server.py.
POSE_ENABLED = False
POSE_CAMERA_INDEX = 0
POSE_TARGET_FPS = 15
POSE_FRAME_QUEUE_MAXSIZE = 1
POSE_CAPTURE_WIDTH = 640
POSE_CAPTURE_HEIGHT = 480
POSE_DETECT_CONF = 0.5
POSE_PRESENCE_CONF = 0.5
POSE_TRACK_CONF = 0.5
POSE_EMA_ALPHA = 0.4          # Separate from hand EMA_ALPHA so each tunes alone.
POSE_NUM_LANDMARKS = 33
POSE_MODEL_FILENAME = "pose_landmarker.task"
# Optional raw landmark logging (pure CV recording, not interpretation).
POSE_RECORD_RAW = False
POSE_RECORD_FILENAME = "pose_raw_log.jsonl"
