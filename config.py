"""Runtime configuration for the active Mode 2/System A Python service.

Python is only responsible for receiving Unity camera frames, detecting hand
landmarks/gestures, and sending HAND_DATA back to Unity.
"""

# Network ports.
TCP_CONTROL_PORT = 9000
UDP_FRAME_PORT = 9001

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
