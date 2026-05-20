# ===========================================================================
# config.py — AI Guardian Server Configuration (Phase 7B)
# ===========================================================================

# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
UDP_HOST = "0.0.0.0"
UDP_PORT = 9001
MAX_UDP_PACKET = 65535

TCP_HOST = "0.0.0.0"
TCP_PORT = 9000

DUMMY_SEND_INTERVAL = 1.0

# ---------------------------------------------------------------------------
# Phase 4: Depth estimation
# ---------------------------------------------------------------------------
DEPTH_ENABLED            = True
DEPTH_TARGET_FPS         = 1.0
DEPTH_INPUT_WIDTH        = 320
DEPTH_INPUT_HEIGHT       = 240

DEPTH_MODEL_NAME         = "depth_anything_v2_small"
DEPTH_USE_METRIC_INDOOR  = True
DEPTH_ALLOW_DUMMY        = False
DEPTH_TRY_DEPTH_PRO      = False

# ---------------------------------------------------------------------------
# Phase 5: Floor segmentation + boundary
# ---------------------------------------------------------------------------
BOUNDARY_MODE            = "ai_floor"
ALLOW_DUMMY_GUARDIAN     = False

SEG_MODEL_NAME           = "nvidia/segformer-b0-finetuned-ade-512-512"
PIPELINE_TARGET_FPS      = 1.0

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
MIN_DEPTH_M              = 0.1
MAX_DEPTH_M              = 10.0
MAX_BACKPROJECT_POINTS   = 5000

RANSAC_ITERATIONS           = 200
RANSAC_INLIER_THRESHOLD_M   = 0.05
PLANE_MIN_INLIERS           = 500
PLANE_MIN_INLIER_RATIO      = 0.45
PLANE_MAX_RMSE_M            = 0.08

# Legacy boundary extents (used as fallback caps)
BOUNDARY_WIDTH_M            = 2.0
BOUNDARY_DEPTH_M            = 2.0
BOUNDARY_NEAR_Z_M           = 1.0
BOUNDARY_FAR_Z_M            = 3.0
BOUNDARY_SAFETY_MARGIN_M    = 0.15

# ============================================================
# Phase 7B: Floor-based guardian (NO fixed-size, NO room modes)
# ============================================================
# Guardian area is derived ONLY from detected floor.
# No fixed-size fallback, no SmallRoom/NormalRoom/OpenArea modes.
GUARDIAN_FIXED_SIZE_ENABLED  = False

# Fixed playable area in meters (legacy, NOT used when FIXED_SIZE=False)
GUARDIAN_WIDTH_M             = 1.8
GUARDIAN_DEPTH_M             = 1.6
GUARDIAN_CENTER_Z_M          = 2.0
GUARDIAN_CENTER_X_M          = 0.0

# Floor lift above plane to avoid Z-fighting
GUARDIAN_FLOOR_LIFT_M        = 0.015

# ============================================================
# Phase 7B: Floor candidate limits
# ============================================================
# Phase 9.1: Boundary must fit within these absolute limits.
# System generates 3 candidates from real floor data.
# Smart Safe Rectangle: min 1.0m, max 3.0m, centered on user.
FLOOR_CANDIDATE_MIN_W       = 1.0   # meters (minimum safe width)
FLOOR_CANDIDATE_MIN_D       = 1.0   # meters (minimum safe depth)
FLOOR_CANDIDATE_MAX_W       = 3.0   # meters (maximum boundary width)
FLOOR_CANDIDATE_MAX_D       = 3.0   # meters (maximum boundary depth)
FLOOR_CANDIDATE_SAFETY_M    = 0.15  # safety margin inward from floor edges
FLOOR_CANDIDATE_COUNT       = 3     # conservative, medium, largest

# ---------------------------------------------------------------------------
# Scan gate (loose)
# ---------------------------------------------------------------------------
FLOOR_MIN_PIXEL_RATIO       = 0.005
LOWER_IMAGE_ROI             = 0.45
PLANE_MIN_HORIZONTAL_SCORE  = 0.60

# ---------------------------------------------------------------------------
# Lock gate (strict — Phase 6 values)
# ---------------------------------------------------------------------------
LOCK_MIN_HORIZONTAL_SCORE   = 0.85
LOCK_MIN_FLOOR_RATIO        = 0.05
LOCK_MIN_INLIER_RATIO       = 0.85
LOCK_MAX_RMSE               = 0.035
LOCK_MIN_CONFIDENCE         = 0.85

# ---------------------------------------------------------------------------
# Fallback policy
# ---------------------------------------------------------------------------
USE_FALLBACK_FLOOR_MASK     = False
ALLOW_FALLBACK_LOCK         = False

# ---------------------------------------------------------------------------
# Candidate pool
# ---------------------------------------------------------------------------
CANDIDATE_POOL_SIZE                         = 8
MAX_CONSECUTIVE_INVALID_BEFORE_POOL_RESET   = 5

# ---------------------------------------------------------------------------
# Image orientation
# ---------------------------------------------------------------------------
INPUT_ROTATION_DEGREES  = 270
TEST_ALL_ROTATIONS      = True

# ---------------------------------------------------------------------------
# Debug image saving
# ---------------------------------------------------------------------------
SAVE_DEBUG_IMAGES       = True
DEBUG_IMAGE_EVERY_N     = 10

# ---------------------------------------------------------------------------
# CSV session logging
# ---------------------------------------------------------------------------
WRITE_DEBUG_CSV         = True

# ============================================================
# Phase 7A: Hand tracking (MediaPipe)
# ============================================================
HAND_TRACKING_ENABLED       = True
HAND_TRACKING_TARGET_FPS    = 15.0   # Phase 9.2: 15 Hz minimum for smooth ray interaction (was 5.0)
# Phase 9.21: detection threshold dropped 0.50 → 0.30 to recover the empty-stream
# session. Tracking stays a bit higher so we don't ping-pong on jittery frames.
# If P7_HAND_NONE persists at 0.30, the root cause is not confidence — see the
# saved hand_none_*.jpg under ai_guardian_server/debug/hand_dumps/ for what
# MediaPipe is actually being shown (rotation/color issue).
HAND_MIN_DETECTION_CONF     = 0.30
HAND_MIN_TRACKING_CONF      = 0.40
HAND_MAX_NUM_HANDS          = 1

# ============================================================
# Phase 7B: Placement mode
# ============================================================
# "user_confirm" = Python sends ready_to_confirm, Unity waits for user Build button.
# "auto_lock"    = Phase 6 legacy behavior (automatic lock after candidate pool fills).
PLACEMENT_MODE              = "auto_lock"

# ============================================================
# Phase 9: Post-lock optimization
# ============================================================
# After guardian locks, stop expensive depth/segmentation.
# Optionally run diagnostics at very low FPS.
POST_LOCK_ENABLED           = True
POST_LOCK_DIAGNOSTIC_FPS    = 0.1   # 1 frame per 10 seconds
POST_LOCK_STOP_DEPTH        = True
POST_LOCK_STOP_SEGMENTATION = True
POST_LOCK_STOP_PLANE_FIT    = True
POST_LOCK_KEEP_HAND         = True

# ---------------------------------------------------------------------------
# Legacy stability (kept for backward compatibility)
# ---------------------------------------------------------------------------
STABLE_REQUIRED             = 5
STABLE_MAX_ANGLE_DIFF       = 10.0
STABLE_MAX_DRIFT_M          = 0.15
STABLE_MIN_CONFIDENCE       = 0.75
STABLE_MAX_RMSE             = 0.04
STABLE_MIN_INLIER_RATIO     = 0.70