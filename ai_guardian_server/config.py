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
# Phase 10.6: Floor detection mode  (HYBRID = SegFormer + depth/geometry)
#
# SegFormer-b0 (ADE20K) produces correct floor masks on many frames, but
# systematically mislabels glossy/patterned tile floors as "bed"/"wall"/"sky"
# (floor_prob ~0.005-0.06 on a floor filling ~50% of the frame). The old pipeline
# GATED all geometry on that semantic label, so on tile the guardian never locked
# (perpetual "semantic_floor_too_low").
#
# The floor is also the dominant near-horizontal PLANE in the lower image — a
# property the depth model (Depth-Anything V2) captures reliably even on
# reflective tile. So we FUSE both signals instead of trusting either alone:
#   - geometry plane = robust backbone (works on tile where SegFormer fails),
#   - SegFormer floor = unioned in where it agrees with the plane (recovers
#     detail/edges, boosts confidence), and is the FALLBACK when depth is
#     degenerate but SegFormer nailed the floor.
# Semantic is used additively + as agreement/fallback only — never to subtract
# floor (the tile mislabel is too inconsistent to trust for removal).
#
#   "hybrid"         : SegFormer + depth-plane fusion (DEFAULT, recommended).
#   "geometry_first" : depth-plane only (diagnostic; does NOT load SegFormer).
#   "semantic"       : legacy SegFormer class-label path (kept for comparison).
# ---------------------------------------------------------------------------
FLOOR_DETECTION_MODE        = "hybrid"
GEO_FLOOR_ROI_TOP_FRAC      = 0.40   # ignore pixels above this frac of image height
GEO_FLOOR_INLIER_THRESH_M   = 0.06   # max distance (m) from plane to count as floor
GEO_FLOOR_MIN_CAM_HORIZ     = 0.35   # min |normal.y| in camera space (rejects walls)
GEO_FLOOR_SEMANTIC_BAND_MULT = 2.0   # semantic-union plane band = mult x inlier_thresh
# Semantic fallback: if geometry finds NO horizontal plane but SegFormer is
# confident, use the semantic floor mask instead of rejecting the frame.
HYBRID_SEM_FALLBACK_RATIO   = 0.08   # min SegFormer floor ratio to trust as fallback
HYBRID_SEM_FALLBACK_CONF    = 0.30   # min SegFormer floor confidence to trust
# Phase 14: trust HIGH-confidence SegFormer floor pixels STRONGLY. Pixels whose
# per-pixel floor probability is >= HYBRID_SEM_HIGH_CONF are unioned into the
# floor mask within a WIDER plane band (HYBRID_SEM_HIGH_BAND_MULT × inlier_thresh)
# than ordinary semantic pixels — so confident floor is recovered even where the
# depth plane fit is a little noisy, while the band still blocks confident
# mislabels that sit far off the trusted RANSAC plane.
HYBRID_SEM_HIGH_CONF        = 0.60   # per-pixel floor prob to count as "high confidence"
HYBRID_SEM_HIGH_BAND_MULT   = 2.5    # plane band (×inlier_thresh) for high-conf pixels

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
# Phase 2I/2K: 5×5 room support. The LOCKED cube is built from the world
# accumulator (capped at FLOOR_ACCUM_MAX_SIZE_M=5.0), NOT from this smart-cube
# candidate — but the per-frame smart cube still gates acceptance and feeds
# confidence, so its cap must match the accumulator or it artificially caps the
# usable floor at 3 m. Raised 3.0 → 5.0 to align with the accumulator.
FLOOR_CANDIDATE_MAX_W       = 5.0   # meters (maximum boundary width)
FLOOR_CANDIDATE_MAX_D       = 5.0   # meters (maximum boundary depth)
FLOOR_CANDIDATE_SAFETY_M    = 0.08  # Phase 10.1: was 0.15. 30cm total shrink
                                    # made every handheld scan look like a too-small
                                    # floor. 0.08 keeps a useful margin (16cm total)
                                    # while letting realistic 1.2-1.8m floors pass.
FLOOR_CANDIDATE_COUNT       = 3     # conservative, medium, largest

# ---------------------------------------------------------------------------
# Scan gate (loose)
# ---------------------------------------------------------------------------
FLOOR_MIN_PIXEL_RATIO       = 0.005
LOWER_IMAGE_ROI             = 0.45
# Phase 10.1: PLANE_MIN_HORIZONTAL_SCORE 0.60 → 0.40. Handheld phone tilts and
# bobs ~15-30° during scan; session_log.csv showed valid floors scoring 0.25-0.50
# rejected as `not_horizontal`. 0.40 still rejects truly vertical surfaces (>40°
# off horizontal would score <0.40 robustly).
PLANE_MIN_HORIZONTAL_SCORE  = 0.40

# ---------------------------------------------------------------------------
# Lock gate (Phase 10.1 — loosened for real-world handheld scanning)
#
# Previous Phase-6 values were calibrated against a stationary tripod with a
# fully-lit indoor floor. On a handheld phone scanning a partially-visible
# floor they reject every frame for the entire session. Lowered conservatively:
#   - horizontal_score: 0.85 → 0.55   (still rejects walls, accepts tilted scans)
#   - inlier_ratio:     0.85 → 0.65   (allows depth noise / partial occlusion)
#   - max_rmse:         0.035 → 0.060 (depth jitter on handheld is ~5cm)
#   - confidence:       0.85 → 0.55   (semantic model is less confident on cluttered floors)
#   - floor_ratio:      0.05 → 0.025  (small visible floor patches still count)
# The candidate-pool of size 8 still requires 8 valid frames to converge before
# emitting a lock, so a single bad frame can't trigger a wrong lock.
# ---------------------------------------------------------------------------
LOCK_MIN_HORIZONTAL_SCORE   = 0.55
LOCK_MIN_FLOOR_RATIO        = 0.025
LOCK_MIN_INLIER_RATIO       = 0.65
LOCK_MAX_RMSE               = 0.060
LOCK_MIN_CONFIDENCE         = 0.55

# ---------------------------------------------------------------------------
# Phase 2K: Unity-space coordinate sanity (req 7). Server-side last-line check
# run BEFORE a lock is allowed. The accumulated rectangle is flat by
# construction, so y_spread here is a structural guard; the floor must also sit
# BELOW the camera (rejects a ceiling/wall captured under a wrong rotation). The
# AUTHORITATIVE max(y)-min(y) corner spread is re-checked on the Unity side
# (AIGuardianReceiver.ValidateCoordinateSanity) after the device re-applies its
# true pose. This gate can only BLOCK a lock — it never forces one.
UNITY_LOCK_MAX_Y_SPREAD_M   = 0.18   # 0.15–0.20 window; reject lock above this

# ---------------------------------------------------------------------------
# Phase 2H: HEADSET lock thresholds REVERTED to the strict handheld values. The
# earlier 0.45 relaxation was a band-aid for a LOW horizontal_score that was
# actually caused by a reversed/mis-rotated frame feeding the geometry a tilted
# plane. The real fix is to pick the orientation that makes the plane FLAT
# (highest horizontal_score) — see the geometry rotation probe — after which a
# real floor scores ~0.9 and passes the normal 0.55 gate. Kept equal to the
# normal gate so headset_mode no longer relaxes anything.
# ---------------------------------------------------------------------------
HEADSET_LOCK_MIN_HORIZONTAL_SCORE = 0.55
HEADSET_LOCK_MIN_CONFIDENCE       = 0.55

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

# ===========================================================================
# Phase 14: Multi-frame WORLD-space floor accumulation (pre-lock)
#
# Instead of locking the single best-scored per-frame camera-space rectangle,
# accumulate every trusted floor frame into a world-space occupancy grid and
# build the final cubic from ALL points seen during scanning:
#   - each accepted frame's RANSAC inliers are transformed to world via the
#     frame's camera_to_world pose,
#   - only points within FLOOR_ACCUM_PLANE_BAND_M of the trusted floor Y are
#     kept (near the RANSAC plane),
#   - cells below FLOOR_ACCUM_MIN_CELL_HITS are dropped as noise, and only the
#     largest connected floor blob is used (outlier rejection),
#   - the rectangle is the oriented min-area box of that blob, clamped to
#     FLOOR_ACCUM_MAX_SIZE_M × FLOOR_ACCUM_MAX_SIZE_M,
#   - a lock is only emitted once center / rotation / floorY / width / depth
#     have been stable across FLOOR_ACCUM_STABILITY_WINDOW frames.
# After lock the rectangle is frozen and Python never updates the boundary.
# ===========================================================================
FLOOR_ACCUM_ENABLED             = True
FLOOR_ACCUM_CELL_M              = 0.05   # world XZ occupancy cell size (m)
FLOOR_ACCUM_MAX_SIZE_M          = 5.0    # hard cap on final width AND depth (m) — 5x5 room
FLOOR_ACCUM_MIN_CELL_HITS       = 2      # a cell counts as floor after N hits (outlier reject)
FLOOR_ACCUM_PLANE_BAND_M        = 0.10   # keep points within this PERPENDICULAR distance
                                         # of the tilted floor plane (absorbs RANSAC
                                         # inlier thresh + depth noise; captures the full
                                         # near+far floor, not just the lowest strip)
# Phase 2N: relaxed so a BIG preview locks quickly instead of stalling in
# preview. These were tuned for a tripod; on a handheld scan of a large room the
# old 6-frame/150-cell/strict-tolerance gate rarely all aligned at once.
FLOOR_ACCUM_MIN_FRAMES          = 4      # min accepted frames before a lock is allowed (was 6)
FLOOR_ACCUM_MIN_CELLS           = 80     # min occupied floor cells before lock (was 150)
FLOOR_ACCUM_MIN_AREA_M2         = 1.0    # min rectangle area before lock
FLOOR_ACCUM_STABILITY_WINDOW    = 4      # frames the rectangle must hold stable (was 5)
FLOOR_ACCUM_CENTER_TOL_M        = 0.15   # max center drift across the window (was 0.10)
FLOOR_ACCUM_YAW_TOL_DEG         = 10.0   # max rotation drift, mod 90° (was 6.0)
FLOOR_ACCUM_FLOORY_TOL_M        = 0.08   # max floor-Y drift (was 0.05)
FLOOR_ACCUM_SIZE_TOL_M          = 0.15   # max width/depth drift
FLOOR_ACCUM_MAX_CELLS           = 40000  # memory cap on the occupancy grid
FLOOR_ACCUM_RESET_AFTER_INVALID = 25     # reset accumulation after this many lost frames
# Area-growth gate — DO NOT lock on the first small stable rectangle. Keep the
# LARGEST floor seen and only lock once the area has stopped growing (the user
# has finished scanning), so the cube fills the real floor, not an early patch.
FLOOR_ACCUM_AREA_GROWTH_FRAMES  = 6      # area must not grow for this many accepted frames
FLOOR_ACCUM_AREA_GROWTH_EPS_M2  = 0.06   # growth smaller than this (m²) counts as "plateaued"

# ---------------------------------------------------------------------------
# Phase 2L: ROLLING lock (fixes "preview shows but never locks").
#
# The strict auto-lock above required, on ONE frame, BOTH the area to have
# plateaued for AREA_GROWTH_FRAMES *and* full 5-axis stability INCLUDING
# width/depth. In a real/large room the area keeps growing while the user sweeps,
# so width/depth drift > SIZE_TOL and the growth counter keeps resetting → the
# two conditions are never satisfied together → it never leaves preview.
#
# Rolling mode decouples them: it locks once the cube's PLACEMENT (center + yaw +
# floorY) is steady AND EITHER the area has plateaued OR enough confident frames
# have been collected (LOCK_MAX_WAIT_FRAMES). Size is allowed to keep growing
# until placement settles, so the lock captures the FULL trusted area, and the
# MAX_WAIT bound guarantees a large room still locks in finite time.
FLOOR_ACCUM_ROLLING_LOCK         = True

# Phase 2Q: LARGEST-AREA lock that never stalls.
#   - Auto-lock needs placement steady (center+yaw+floorY) AND area >=
#     AUTO_LOCK_MIN_AREA_M2 (so a tiny partial patch never auto-locks).
#   - While the trusted area is still GROWING the preview stays yellow and keeps
#     expanding, so we capture the whole room; we lock the largest-so-far rect
#     once growth pauses for FLOOR_ACCUM_AREA_GROWTH_FRAMES, OR after
#     LOCK_MAX_WAIT_FRAMES accepted frames as an absolute anti-stuck backstop.
#   - Manual "Lock Now" can lock the current preview at any size that passes the
#     safe minimums + coord sanity (covers a room that never quite plateaus).
# Auto-lock area floor = the smallest USABLE guardian, so a genuinely small room
# still completes the pipeline (doesn't sit in yellow forever). 1.0 m² == a 1×1 m
# square, which is exactly the hard minimum already enforced downstream
# (FLOOR_CANDIDATE_MIN_W/D=1.0 and Unity ValidateCoordinateSanity rejects w/d<1.0).
# This is a SAFE limit, not a blind lowering: nothing tinier than 1×1 m can lock.
# Big rooms are unaffected — the "keep yellow while still growing" rule below means
# crossing 1.0 m² early does NOT lock; it only locks once growth stops (you hold).
FLOOR_ACCUM_AUTO_LOCK_MIN_AREA_M2 = 1.0   # min trusted area (m²) for an AUTO lock (was 1.5)
FLOOR_ACCUM_LOCK_MAX_WAIT_FRAMES  = 25     # absolute backstop (accepted frames) — generous so big rooms grow fully

# Phase 2L: MANUAL lock (req 4). When the user presses "Lock Now" Unity sends
# {"type":"UNITY_REQUEST_LOCK"}; the server locks the CURRENT accumulated preview
# immediately, but ONLY if it still clears the safe minimums (frames/cells/area),
# the per-frame lock gate, and the Unity-space coordinate sanity check. The
# manual path skips the area-growth/stability WAIT, never the SAFETY checks.
FLOOR_ACCUM_MANUAL_LOCK_ENABLED  = True

# ---------------------------------------------------------------------------
# Image orientation
# ---------------------------------------------------------------------------
INPUT_ROTATION_DEGREES  = 270
TEST_ALL_ROTATIONS      = True

# ---------------------------------------------------------------------------
# Phase 2M: Python-side optical-axis correction (req 6).
#
# The image is rotated by input_rotation_degrees for SegFormer/depth, but
# camera_to_world is the UN-rotated physical-camera pose. Today Unity reconciles
# this per frame by rotating the received boundary about the optical axis
# (corrRot) BEFORE the pose. That fixes the FINAL cube, but Python's own world
# (accumulator, normal, y_spread) is left in the rotated/fictitious frame, which
# smears the floor cloud across frames in landscape (jittery yaw → never stable).
#
# When this flag is ON, Python applies the SAME correction Unity sends on the
# frame packet (conversion_correction_degrees) to the camera-space points BEFORE
# camera_to_world, so Python's world == the true world. Python then tags the
# outgoing boundary with optical_axis_corrected=true, and Unity SKIPS its own
# corrRot (no double-correction). Correction is 0 in portrait → portrait is
# byte-for-byte unchanged. Default OFF until verified on-device; flip the SIGN if
# [COORD_SANITY]/[H_COORD_SANITY] worsen when enabled.
APPLY_OPTICAL_AXIS_CORRECTION   = False
OPTICAL_AXIS_CORRECTION_SIGN    = 1      # +1 or -1; flip if landscape comes out mirrored

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
# Phase 10.4: 15 → 30 Hz. Post-lock the depth + segmentation models are off
# (POST_LOCK_STOP_DEPTH=True, POST_LOCK_STOP_SEGMENTATION=True), so the GPU
# is free for MediaPipe Hand. 30 Hz halves the visible vibration on the ray
# without changing detection accuracy. Pre-lock the worker shares budget with
# depth/seg so it may run slower; the bump only takes effect after lock.
HAND_TRACKING_TARGET_FPS    = 30.0
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
# Phase 3 [HAND_FPS]: DepthAnything is ~100-300 ms/frame and shares the ONE GPU
# with MediaPipe, so any depth inference starves the hand — the observed rate
# stuck at ~8 Hz and the ray read as discrete. Spec: "after lock stop
# SegFormer/Depth/plane, keep MediaPipe hand only, target >=15 FPS. If
# DepthAnything is not needed for button selection, disable it during lobby."
# Button selection does NOT need depth (the hand-ray hit-test works from the
# index-finger geometry), so post-lock we FULLY STOP depth and reconstruct the
# 3D hand from MediaPipe's own (x,y,z) landmarks (see hand_tracker.detect). This
# frees the whole GPU for MediaPipe → 20-30 Hz hand, and the glove shape comes
# straight from MediaPipe so it is stable, never the old noisy-depth "random"
# look. SegFormer + plane-fit are also OFF.
POST_LOCK_STOP_DEPTH        = True
POST_LOCK_STOP_SEGMENTATION = True
POST_LOCK_STOP_PLANE_FIT    = True
POST_LOCK_KEEP_HAND         = True
# Only used if POST_LOCK_STOP_DEPTH is set back to False (throttled-depth mode):
POST_LOCK_HAND_DEPTH_FPS    = 4.0   # base depth refresh rate for the fingertip post-lock
# Hand-rate floor for the throttled-depth mode (adaptive back-off in
# _cache_frame_for_hand). Ignored when POST_LOCK_STOP_DEPTH=True.
POST_LOCK_HAND_MIN_HZ       = 15.0
# Phase 3: if the Unity lock ACK / lobby-ready UDP never arrives, force post-lock
# this many seconds after our accumulator first becomes lock-ready, so the hand
# still speeds up.
POST_LOCK_FALLBACK_DELAY_S  = 2.0

# ---------------------------------------------------------------------------
# Legacy stability (kept for backward compatibility)
# ---------------------------------------------------------------------------
STABLE_REQUIRED             = 5
STABLE_MAX_ANGLE_DIFF       = 10.0
STABLE_MAX_DRIFT_M          = 0.15
STABLE_MIN_CONFIDENCE       = 0.75
STABLE_MAX_RMSE             = 0.04
STABLE_MIN_INLIER_RATIO     = 0.70