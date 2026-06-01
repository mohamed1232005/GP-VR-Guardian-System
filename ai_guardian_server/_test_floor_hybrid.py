"""Smoke test for the high-confidence SegFormer fusion path in
detect_floor_from_depth (Phase 14, point 1). Builds a synthetic floor plane
+ semantic prob map and confirms the new semantic_prob branch runs and pulls
high-confidence pixels into the mask without error.
Run: venv/Scripts/python.exe _test_floor_hybrid.py
"""
import numpy as np
from geometry.floor_from_depth import detect_floor_from_depth

H, W = 240, 320
fx = fy = 250.0
cx, cy = W / 2.0, H / 2.0
CAM_HEIGHT = 1.2   # camera 1.2 m above the floor (OpenCV Y=down -> floor at +Y)

# Build a depth map of a real floor plane for the lower ROI rows (v > cy).
depth = np.zeros((H, W), dtype=np.float32)
vs = np.arange(H)
for v in range(H):
    if v > cy + 2:
        z = CAM_HEIGHT * fy / (v - cy)          # depth where the ray hits the floor
        if 0.3 < z < 8.0:
            depth[v, :] = z
        else:
            depth[v, :] = 0.0
# add mild noise
depth[depth > 0] += np.random.default_rng(0).normal(0, 0.01, depth[depth > 0].shape)

# SegFormer outputs: lower 55% labelled floor, high prob there, a confident
# (but plane-consistent) extra strip just above to exercise high-conf inclusion.
sem_mask = np.zeros((H, W), dtype=bool)
sem_mask[int(H * 0.55):, :] = True
sem_prob = np.zeros((H, W), dtype=np.float32)
sem_prob[int(H * 0.50):, :] = 0.85           # high-confidence floor band

mask, stats = detect_floor_from_depth(
    depth_map=depth, fx=fx, fy=fy, cx=cx, cy=cy,
    roi_top_frac=0.40, inlier_threshold=0.06,
    semantic_floor_mask=sem_mask,
    semantic_band_mult=2.0,
    semantic_prob=sem_prob,
    semantic_high_conf=0.60,
    semantic_high_band_mult=2.5,
    frame_id="smoke",
)

print("rejected:", stats.get("rejected"), "reason:", stats.get("reject_reason"))
print("mode:", stats.get("mode"), "pixels:", stats.get("floor_pixels"),
      "conf:", stats.get("confidence"),
      "sem_high_pixels:", stats.get("semantic_high_pixels"))

assert not stats.get("rejected"), f"floor unexpectedly rejected: {stats.get('reject_reason')}"
assert stats.get("floor_pixels", 0) > 500, "too few floor pixels"
assert stats.get("semantic_high_pixels", 0) > 0, "high-confidence semantic path not exercised"
print("PASS: high-confidence SegFormer fusion path runs and contributes pixels")
