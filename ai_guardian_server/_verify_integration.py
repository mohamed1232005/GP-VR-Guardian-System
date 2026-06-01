# Integration test for HYBRID floor detection (SegFormer + depth/geometry).
# Routes real debug frames through the ACTUAL pipeline method
# AIPipelineWorker._detect_floor_hybrid with FLOOR_DETECTION_MODE="hybrid".
# Loads BOTH models (SegFormer is NOT removed) and confirms fusion + stats
# interface + debug-overlay save all work end-to-end.
import os, glob, sys
import numpy as np, cv2, torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import config
config.FLOOR_DETECTION_MODE = "hybrid"

from models.ai_pipeline_worker import AIPipelineWorker
from models.floor_segmenter import FloorSegmenter
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

DEBUG = os.path.join(os.path.dirname(__file__), "debug")
device = "cuda" if torch.cuda.is_available() else "cpu"
DEPTH_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"

print(f"[itest] device={device} loading depth + SegFormer (HYBRID)...", flush=True)
dproc = AutoImageProcessor.from_pretrained(DEPTH_ID)
dmodel = AutoModelForDepthEstimation.from_pretrained(DEPTH_ID).to(device).eval()


@torch.no_grad()
def depth(bgr):
    h, w = bgr.shape[:2]
    inp = dproc(images=Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), return_tensors="pt")
    inp = {k: v.to(device) for k, v in inp.items()}
    pd = dmodel(**inp).predicted_depth
    return torch.nn.functional.interpolate(pd.unsqueeze(1), size=(h, w), mode="bicubic",
                                           align_corners=False).squeeze().cpu().numpy().astype(np.float32)


# Build worker (no start()), then attach a REAL SegFormer so the hybrid path
# has a live semantic model — this proves SegFormer is still a dependency.
worker = AIPipelineWorker(seg_model_name=config.SEG_MODEL_NAME)
assert worker._floor_detection_mode == "hybrid", "mode not picked up"
worker._floor_segmenter = FloorSegmenter(
    model_id=config.SEG_MODEL_NAME, device=device, save_debug=False)
print(f"[itest] mode={worker._floor_detection_mode} band_mult={worker._geo_semantic_band_mult} "
      f"sem_fb(ratio>={worker._hybrid_sem_fb_ratio},conf>={worker._hybrid_sem_fb_conf})", flush=True)

frames = ["frame_405_1780059909", "frame_203_1780059888", "frame_109_1780059877",
          "frame_514_1780059921", "frame_299_1780059898"]

LOCK_R = config.LOCK_MIN_FLOOR_RATIO
LOCK_C = config.LOCK_MIN_CONFIDENCE
print(f"\n{'frame':<26} {'sem%':>6} {'final%':>7} {'agree':>6} {'conf':>6} {'lock?':>6}  mode")
print("-" * 88)
n_pass = 0
for name in frames:
    fs = glob.glob(os.path.join(DEBUG, name + "_rgb.jpg"))
    if not fs:
        print(f"{name:<26} (missing rgb)"); continue
    bgr = cv2.imread(fs[0]); h, w = bgr.shape[:2]
    fx = fy = (w / 2.0) / np.tan(np.radians(60) / 2.0)
    cx, cy = w / 2.0, h / 2.0
    dmap = depth(bgr)

    worker._geo_call_count = 0  # force a debug overlay save on this call
    mask, stats = worker._detect_floor_hybrid(bgr, dmap, fx, fy, cx, cy, name)

    lock_ok = (not stats["rejected"]
               and stats["ratio"] >= LOCK_R and stats["confidence"] >= LOCK_C)
    n_pass += int(lock_ok)
    for k in ("floor_pixels", "ratio", "confidence", "used_fallback", "rejected",
              "mode", "semantic_ratio", "semantic_confidence", "semantic_agreement"):
        assert k in stats, f"missing stats key {k}"
    print(f"{name:<26} {stats['semantic_ratio']*100:>5.1f} {stats['ratio']*100:>6.1f} "
          f"{stats['semantic_agreement']:>6.2f} {stats['confidence']:>6.2f} "
          f"{str(lock_ok):>6}  {stats['mode']}")

geo_jpgs = glob.glob(os.path.join(DEBUG, "*_geo.jpg"))
print(f"\n[itest] hybrid overlays written: {len(geo_jpgs)} (green=fused floor, red=SegFormer rejected by fusion)")
print(f"[itest] {n_pass}/{len(frames)} frames pass the lock gate via the hybrid worker method")
print("[itest] DONE — hybrid wiring verified" if n_pass == len(frames)
      else "[itest] WARNING — not all frames passed")
