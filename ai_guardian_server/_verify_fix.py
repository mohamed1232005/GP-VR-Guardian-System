# Verify the PRODUCTION module geometry/floor_from_depth.detect_floor_from_depth
# on the saved debug RGB frames, exactly as the pipeline will call it.
import os, glob, sys
import numpy as np, cv2, torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from geometry.floor_from_depth import detect_floor_from_depth
from transformers import (AutoImageProcessor, AutoModelForDepthEstimation,
                          SegformerForSemanticSegmentation, SegformerImageProcessor)

DEBUG = os.path.join(os.path.dirname(__file__), "debug")
device = "cuda" if torch.cuda.is_available() else "cpu"
DEPTH_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
SEG_ID = "nvidia/segformer-b0-finetuned-ade-512-512"

print(f"[verify] device={device} loading depth + seg(b0, for baseline)...", flush=True)
dproc = AutoImageProcessor.from_pretrained(DEPTH_ID)
dmodel = AutoModelForDepthEstimation.from_pretrained(DEPTH_ID).to(device).eval()
sproc = SegformerImageProcessor.from_pretrained(SEG_ID)
smodel = SegformerForSemanticSegmentation.from_pretrained(SEG_ID).to(device).eval()
floor_id = next(i for i, l in smodel.config.id2label.items() if l.lower() == "floor")

@torch.no_grad()
def depth(bgr):
    h, w = bgr.shape[:2]
    inp = dproc(images=Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), return_tensors="pt")
    inp = {k: v.to(device) for k, v in inp.items()}
    pd = dmodel(**inp).predicted_depth
    return torch.nn.functional.interpolate(pd.unsqueeze(1), size=(h, w), mode="bicubic",
                                           align_corners=False).squeeze().cpu().numpy().astype(np.float32)

@torch.no_grad()
def sem_floor(bgr):
    h, w = bgr.shape[:2]
    inp = sproc(images=Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), return_tensors="pt")
    inp = {k: v.to(device) for k, v in inp.items()}
    up = torch.nn.functional.interpolate(smodel(**inp).logits, size=(h, w), mode="bilinear", align_corners=False)
    return (up.argmax(1).squeeze(0).cpu().numpy() == floor_id)

frames = ["frame_405_1780059909", "frame_203_1780059888", "frame_109_1780059877",
          "frame_514_1780059921", "frame_299_1780059898"]

print(f"\n{'frame':<26} {'b0 sem':>8} {'GEO fix':>8} {'lock?':>6}  reason / plane")
print("-" * 92)
LOCK_MIN_FLOOR_RATIO = 0.025
LOCK_MIN_CONFIDENCE = 0.55
for name in frames:
    fs = glob.glob(os.path.join(DEBUG, name + "_rgb.jpg"))
    if not fs:
        continue
    bgr = cv2.imread(fs[0]); h, w = bgr.shape[:2]
    fx = fy = (w / 2.0) / np.tan(np.radians(60) / 2.0)
    cx, cy = w / 2.0, h / 2.0

    dmap = depth(bgr)
    semf = sem_floor(bgr)
    mask, stats = detect_floor_from_depth(dmap, fx, fy, cx, cy, frame_id=name)

    # would the LOCK gate now pass on this single frame?
    lock_ok = (not stats["rejected"]
               and stats["ratio"] >= LOCK_MIN_FLOOR_RATIO
               and stats["confidence"] >= LOCK_MIN_CONFIDENCE)
    n = stats.get("plane_normal")
    nstr = f"n=({n[0]:.2f},{n[1]:.2f},{n[2]:.2f})" if n else stats["reject_reason"]
    print(f"{name:<26} {semf.mean():>7.1%} {stats['ratio']:>7.1%} {str(lock_ok):>6}  "
          f"conf={stats['confidence']:.2f} {nstr}")

    ov = bgr.copy()
    ov[semf] = (ov[semf] * [0.3, 0.3, 1.0]).clip(0, 255).astype(np.uint8)   # red = OLD semantic
    ov[mask] = (ov[mask] * [0.3, 1.0, 0.3]).clip(0, 255).astype(np.uint8)   # green = NEW geometry
    cv2.putText(ov, f"GEO {stats['ratio']*100:.0f}% conf{stats['confidence']:.2f}",
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.imwrite(os.path.join(DEBUG, "_FIX_" + name + ".jpg"), ov)

print("\n[verify] done — _FIX_*.jpg written (red=old semantic, green=new geometry)")
