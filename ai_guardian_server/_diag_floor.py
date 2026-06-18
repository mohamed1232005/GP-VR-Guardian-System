# Diagnostic: what does SegFormer actually do to the glossy tile floor?
# Runs the real model on saved debug RGB frames and reports class histogram,
# floor-probability distribution, and how much a threshold/sibling-class/
# morphology strategy would recover. Read-only analysis, saves nothing harmful.
import os, glob, sys
import numpy as np
import cv2
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

MODEL_ID = "nvidia/segformer-b0-finetuned-ade-512-512"
DEBUG = os.path.join(os.path.dirname(__file__), "debug")
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[diag] device={device} loading {MODEL_ID}")
proc = SegformerImageProcessor.from_pretrained(MODEL_ID)
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID).to(device).eval()
id2label = model.config.id2label

# floor-like ADE20K classes worth considering as "walkable floor"
floor_id = next(i for i, l in id2label.items() if l.lower() == "floor")
sibling_names = ["floor", "rug", "earth", "road", "sidewalk", "path", "land", "field"]
sibling_ids = [i for i, l in id2label.items()
               if any(s == l.lower().split(";")[0].strip() or s in l.lower() for s in sibling_names)]
print(f"[diag] floor_id={floor_id} ({id2label[floor_id]})")
print(f"[diag] sibling floor-like ids: {[(i, id2label[i]) for i in sibling_ids]}")

# pick the representative wide-shot + close-up frames
targets = []
for pat in ["frame_405_*_rgb.jpg", "frame_203_*_rgb.jpg", "frame_109_*_rgb.jpg",
            "frame_514_*_rgb.jpg", "frame_299_*_rgb.jpg"]:
    targets += sorted(glob.glob(os.path.join(DEBUG, pat)))

@torch.no_grad()
def infer(bgr):
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    inp = proc(images=Image.fromarray(rgb), return_tensors="pt")
    inp = {k: v.to(device) for k, v in inp.items()}
    logits = model(**inp).logits
    up = torch.nn.functional.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
    seg = up.argmax(1).squeeze(0).cpu().numpy()
    probs = torch.nn.functional.softmax(up, 1).squeeze(0).cpu().numpy()  # (C,H,W)
    return seg, probs

def largest_cc_fill(mask):
    m = (mask.astype(np.uint8)) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return m.astype(bool)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = (lab == biggest).astype(np.uint8) * 255
    # fill holes
    ff = out.copy()
    hh, ww = out.shape
    fmask = np.zeros((hh + 2, ww + 2), np.uint8)
    cv2.floodFill(ff, fmask, (0, 0), 255)
    out = out | cv2.bitwise_not(ff)
    return out.astype(bool)

for path in targets:
    bgr = cv2.imread(path)
    if bgr is None:
        continue
    h, w = bgr.shape[:2]
    seg, probs = infer(bgr)
    total = h * w
    roi = slice(int(h * 0.45), h)  # lower 55% = the ROI the pipeline keeps

    argmax_floor = (seg == floor_id)
    fprob = probs[floor_id]

    print("\n" + "=" * 70)
    print(f"{os.path.basename(path)}  ({w}x{h})")
    print(f"  argmax floor ratio (full)   = {argmax_floor.mean():.3f}")
    print(f"  argmax floor ratio (ROI)    = {argmax_floor[roi].mean():.3f}")

    # class histogram in the LOWER region (where floor must be)
    vals, counts = np.unique(seg[roi], return_counts=True)
    order = np.argsort(-counts)
    print("  predicted classes in lower-ROI (what the floor is mistaken for):")
    for idx in order[:8]:
        cid = int(vals[idx]); frac = counts[idx] / seg[roi].size
        print(f"     {frac*100:5.1f}%  id={cid:3d}  {id2label[cid]}")

    # floor probability distribution in ROI
    pr = fprob[roi]
    print(f"  floor_prob in ROI: mean={pr.mean():.3f} p50={np.percentile(pr,50):.3f} "
          f"p75={np.percentile(pr,75):.3f} p90={np.percentile(pr,90):.3f} max={pr.max():.3f}")

    # strategy A: probability threshold
    for T in (0.10, 0.20, 0.30, 0.40):
        m = (fprob > T)
        print(f"  [A] floor_prob>{T:.2f}        ratio(full)={m.mean():.3f} ratio(ROI)={m[roi].mean():.3f}")

    # strategy B: union of sibling floor-like classes (argmax)
    sib = np.isin(seg, sibling_ids)
    print(f"  [B] sibling-class union       ratio(full)={sib.mean():.3f} ratio(ROI)={sib[roi].mean():.3f}")

    # strategy C: sum of sibling probabilities > T
    sib_prob = probs[sibling_ids].sum(0)
    for T in (0.30, 0.40, 0.50):
        m = sib_prob > T
        print(f"  [C] sum(sibling_prob)>{T:.2f}   ratio(full)={m.mean():.3f} ratio(ROI)={m[roi].mean():.3f}")

    # strategy D: threshold 0.30 + morphology (close+largest-cc+fill) within ROI
    seed = (fprob > 0.30)
    seed[:int(h * 0.45), :] = False
    refined = largest_cc_fill(seed)
    print(f"  [D] prob>0.30 + morph         ratio(full)={refined.mean():.3f} ratio(ROI)={refined[roi].mean():.3f}")

    # save side-by-side for the prob>0.30+morph vs argmax
    ov = bgr.copy()
    ov[argmax_floor] = (ov[argmax_floor] * [0.3, 0.3, 1.0]).clip(0, 255).astype(np.uint8)  # red = argmax
    ov[refined] = (ov[refined] * [0.3, 1.0, 0.3]).clip(0, 255).astype(np.uint8)            # green = refined
    out = os.path.join(DEBUG, "_diag_" + os.path.basename(path).replace("_rgb.jpg", "_cmp.jpg"))
    cv2.imwrite(out, ov)
    print(f"  saved {out}")

print("\n[diag] done")
