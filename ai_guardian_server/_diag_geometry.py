# Verify the GEOMETRY-FIRST floor detection: detect the floor from the depth
# plane instead of trusting SegFormer's class label. Uses the already-cached
# Depth-Anything-V2 metric-indoor model + the project's own RANSAC plane fit.
import os, glob, sys
import numpy as np, cv2, torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from geometry.backproject import backproject_floor_points
from geometry.plane_fit import fit_floor_plane
from transformers import (AutoImageProcessor, AutoModelForDepthEstimation,
                          SegformerForSemanticSegmentation, SegformerImageProcessor)

DEBUG = os.path.join(os.path.dirname(__file__), "debug")
device = "cuda" if torch.cuda.is_available() else "cpu"

DEPTH_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
SEG_ID = "nvidia/segformer-b0-finetuned-ade-512-512"
print(f"[geo] device={device} loading depth + seg(b0)...", flush=True)
dproc = AutoImageProcessor.from_pretrained(DEPTH_ID)
dmodel = AutoModelForDepthEstimation.from_pretrained(DEPTH_ID).to(device).eval()
sproc = SegformerImageProcessor.from_pretrained(SEG_ID)
smodel = SegformerForSemanticSegmentation.from_pretrained(SEG_ID).to(device).eval()
id2label = smodel.config.id2label
floor_id = next(i for i, l in id2label.items() if l.lower() == "floor")
# classes we are confident are NOT floor (used only to subtract, never to add)
NONFLOOR = {i for i, l in id2label.items() if l.lower().split(";")[0].strip() in
            {"wall", "ceiling", "window", "windowpane", "door", "curtain",
             "painting", "sky", "person", "sofa", "armchair", "cushion"}}

@torch.no_grad()
def depth(bgr):
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    inp = dproc(images=Image.fromarray(rgb), return_tensors="pt")
    inp = {k: v.to(device) for k, v in inp.items()}
    pd = dmodel(**inp).predicted_depth
    d = torch.nn.functional.interpolate(pd.unsqueeze(1), size=(h, w),
                                        mode="bicubic", align_corners=False).squeeze()
    return d.cpu().numpy().astype(np.float32)

@torch.no_grad()
def seg(bgr):
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    inp = sproc(images=Image.fromarray(rgb), return_tensors="pt")
    inp = {k: v.to(device) for k, v in inp.items()}
    up = torch.nn.functional.interpolate(smodel(**inp).logits, size=(h, w),
                                         mode="bilinear", align_corners=False)
    return up.argmax(1).squeeze(0).cpu().numpy()

def largest_cc_fill(mask):
    m = (mask.astype(np.uint8)) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return m.astype(bool)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (lab == biggest)

targets = []
for pat in ["frame_405_1780059909", "frame_203_1780059888", "frame_109_1780059877",
            "frame_514_1780059921", "frame_299_1780059898"]:
    fs = glob.glob(os.path.join(DEBUG, pat + "_rgb.jpg"))
    if fs: targets.append(fs[0])

for path in targets:
    bgr = cv2.imread(path); h, w = bgr.shape[:2]
    # approximate phone intrinsics (portrait ~60 deg horizontal FOV)
    fx = fy = (w / 2.0) / np.tan(np.radians(60) / 2.0)
    cx, cy = w / 2.0, h / 2.0

    dmap = depth(bgr)
    smap = seg(bgr)

    # geometric ROI: lower 60% of image, exclude pixels the model is SURE are non-floor
    geo_roi = np.zeros((h, w), bool)
    geo_roi[int(h * 0.40):, :] = True
    sure_nonfloor = np.isin(smap, list(NONFLOOR))
    geo_roi &= ~sure_nonfloor

    pts, _ = backproject_floor_points(dmap, geo_roi, fx, fy, cx, cy,
                                      min_depth=0.1, max_depth=10.0,
                                      max_points=8000, frame_id="geo")
    if len(pts) < 50:
        print(f"{os.path.basename(path)}: too few pts ({len(pts)})"); continue
    plane, pstats = fit_floor_plane(pts, ransac_iterations=300,
                                    inlier_threshold=0.06, min_inliers=200,
                                    min_inlier_ratio=0.30, max_rmse=0.12, frame_id="geo")
    if plane is None:
        # relax once
        plane, pstats = fit_floor_plane(pts, ransac_iterations=400,
                                        inlier_threshold=0.10, min_inliers=100,
                                        min_inlier_ratio=0.20, max_rmse=0.20, frame_id="geo")
    n = np.array(plane["normal"]) if plane else None
    dv = plane["d"] if plane else None

    # build floor mask: every ROI pixel whose 3D point is near the plane
    if plane is not None:
        vs, us = np.where(geo_roi & np.isfinite(dmap) & (dmap > 0.1) & (dmap < 10))
        z = dmap[vs, us]
        X = (us - cx) * z / fx; Y = (vs - cy) * z / fy
        P = np.stack([X, Y, z], 1)
        dist = np.abs(P @ n + dv)
        near = dist < 0.06
        gmask = np.zeros((h, w), bool)
        gmask[vs[near], us[near]] = True
        gmask = largest_cc_fill(gmask)
        horiz = abs(n[1])  # |Y component| ~1 means horizontal plane
    else:
        gmask = np.zeros((h, w), bool); horiz = 0.0

    sem_floor = (smap == floor_id)
    roi = slice(int(h * 0.45), h)
    print(f"\n{os.path.basename(path)}")
    print(f"  SEMANTIC b0 floor ratio(full)={sem_floor.mean():.3f}")
    print(f"  GEOMETRY floor ratio(full)={gmask.mean():.3f} roi={gmask[roi].mean():.3f} "
          f"| plane_normal=({n[0]:.2f},{n[1]:.2f},{n[2]:.2f}) horiz={horiz:.2f} "
          f"inlier_ratio={pstats.get('inlier_ratio',0):.2f} rmse={pstats.get('rmse',0):.3f}"
          if plane is not None else "  GEOMETRY: plane fit FAILED")

    ov = bgr.copy()
    ov[sem_floor] = (ov[sem_floor] * [0.3, 0.3, 1.0]).clip(0, 255).astype(np.uint8)  # red=semantic
    ov[gmask]     = (ov[gmask]     * [0.3, 1.0, 0.3]).clip(0, 255).astype(np.uint8)  # green=geometry
    cv2.imwrite(os.path.join(DEBUG, "_geo_" + os.path.basename(path).replace("_rgb.jpg","_cmp.jpg")), ov)

    # depth visualization
    dv_img = dmap.copy(); dv_img = (dv_img - dv_img.min()) / (np.ptp(dv_img) + 1e-6)
    cv2.imwrite(os.path.join(DEBUG, "_geo_" + os.path.basename(path).replace("_rgb.jpg","_depth.jpg")),
                cv2.applyColorMap((dv_img*255).astype(np.uint8), cv2.COLORMAP_INFERNO))

print("\n[geo] done", flush=True)
