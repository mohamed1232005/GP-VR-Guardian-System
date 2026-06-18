# Test whether a STRONGER semantic model fixes the "floor=bed" misclassification.
import os, glob
import numpy as np, cv2, torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

DEBUG = os.path.join(os.path.dirname(__file__), "debug")
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "nvidia/segformer-b4-finetuned-ade-512-512"
print(f"[big] downloading/loading {MODEL_ID} on {device} ...", flush=True)
proc = SegformerImageProcessor.from_pretrained(MODEL_ID)
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID).to(device).eval()
id2label = model.config.id2label
floor_id = next(i for i, l in id2label.items() if l.lower() == "floor")
print(f"[big] loaded. floor_id={floor_id}", flush=True)

@torch.no_grad()
def infer(bgr):
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    inp = proc(images=Image.fromarray(rgb), return_tensors="pt")
    inp = {k: v.to(device) for k, v in inp.items()}
    up = torch.nn.functional.interpolate(model(**inp).logits, size=(h, w),
                                         mode="bilinear", align_corners=False)
    return up.argmax(1).squeeze(0).cpu().numpy()

for pat in ["frame_405_1780059909_rgb.jpg", "frame_203_1780059888_rgb.jpg",
            "frame_109_1780059877_rgb.jpg", "frame_514_1780059921_rgb.jpg",
            "frame_299_1780059898_rgb.jpg"]:
    fs = glob.glob(os.path.join(DEBUG, pat))
    if not fs:
        continue
    bgr = cv2.imread(fs[0]); h, w = bgr.shape[:2]
    seg = infer(bgr)
    roi = slice(int(h*0.45), h)
    fr = (seg == floor_id)
    vals, counts = np.unique(seg[roi], return_counts=True)
    order = np.argsort(-counts)
    top = [(round(counts[i]/seg[roi].size*100,1), id2label[int(vals[i])]) for i in order[:4]]
    print(f"{pat}: b4 floor_ratio(full)={fr.mean():.3f} roi={fr[roi].mean():.3f} top={top}", flush=True)
    ov = bgr.copy(); ov[fr] = (ov[fr]*[0.3,1.0,0.3]).clip(0,255).astype(np.uint8)
    cv2.imwrite(os.path.join(DEBUG, "_b4_"+pat.replace("_rgb.jpg","_overlay.jpg")), ov)
print("[big] done", flush=True)
