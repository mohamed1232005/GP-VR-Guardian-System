import cv2, os
src = r"C:/Users/Hamza/Downloads/gpsys3.mp4"
out = r"C:/Users/Hamza/Downloads/vid_frames"
os.makedirs(out, exist_ok=True)
cap = cv2.VideoCapture(src)
fps = cap.get(cv2.CAP_PROP_FPS)
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"fps={fps:.2f} frames={n} dur={n/fps if fps else 0:.1f}s "
      f"size={int(cap.get(3))}x{int(cap.get(4))}")
step = max(1, int(fps * 2))  # 1 frame every 2 seconds
i = saved = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if i % step == 0:
        t = i / fps if fps else i
        scale = 640.0 / frame.shape[1]
        small = cv2.resize(frame, (640, int(frame.shape[0] * scale)))
        cv2.imwrite(os.path.join(out, f"f_{saved:03d}_t{t:05.1f}.jpg"), small)
        saved += 1
    i += 1
cap.release()
print(f"saved {saved} frames to {out}")
