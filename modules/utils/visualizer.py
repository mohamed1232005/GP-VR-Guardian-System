"""
Visualizer Module — FIXED
==========================
FIX: show() previously called processed_data['depth_map'] but the server
result dict has no 'depth_map' key (keys are: frame, fps, process_ms,
floor, hands, play_area). This caused a KeyError crash whenever visualization
was enabled.

FIX: show() now takes depth_map as a SEPARATE argument, passed by server.py
which has the normalized depth array directly from the depth cache.
"""

import cv2
import numpy as np


class Visualizer:
    def __init__(self, logger):
        self.logger         = logger
        self.window_created = False

    def _ensure_windows(self):
        if not self.window_created:
            cv2.namedWindow("Guardian — Camera + Hands", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Guardian — Depth Map",      cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Guardian — Camera + Hands", 640, 480)
            cv2.resizeWindow("Guardian — Depth Map",      640, 480)
            self.window_created = True

    def show(self, img_bgr, result, fps, depth_map=None):
        """
        Display debug windows.

        Args:
            img_bgr    : BGR numpy array (the frame from Unity, already resized)
            result     : dict with keys floor, hands, play_area, fps, process_ms
            fps        : current server FPS float
            depth_map  : normalized [0-1] float32 numpy array OR None
                         — passed directly from server.py depth cache,
                           NOT from the result dict (which has no depth_map key)
        """
        self._ensure_windows()
        floor_data = result.get('floor', {'detected': False, 'confidence': 0})
        hand_data  = result.get('hands', {'detected': False, 'hands': []})

        # ── Camera view with overlays ────────────────────────────────────────
        disp = cv2.resize(img_bgr, (640, 480))
        h, w = disp.shape[:2]

        # Floor status
        if floor_data.get('detected') and floor_data.get('confidence', 0) > 40:
            cv2.rectangle(disp, (8,8), (w-8, 88), (0,255,0), 3)
            cv2.putText(disp, f"FLOOR  {floor_data['confidence']}%",
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        else:
            cv2.rectangle(disp, (8,8), (w-8, 58), (0,0,255), 2)
            cv2.putText(disp, "Searching floor...",
                        (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # Hand skeleton overlay
        if hand_data.get('detected') and hand_data.get('hands'):
            CONNS = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                     (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
                     (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]
            for hand in hand_data['hands']:
                lms  = hand.get('landmarks', [])
                col  = (0,255,255) if hand.get('is_pointing') else (255,255,0)
                for a, b in CONNS:
                    if a < len(lms) and b < len(lms):
                        p1 = (int(lms[a]['x']*w), int(lms[a]['y']*h))
                        p2 = (int(lms[b]['x']*w), int(lms[b]['y']*h))
                        cv2.line(disp, p1, p2, col, 2)
                for lm in lms:
                    cv2.circle(disp, (int(lm['x']*w), int(lm['y']*h)), 4, col, -1)
                tip = hand.get('index_tip')
                if tip:
                    cv2.circle(disp, (int(tip['x']*w), int(tip['y']*h)), 9, (0,0,255), -1)

        # FPS
        col = (0,255,0) if fps >= 10 else (0,165,255) if fps >= 5 else (0,0,255)
        cv2.putText(disp, f"FPS: {fps:.1f}", (12, h-14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

        cv2.imshow("Guardian — Camera + Hands", disp)

        # ── Depth map view ───────────────────────────────────────────────────
        # FIX: depth_map comes as a separate argument, NOT from result dict
        if depth_map is not None:
            d8 = (np.clip(depth_map, 0, 1) * 255).astype(np.uint8)
            depth_col = cv2.applyColorMap(d8, cv2.COLORMAP_PLASMA)
            depth_col = cv2.resize(depth_col, (640, 480))
            if floor_data.get('detected') and floor_data.get('confidence', 0) > 40:
                ov = depth_col.copy()
                cv2.rectangle(ov, (0, int(480*0.6)), (640, 480), (0,255,0), -1)
                depth_col = cv2.addWeighted(depth_col, 0.75, ov, 0.25, 0)
                cv2.putText(depth_col, "FLOOR AREA", (12, 468),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        else:
            depth_col = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(depth_col, "Depth: disabled / cached",
                        (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128,128,128), 1)

        cv2.imshow("Guardian — Depth Map", depth_col)