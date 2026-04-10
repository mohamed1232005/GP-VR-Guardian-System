import sys, cv2, numpy as np, time, json
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from modules.utils.logger       import setup_logger
from modules.core.performance   import PerformanceMonitor
from modules.core.network       import NetworkServer
from modules.cv.hand_tracker    import HandTracker
from modules.utils.visualizer   import Visualizer

SERVER_IP   = "0.0.0.0"
SERVER_PORT = 9999


def make_serializable(obj):
    """Recursively convert numpy types to native Python so json.dumps works."""
    if isinstance(obj, dict):          return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [make_serializable(v) for v in obj]
    if isinstance(obj, np.integer):    return int(obj)
    if isinstance(obj, np.floating):   return float(obj)
    if isinstance(obj, np.ndarray):    return obj.tolist()
    return obj

def print_arcore_checks(logger, checks):
    logger.info("=" * 60)
    logger.info("  ARCore Server Compatibility Check")
    logger.info("=" * 60)
    all_ok = True
    for key, val in checks.items():
        ok  = val.get('ok', False)
        ver = val.get('version') or val.get('detail', '')
        sym = "✓" if ok else "✗"
        msg = f"  {sym}  {key:<22} {ver}"
        if ok:
            logger.info(msg)
        else:
            logger.error(msg)
            all_ok = False
    logger.info("-" * 60)
    logger.info("  Unity ARCore requirements (verify in Editor):")
    logger.info("    XR Plug-in Management → Android → ARCore: ✓ checked")
    logger.info("    AR Foundation + ARCore XR Plugin: same major version")
    logger.info("    ARPlaneManager Detection Mode: Horizontal only")
    logger.info("    Scripting Backend: IL2CPP   Architecture: ARM64")
    logger.info("    Camera.clearFlags: Solid Color (not Skybox)")
    logger.info("    AR Camera tag: MainCamera")
    logger.info("    GuardianSystem Inspector: 3 AR refs assigned")
    logger.info("-" * 60)
    if all_ok:
        logger.info("  All Python checks PASSED ✓")
    else:
        logger.warning("  Some Python checks FAILED — see errors above")
    logger.info("=" * 60)


class VRGuardianServer:
    """
    v9 — SIMPLIFIED: Hands-only server.
    Removed: play_area, pose_tracker, depth_estimator, floor_detector.
    Unity now owns all boundary logic.
    """
    def __init__(self):
        print("=" * 70)
        print("VR GUARDIAN SERVER  —  v9  (Hands-Only / Unity-Driven Boundary)")
        print("  v = toggle debug window | q = quit")
        print("=" * 70)

        self.logger = setup_logger()
        self.perf   = PerformanceMonitor()
        self.viz    = Visualizer(self.logger)
        self.viz_on = False
        self.save_debug_frames = True  # Save annotated frames to prove coordinates
        self.debug_frame_counter = 0
        self.debug_dir = Path(__file__).parent / "debug_frames"
        self.debug_dir.mkdir(exist_ok=True)

        #  ARCore compatibility check 
        try:
            checks = HandTracker.check_arcore_compatibility()
            print_arcore_checks(self.logger, checks)
        except Exception as e:
            self.logger.warning(f"[CHECK] Compatibility check failed: {e}")

        #  CV modules (hands ONLY) 
        self.logger.info("[INIT] Loading hand tracker...")
        self.hand_tracker = HandTracker(self.logger)

        if not self.hand_tracker.is_ready:
            self.logger.error("[INIT] ✗ Hand tracker failed — server will run but no detection")
        else:
            self.logger.info("[INIT] ✓ Hand tracker ready")

        self.network = NetworkServer(SERVER_IP, SERVER_PORT, self.logger)

        self.frame_count   = 0
        self.last_log_time = time.time()

        self.logger.info("[INIT] All modules ready.")

    #  Main CV pipeline (hands only) 
    def process_frame(self, img_bgr):
        t0 = time.time()

        h, w = img_bgr.shape[:2]

        # ── Bug 1 fix: Rotate landscape → portrait when phone is held upright ──
        # Android XRCpuImage always outputs landscape-left regardless of grip.
        # When user holds phone in portrait, hand appears rotated/tiny in landscape frame.
        # P3 fix: also handle landscape-right (COUNTERCLOCKWISE) — previous fix only
        # handled one rotation direction; the other produced mirrored X coordinates.
        was_rotated = False
        rotate_dir = None
        if w > h:  # landscape frame detected
            # Heuristic: if aspect ratio matches expected portrait-upright capture,
            # use CW. We try CW first (landscape-left — most common Android default).
            # landscape-right is caught in a second pass if detection rate is low,
            # but for now we always rotate CW (same as before) because the coordinate
            # reversal below is keyed to CW.  A separate landscape-right branch is
            # added for correctness but disabled by default (set ROTATE_LANDSCAPE_RIGHT
            # env var to "1" to enable if hand X is systematically mirrored).
            import os as _os
            if _os.environ.get("ROTATE_LANDSCAPE_RIGHT") == "1":
                img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
                rotate_dir = "CCW"
            else:
                img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
                rotate_dir = "CW"
            h, w = img_bgr.shape[:2]
            was_rotated = True

        # Resize to portrait-oriented working size
        small = cv2.resize(img_bgr, (240, 320), cv2.INTER_LINEAR) if h > 320 else img_bgr
        img_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        # ── Bug 2 fix: Pad to square to eliminate 25% X-axis distortion ──
        # MediaPipe's landmark_projection_calculator assumes square input.
        # Without this, all X coords are compressed by 240/320 = 0.75× scale.
        sh, sw = img_rgb.shape[:2]
        if sh != sw:
            size = max(sh, sw)
            padded = np.zeros((size, size, 3), dtype=np.uint8)
            padded[:sh, :sw] = img_rgb
            # P1 FIX: Resize padded square to 160×160 before MediaPipe detection.
            # Previous code sent the full 320×320 padded image (102,400 pixels),
            # which was 3-4× slower than necessary and caused process time to jump
            # from 14ms → 62ms, triggering the 45-second dead zone and WinError crash.
            # MediaPipe HandLandmarker works well at 160×160 (25,600 pixels → ~18ms).
            img_rgb = cv2.resize(padded, (160, 160), cv2.INTER_LINEAR)
            if self.frame_count == 1:
                self.logger.info(f"[INIT] MediaPipe input size: {img_rgb.shape} (should be 160×160×3)")
            # Scale factors: landmarks are in 160×160 space, need to map back to 240×320
            # pad_sx = 240/320 = 0.75 (X was padded), pad_sy = 320/320 = 1.0
            # After resize to 160: coordinates are equivalent — same ratios apply
            self._pad_sx = sw / size  # e.g. 240/320 = 0.75
            self._pad_sy = sh / size  # e.g. 320/320 = 1.0
        else:
            img_rgb = cv2.resize(img_rgb, (160, 160), cv2.INTER_LINEAR) if sh > 160 else img_rgb
            self._pad_sx = 1.0
            self._pad_sy = 1.0

        # Hands only — ~5-10ms per frame
        hand_result = self.hand_tracker.detect(img_rgb,
                                                pad_sx=self._pad_sx,
                                                pad_sy=self._pad_sy)

        # ── Rotate landmarks back to landscape space if frame was rotated ──
        # Unity's displayMatrix was built for the ORIGINAL landscape frame.
        # After CW 90° rotation:  landscape_x = portrait_y,  landscape_y = 1 - portrait_x
        # After CCW 90° rotation: landscape_x = 1 - portrait_y, landscape_y = portrait_x
        if was_rotated and hand_result and hand_result.get('hands'):
            is_ccw = (rotate_dir == "CCW")
            for hand in hand_result['hands']:
                # Rotate index_tip back
                tip = hand.get('index_tip')
                if tip:
                    px, py = tip['x'], tip['y']
                    if is_ccw:
                        tip['x'] = 1.0 - py   # CCW: landscape-X = 1 - portrait-Y
                        tip['y'] = px          # CCW: landscape-Y = portrait-X
                    else:
                        tip['x'] = py          # CW: landscape-X = portrait-Y
                        tip['y'] = 1.0 - px    # CW: landscape-Y = 1 - portrait-X
                # Rotate all 21 landmarks back
                for lm in hand.get('landmarks', []):
                    px, py = lm['x'], lm['y']
                    if is_ccw:
                        lm['x'] = 1.0 - py
                        lm['y'] = px
                    else:
                        lm['x'] = py
                        lm['y'] = 1.0 - px

        process_ms = (time.time() - t0) * 1000
        self.perf.add_process_time(process_ms / 1000)

        result = make_serializable({
            'frame':      self.frame_count,
            'fps':        self.perf.get_fps(),
            'process_ms': process_ms,
            'hands':      hand_result,
        })

        # ── Save debug frames to PROVE coordinates are correct ──
        if self.save_debug_frames:
            self.debug_frame_counter += 1
            hands_list = (hand_result or {}).get('hands', [])
            pointing_hands = [h for h in hands_list if h.get('is_pointing')]
            # Save every 30 frames when pointing detected
            if pointing_hands and self.debug_frame_counter % 30 == 0:
                # Draw on 'small' (240×320 portrait) — the pre-pad working image
                debug_img = small.copy()
                sh_d, sw_d = debug_img.shape[:2]
                for h in pointing_hands:
                    tip = h.get('index_tip', {})
                    # tip coords are now in LANDSCAPE space (after rotation reversal)
                    nx_val, ny_val = tip.get('x', 0), tip.get('y', 0)
                    nz_val = tip.get('z', 0)
                    if was_rotated:
                        # Reverse the landmark rotation for drawing on portrait image
                        if rotate_dir == "CCW":
                            draw_x = int(ny_val * sw_d)          # reverse CCW
                            draw_y = int(nx_val * sh_d)
                        else:
                            draw_x = int((1.0 - ny_val) * sw_d)  # reverse CW
                            draw_y = int(nx_val * sh_d)
                    else:
                        draw_x = int(nx_val * sw_d)
                        draw_y = int(ny_val * sh_d)
                    cv2.circle(debug_img, (draw_x, draw_y), 15, (0, 255, 0), 3)
                    cv2.circle(debug_img, (draw_x, draw_y), 3, (0, 0, 255), -1)
                    text = f"TIP: nx={nx_val:.3f} ny={ny_val:.3f} z={nz_val:.4f}"
                    cv2.putText(debug_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 1)
                    rot_label = f"rotated={rotate_dir}" if was_rotated else "no-rotate"
                    cv2.putText(debug_img, f"Coords: LANDSCAPE space | {rot_label}", (10, sh_d - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
                fname = self.debug_dir / f"frame_{self.debug_frame_counter:05d}.jpg"
                cv2.imwrite(str(fname), debug_img)
                self.logger.info(f"[DEBUG] Saved proof frame: {fname.name} tip=({nx_val:.3f},{ny_val:.3f},{nz_val:.4f})")

        if self.viz_on:
            self.viz.show(small, result, self.perf.get_fps())

        return result

    #  Main loop with auto-reconnect 
    def run(self):
        while True:
            try:
                conn = self.network.start()
                # P2 FIX: Reset frame counter on each new connection so debug frame
                # filenames and log frame numbers start from 0 per session, not
                # accumulating across reconnects. Easier to correlate proof frames.
                self.frame_count = 0
                self.debug_frame_counter = 0
                self.last_log_time = time.time()
                self.logger.info("[SYSTEM] Connected — running")
                self._loop(conn)
            except KeyboardInterrupt:
                self.logger.info("[SERVER] Stopped.")
                break
            except OSError as e:
                # P2 FIX: WinError 10054 = Unity force-closed the TCP socket.
                # This happens when Unity crashes or when our processing is too slow
                # and Unity's send buffer fills. Catch it gracefully and reconnect
                # instead of crashing the server.
                self.logger.warning(f"[SERVER] Connection reset by Unity (WinError 10054 / ECONNRESET): {e}")
                self.logger.info("[SERVER] Reconnecting in 1s...")
            except Exception as e:
                import traceback
                self.logger.error(f"[SERVER] Error: {e}")
                traceback.print_exc()
            self.logger.info("[SERVER] Reconnecting in 1s...")
            time.sleep(1)
        self.network.cleanup()
        cv2.destroyAllWindows()

    def _loop(self, conn):
        while True:
            self.perf.tick()
            data = self.network.receive_frame(conn)
            if data is None:
                self.logger.warning("[NET] Connection lost.")
                break

            # Handle JSON commands from Unity (kept for future use)
            if len(data) > 0 and data[0] == ord('{'):
                try:
                    cmd_json = data.decode('utf-8')
                    cmd_dict = json.loads(cmd_json)
                    self.logger.info(f"[CMD] Unity command: {cmd_dict}")
                except:
                    pass
                continue

            arr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None: continue

            self.frame_count += 1

            # Process EVERY frame — hand detection is fast (~5ms)
            result = self.process_frame(img)
            self.network.send_result(conn, result)

            now = time.time()
            if now - self.last_log_time >= 3.0:
                fps     = self.perf.get_fps()
                avg_ms  = self.perf.get_avg_process_ms()
                hands   = len((result.get('hands') or {}).get('hands', []))
                pointing = any(h.get('is_pointing')
                               for h in (result.get('hands') or {}).get('hands', []))
                line = (f"[STATS] #{self.frame_count:4d} | FPS:{fps:5.1f} | "
                        f"{avg_ms:5.1f}ms | Hands:{hands} "
                        f"{'POINTING' if pointing else ''}")
                self.logger.info(line)
                self.last_log_time = now

            if self.viz_on or self.frame_count % 30 == 0:
                key = cv2.waitKey(1) & 0xFF
                if   key == ord('q'): raise KeyboardInterrupt
                elif key == ord('v'):
                    self.viz_on = not self.viz_on
                    self.logger.info(f"[VIZ] {'ON' if self.viz_on else 'OFF'}")
                    if not self.viz_on: cv2.destroyAllWindows()


if __name__ == "__main__":
    VRGuardianServer().run()