

import sys, cv2, numpy as np, time, json
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from modules.utils.logger       import setup_logger
from modules.core.performance   import PerformanceMonitor
from modules.core.network       import NetworkServer
from modules.cv.depth_estimator import DepthEstimator
from modules.cv.floor_detector  import FloorDetector
from modules.cv.hand_tracker    import HandTracker
from modules.cv.pose_tracker    import PoseTracker
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
    """
    Print ARCore / server compatibility check results.
    This runs at startup so issues are caught before the first connection.
    """
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

    # Unity ARCore requirements reminder
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
    def __init__(self):
        print("=" * 70)
        print("VR GUARDIAN SERVER  —  v8  (ARCore Check + EMA Smoothing)")
        print("  v = toggle debug window | c = clear area | q = quit")
        print("=" * 70)

        self.logger = setup_logger()
        self.perf   = PerformanceMonitor()
        self.viz    = Visualizer(self.logger)
        self.viz_on = False
        self.frame_skip = 2

        # ── ARCore compatibility check (v8 NEW) ───────────────────────────────
        try:
            checks = HandTracker.check_arcore_compatibility()
            print_arcore_checks(self.logger, checks)
        except Exception as e:
            self.logger.warning(f"[CHECK] Compatibility check failed: {e}")

        # ── CV modules ────────────────────────────────────────────────────────
        self.logger.info("[INIT] Loading AI modules...")

        self.hand_tracker = HandTracker(self.logger)   # v8: EMA + debounce

        try:
            self.pose_tracker = PoseTracker(self.logger)
        except Exception as e:
            self.logger.error(f"[INIT] Pose failed: {e} — disabled")
            self.pose_tracker = None

        self.enable_depth = True
        try:
            self.depth_estimator = DepthEstimator(self.logger)
            self.floor_detector  = FloorDetector(self.logger)
            self.logger.info("[INIT] Depth + Floor loaded OK")
        except Exception as e:
            self.logger.error(f"[INIT] Depth failed: {e} — disabled")
            self.enable_depth    = False
            self.depth_estimator = None
            self.floor_detector  = None

        self.network = NetworkServer(SERVER_IP, SERVER_PORT, self.logger)

        # ── Caches ────────────────────────────────────────────────────────────
        self.depth_cache = {
            'depth_map':           None,
            'raw_depth':           None,
            'frames_since_update': 0,
            'update_interval':     10,
            'last_depth_ms':       0.0,
        }
        self.floor_cache = {
            'detected': False, 'plane': [0., 0., 0., 0.], 'confidence': 0
        }
        self.floor_pending           = False
        self.floor_pending_countdown = 0

        # ── Play area ─────------------────────────────-----------───────────────────
        self.play_area_points   = []
        self.last_point_time    = 0.0
        self.point_add_interval = 0.5    # seconds between auto-adds

        self.frame_count   = 0
        self.last_log_time = time.time()

        if self.depth_estimator and hasattr(self.depth_estimator, "device"):
            device = "CUDA" if self.depth_estimator.device.type == "cuda" else "CPU"
            self.logger.info(f"[GPU] {device}")
        else:
            self.logger.warning("[DEPTH] DepthEstimator missing device → disabling depth")
            self.enable_depth = False
            self.depth_estimator = None

        self.logger.info("[INIT] All modules ready.")

    # ── Depth scheduling ──────────────────────────────────────────────────────
    def should_run_depth(self):
        if not self.enable_depth: return False
        self.depth_cache['frames_since_update'] += 1
        ms = self.depth_cache['last_depth_ms']
        self.depth_cache['update_interval'] = (
            20 if ms > 300 else 15 if ms > 150 else 10
        )
        if self.depth_cache['frames_since_update'] >= self.depth_cache['update_interval']:
            self.depth_cache['frames_since_update'] = 0
            return True
        return False

    # ── Main CV pipeline ──────────────────────────────────────────────────────
    def process_frame(self, img_bgr):
        t0 = time.time()

        h, w  = img_bgr.shape[:2]
        small   = cv2.resize(img_bgr, (320, 240), cv2.INTER_LINEAR) if w > 320 else img_bgr
        img_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        H, W    = small.shape[:2]

        # Hands (EMA + debounced in v8)
        hand_result = self.hand_tracker.detect(img_rgb)

        # Pose (lite model every frame)
        pose_result = self.pose_tracker.detect(img_rgb) if self.pose_tracker else {
            "detected": False, "landmarks": []
        }

        # Depth (adaptive interval)
        if self.should_run_depth():
            td  = time.time()
            raw = self.depth_estimator.estimate(img_rgb)
            self.depth_cache['last_depth_ms'] = (time.time() - td) * 1000
            if raw is not None:
                lo, hi = float(raw.min()), float(raw.max())
                norm   = ((raw - lo) / (hi - lo)).astype(np.float32) if hi - lo > 1e-6 else raw
                self.depth_cache['depth_map'] = norm
                self.depth_cache['raw_depth'] = raw
                self.floor_pending            = True
                self.floor_pending_countdown  = 5

        # Floor (staggered 5 frames after depth)
        if self.floor_pending and self.floor_detector:
            self.floor_pending_countdown -= 1
            if self.floor_pending_countdown <= 0:
                self.floor_pending = False
                raw = self.depth_cache.get('raw_depth')
                if raw is not None:
                    self.floor_cache = self.floor_detector.detect(raw, img_rgb)

        depth_map    = self.depth_cache['depth_map']
        floor_result = self.floor_cache

        self.update_play_area(hand_result, depth_map, W, H)

        if self.viz_on:
            self.viz.show(small, {'floor': floor_result}, self.perf.get_fps(), depth_map)

        process_ms = (time.time() - t0) * 1000
        self.perf.add_process_time(process_ms / 1000)

        result = make_serializable({
            'frame':      self.frame_count,
            'fps':        self.perf.get_fps(),
            'process_ms': process_ms,
            'floor':      floor_result,
            'hands':      hand_result,
            'pose':       pose_result,
            'play_area':  self.get_play_area(),
        })
        return result, small, depth_map

    # ── Play area update ─────────────────────────────────────────────────────
    def update_play_area(self, hand_result, depth_map, W, H):
        if not hand_result or not hand_result.get('detected'): return
        now = time.time()
        if now - self.last_point_time < self.point_add_interval: return

        MAX_AREA_PTS = 10

        for hand in hand_result.get('hands', []):
            if not hand.get('is_pointing'): continue

            # v8: skip low-confidence pointing (0.3 threshold)
            # conf = hand.get('point_conf', 1.0)
            # if conf < 0.3: continue

            tip = hand.get('index_tip')   # already EMA-smoothed in v8
            if not tip: continue

            z = 0.5
            if depth_map is not None:
                px = max(0, min(W-1, int(tip['x'] * W)))
                py = max(0, min(H-1, int(tip['y'] * H)))
                z  = float(depth_map[py, px])

            pt = {'x': float(tip['x']), 'y': float(tip['y']), 'z': z}

            if self.play_area_points:
                last  = self.play_area_points[-1]
                dist2 = (pt['x'] - last['x'])**2 + (pt['y'] - last['y'])**2
                if dist2 < 0.0025: continue   # < 5% image width — too close

                # v8: close-loop detection — if near first point AND >= 3 pts
                if len(self.play_area_points) >= 3:
                    first = self.play_area_points[0]
                    d_first = (pt['x'] - first['x'])**2 + (pt['y'] - first['y'])**2
                    if d_first < 0.015:   # within ~12% of image width = ~close loop
                        self.logger.info("[AREA] Close-loop detected — completing boundary")
                        self.play_area_points.append(pt)
                        self.last_point_time = now
                        return   # signal complete via is_complete=True (>= 3 pts)

            if len(self.play_area_points) < MAX_AREA_PTS:
                self.play_area_points.append(pt)
                self.last_point_time = now
                self.logger.info(
                    f"[AREA] Pt {len(self.play_area_points)}/{MAX_AREA_PTS} "
                    f"conf:{conf:.2f}  z:{z:.3f}  EMA({tip['x']:.3f},{tip['y']:.3f})")

    def get_play_area(self):
        n = len(self.play_area_points)
        return {'points': self.play_area_points, 'num_points': n, 'is_complete': n >= 3}

    # ── Main loop with auto-reconnect ──────────────────────────────────────────
    def run(self):
        while True:
            try:
                conn = self.network.start()
                self.logger.info("[SYSTEM] Connected — running")
                self._loop(conn)
            except KeyboardInterrupt:
                self.logger.info("[SERVER] Stopped.")
                break
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

            arr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None: continue

            self.frame_count += 1
            if self.frame_count % self.frame_skip != 0:
                
                self.network.send_result(conn, result)
                continue


            result, small, depth_map = self.process_frame(img)
            self.network.send_result(conn, result)

            now = time.time()
            if now - self.last_log_time >= 3.0:
                fps     = self.perf.get_fps()
                avg_ms  = self.perf.get_avg_process_ms()
                hands   = len((result.get('hands') or {}).get('hands', []))
                pointing = any(h.get('is_pointing')
                               for h in (result.get('hands') or {}).get('hands', []))
                pose_ok = (result.get('pose') or {}).get('detected', False)
                floor_ok= (result.get('floor') or {}).get('detected', False)
                line = (f"[STATS] #{self.frame_count:4d} | FPS:{fps:5.1f} | "
                        f"{avg_ms:5.1f}ms | Hands:{hands} "
                        f"{'POINTING' if pointing else ''} | "
                        f"Pose:{'Y' if pose_ok else 'N'} | "
                        f"Area:{len(self.play_area_points)}pt")
                if floor_ok:
                    line += f" | Floor:{result['floor']['confidence']}%"
                self.logger.info(line)
                self.last_log_time = now

            if self.viz_on or self.frame_count % 30 == 0:
                key = cv2.waitKey(1) & 0xFF
                if   key == ord('q'): raise KeyboardInterrupt
                elif key == ord('c'):
                    self.play_area_points = []
                    self.logger.info("[AREA] Cleared")
                elif key == ord('v'):
                    self.viz_on = not self.viz_on
                    self.logger.info(f"[VIZ] {'ON' if self.viz_on else 'OFF'}")
                    if not self.viz_on: cv2.destroyAllWindows()


if __name__ == "__main__":
    VRGuardianServer().run()