"""
VR GUARDIAN SERVER - COMPLETELY FIXED
All errors resolved, proper initialization, GPU acceleration
"""

import sys
import cv2
import numpy as np
import time
import json
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent))

# Import custom modules
from modules.utils.logger import setup_logger
from modules.core.performance import PerformanceMonitor
from modules.core.network import NetworkServer
from modules.cv.depth_estimator import DepthEstimator
from modules.cv.floor_detector import FloorDetector
from modules.cv.hand_tracker import HandTracker
from modules.utils.visualizer import Visualizer

# Configuration
SERVER_IP = "0.0.0.0"
SERVER_PORT = 9999


class VRGuardianServer:
    """Main VR Guardian server - COMPLETELY FIXED"""
    
    def __init__(self):
        """Initialize server and all modules"""
        print("=" * 80)
        print("VR GUARDIAN SERVER - GPU ACCELERATED (FIXED)")
        print("=" * 80)
        
        # Setup logger
        self.logger = setup_logger()
        
        # Performance monitor
        self.perf = PerformanceMonitor()
        
        # Initialize AI modules
        self.logger.info("\n[INIT] Loading AI modules...")
        self.depth_estimator = DepthEstimator(self.logger)
        self.floor_detector = FloorDetector(self.logger)
        self.hand_tracker = HandTracker(self.logger)
        
        # Visualizer
        self.visualizer = Visualizer(self.logger)
        
        # Network
        self.network = NetworkServer(SERVER_IP, SERVER_PORT, self.logger)
        
        # Processing cache
        self.depth_cache = {
            'depth_map': None,
            'frames_since_update': 0,
            'update_interval': 3
        }
        
        # Floor cache - FIXED: Initialize with proper structure
        self.floor_cache = {
            "detected": False,
            "plane": [0.0, 0.0, 0.0, 0.0],
            "confidence": 0
        }
        
        # Play area manager
        self.play_area_points = []
        self.last_point_time = 0
        self.point_add_interval = 0.5
        
        # Stats
        self.frame_count = 0
        self.last_log_time = time.time()
        
        self.logger.info("[INIT] All modules loaded!")
        self.logger.info(f"[GPU] Using: {'CUDA' if self.depth_estimator.device.type == 'cuda' else 'CPU'}")
    
    def should_process_depth(self):
        """Check if we should process depth this frame"""
        self.depth_cache['frames_since_update'] += 1
        
        if self.depth_cache['frames_since_update'] >= self.depth_cache['update_interval']:
            self.depth_cache['frames_since_update'] = 0
            return True
        return False
    
    def process_frame(self, img):
        """Process frame with all AI models"""
        start_time = time.time()
        
        # Resize for processing
        img_small = cv2.resize(img, (320, 240), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
        h, w = img_small.shape[:2]
        
        # STEP 1: Depth & Floor (cached)
        if self.should_process_depth():
            depth_map = self.depth_estimator.estimate(img_rgb)
            
            if depth_map is not None:
                floor_result = self.floor_detector.detect(depth_map, img_rgb)
                self.depth_cache['depth_map'] = depth_map
                self.floor_cache = floor_result
        
        # Use cached values
        depth_map = self.depth_cache['depth_map']
        floor_result = self.floor_cache
        
        # STEP 2: Hand tracking
        hand_result = self.hand_tracker.detect(img_rgb)
        
        # STEP 3: Update play area
        self.update_play_area(hand_result, depth_map, w, h)
        
        # Calculate processing time
        process_time = time.time() - start_time
        self.perf.add_process_time(process_time)
        
        # Build result
        result = {
            'frame': self.frame_count,
            'floor': floor_result,
            'hands': hand_result,
            'play_area': self.get_play_area_data(),
            'fps': self.perf.get_fps(),
            'process_ms': process_time * 1000
        }
        
        return result, img_small, depth_map
    
    def update_play_area(self, hand_result, depth_map, width, height):
        """Update play area from finger pointing"""
        if not hand_result.get('detected', False):
            return
        
        current_time = time.time()
        if current_time - self.last_point_time < self.point_add_interval:
            return
        
        for hand in hand_result.get('hands', []):
            if hand.get('is_pointing', False):
                index_tip = hand.get('index_tip')
                
                if index_tip and depth_map is not None:
                    px = int(index_tip['x'] * width)
                    py = int(index_tip['y'] * height)
                    px = max(0, min(width - 1, px))
                    py = max(0, min(height - 1, py))
                    
                    depth_value = depth_map[py, px]
                    
                    point = {
                        'x': float(index_tip['x']),
                        'y': float(index_tip['y']),
                        'z': float(depth_value)
                    }
                    
                    should_add = True
                    if len(self.play_area_points) > 0:
                        last = self.play_area_points[-1]
                        dist = ((point['x'] - last['x'])**2 + (point['y'] - last['y'])**2)**0.5
                        if dist < 0.05:
                            should_add = False
                    
                    if should_add and len(self.play_area_points) < 20:
                        self.play_area_points.append(point)
                        self.last_point_time = current_time
                        self.logger.info(f"[AREA] Point {len(self.play_area_points)} added")
    
    def get_play_area_data(self):
        """Get play area data"""
        return {
            'points': self.play_area_points,
            'num_points': len(self.play_area_points),
            'is_complete': len(self.play_area_points) >= 3
        }
    
    def run(self):
        """Main server loop"""
        try:
            # Start network server
            conn = self.network.start()
            
            self.logger.info("\n[SYSTEM] All systems GO!")
            self.logger.info("[SYSTEM] Waiting for frames...\n")
            
            # Main loop
            while True:
                self.perf.tick()
                
                # Receive frame
                frame_data = self.network.receive_frame(conn)
                if frame_data is None:
                    self.logger.warning("[NETWORK] Connection lost")
                    break
                
                # Decode
                np_arr = np.frombuffer(frame_data, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if img is None:
                    continue
                
                self.frame_count += 1
                
                # Process
                result, img_small, depth_map = self.process_frame(img)
                
                # Send to Unity
                self.network.send_result(conn, result)
                
                # Logging
                current_time = time.time()
                if current_time - self.last_log_time >= 2.0:
                    fps = self.perf.get_fps()
                    avg_process = self.perf.get_avg_process_ms()
                    
                    # Safe access to floor result
                    floor_detected = result['floor'].get('detected', False) if result['floor'] else False
                    hands_count = len(result['hands'].get('hands', [])) if result['hands'] else 0
                    
                    status = f"[STATS] Frame {self.frame_count:4d} | "
                    status += f"FPS: {fps:5.1f} | "
                    status += f"Process: {avg_process:5.1f}ms | "
                    status += f"Floor: {'YES' if floor_detected else 'NO'} | "
                    status += f"Hands: {hands_count} | "
                    status += f"Area: {len(self.play_area_points)} pts"
                    
                    self.logger.info(status)
                    self.last_log_time = current_time
                
                # Visualization
                if self.frame_count % 2 == 0:
                    processed_data = {
                        'floor': result['floor'],
                        'depth_map': depth_map
                    }
                    self.visualizer.show(img_small, processed_data, fps)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.logger.info("\n[SERVER] Quit requested")
                        break
                    elif key == ord('c'):
                        self.play_area_points = []
                        self.logger.info("[AREA] Cleared")
        
        except KeyboardInterrupt:
            self.logger.info("\n[SERVER] Stopped by user")
        
        except Exception as e:
            self.logger.error(f"\n[ERROR] {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Cleanup
            self.logger.info("\n[CLEANUP] Shutting down...")
            
            self.network.cleanup()
            cv2.destroyAllWindows()
            
            # Final stats
            fps_final = self.perf.get_fps()
            
            self.logger.info(f"\n[FINAL STATS]")
            self.logger.info(f"  Total frames: {self.frame_count}")
            self.logger.info(f"  Final FPS: {fps_final:.1f}")
            self.logger.info(f"  Play area points: {len(self.play_area_points)}")
            
            if fps_final >= 8:
                self.logger.info("\n[SUCCESS] Good FPS!")
            elif fps_final >= 5:
                self.logger.info("\n[WARNING] Acceptable FPS")
            else:
                self.logger.info("\n[WARNING] Low FPS")
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info("Server shutdown complete")
            self.logger.info("=" * 80)


if __name__ == "__main__":
    server = VRGuardianServer()
    server.run()