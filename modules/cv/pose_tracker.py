"""
Pose Tracker Module - FIXED for MediaPipe 0.10.32+
Proper pose estimation with correct imports
"""

class PoseTracker:
    def __init__(self, logger):
        """Initialize pose tracker with proper MediaPipe handling"""
        self.logger = logger
        self.is_ready = False
        self.pose = None
        self.mp_pose = None
        
        try:
            self.logger.info("[POSE] Loading MediaPipe...")
            
            # Import MediaPipe
            import mediapipe as mp
            
            # FIXED: Correct import for MediaPipe 0.10.x+
            if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'pose'):
                self.mp_pose = mp.solutions.pose
                self.logger.info(f"[POSE] ✓ MediaPipe version: {mp.__version__}")
            else:
                raise ImportError("MediaPipe pose module not found")
            
            # Initialize pose detector with lite model for better performance
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=0,  # 0=lite, 1=full, 2=heavy
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            self.is_ready = True
            self.logger.info("[POSE] ✓ Pose tracking ready!")
        
        except ImportError as e:
            self.logger.error(f"[POSE] MediaPipe import failed: {e}")
            self.logger.warning("[POSE] ✗ Pose tracking DISABLED")
            self.is_ready = False
        
        except Exception as e:
            self.logger.error(f"[POSE] Initialization failed: {e}")
            self.is_ready = False
    
    def detect(self, img_rgb):
        """
        Detect pose in RGB image
        Returns 33 body landmarks if detected
        """
        if not self.is_ready:
            return {
                "detected": False,
                "landmarks": []
            }
        
        try:
            # Process image with MediaPipe
            results = self.pose.process(img_rgb)
            
            pose_data = {
                "detected": False,
                "landmarks": []
            }
            
            # If pose detected
            if results.pose_landmarks:
                pose_data["detected"] = True
                
                # Add all 33 pose landmarks
                for lm in results.pose_landmarks.landmark:
                    pose_data["landmarks"].append({
                        "x": float(lm.x),
                        "y": float(lm.y),
                        "z": float(lm.z),
                        "visibility": float(lm.visibility)
                    })
            
            return pose_data
        
        except Exception as e:
            self.logger.error(f"[POSE] Detection failed: {e}")
            return {
                "detected": False,
                "landmarks": []
            }
    
    def cleanup(self):
        """Cleanup resources"""
        if self.pose:
            try:
                self.pose.close()
                self.logger.info("[POSE] Cleaned up")
            except:
                pass