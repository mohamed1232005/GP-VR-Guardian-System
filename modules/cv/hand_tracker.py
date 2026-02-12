"""
Hand Tracker Module - PRODUCTION READY
MediaPipe hand tracking with pointing gesture detection
Compatible with server_COMPLETE.py
"""

import mediapipe as mp


class HandTracker:
    """
    Hand tracking using MediaPipe
    Detects hands and pointing gestures
    """
    
    def __init__(self, logger):
        """Initialize hand tracker"""
        self.logger = logger
        self.is_ready = False
        
        try:
            self.logger.info("[HANDS] Loading MediaPipe Hands...")
            
            # Import MediaPipe solutions
            self.mp_hands = mp.solutions.hands
            
            # Initialize hand detector
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=1,  # 0=lite, 1=full
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            self.is_ready = True
            self.logger.info("[HANDS] ✓ Hand tracker ready!")
            
        except Exception as e:
            self.logger.error(f"[HANDS] Failed to initialize: {e}")
            self.logger.warning("[HANDS] Hand tracking will be DISABLED")
            self.is_ready = False
    
    def is_pointing(self, landmarks):
        """
        Check if hand is making pointing gesture
        
        Returns True if:
        - Index finger is extended (tip higher than base)
        - Middle, ring, pinky are curled (tips lower than bases)
        
        MediaPipe hand landmarks:
        - 0: Wrist
        - 5: Index MCP (base)
        - 8: Index tip
        - 9: Middle MCP
        - 12: Middle tip
        - 13: Ring MCP  
        - 16: Ring tip
        - 17: Pinky MCP
        - 20: Pinky tip
        """
        try:
            # Index finger should be extended
            # (tip is higher on screen than base, so y is smaller)
            index_extended = landmarks[8].y < landmarks[5].y
            
            # Other fingers should be curled
            # (tips are lower on screen than bases, so y is larger)
            middle_curled = landmarks[12].y > landmarks[9].y
            ring_curled = landmarks[16].y > landmarks[13].y
            pinky_curled = landmarks[20].y > landmarks[17].y
            
            # All conditions must be true for pointing gesture
            return index_extended and middle_curled and ring_curled and pinky_curled
            
        except Exception as e:
            self.logger.error(f"[HANDS] Gesture check failed: {e}")
            return False
    
    def detect(self, img_rgb):
        """
        Detect hands in RGB image
        
        Args:
            img_rgb: RGB image (numpy array)
        
        Returns:
            dict with:
                - detected: bool
                - hands: list of hand data
                    - id: int (hand index)
                    - is_pointing: bool
                    - index_tip: dict {x, y, z}
                    - landmarks: list of 21 points
        """
        if not self.is_ready:
            return {
                "detected": False,
                "hands": []
            }
        
        try:
            # Process image with MediaPipe
            results = self.hands.process(img_rgb)
            
            # Build output
            output = {
                "detected": False,
                "hands": []
            }
            
            # If hands detected
            if results.multi_hand_landmarks:
                output["detected"] = True
                
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Check if making pointing gesture
                    pointing = self.is_pointing(hand_landmarks.landmark)
                    
                    # Get all 21 hand landmarks
                    landmarks_list = []
                    for lm in hand_landmarks.landmark:
                        landmarks_list.append({
                            "x": float(lm.x),
                            "y": float(lm.y),
                            "z": float(lm.z)
                        })
                    
                    # Get index finger tip position (landmark 8)
                    idx_tip = hand_landmarks.landmark[8]
                    
                    # Add hand data
                    output["hands"].append({
                        "id": idx,
                        "is_pointing": pointing,
                        "index_tip": {
                            "x": float(idx_tip.x),
                            "y": float(idx_tip.y),
                            "z": float(idx_tip.z)
                        },
                        "landmarks": landmarks_list
                    })
            
            return output
            
        except Exception as e:
            self.logger.error(f"[HANDS] Detection failed: {e}")
            return {
                "detected": False,
                "hands": []
            }
    
    def cleanup(self):
        """Cleanup resources"""
        if self.is_ready and self.hands:
            try:
                self.hands.close()
                self.logger.info("[HANDS] Cleaned up")
            except:
                pass