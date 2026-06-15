import numpy as np

class SeatedExerciseAnalyzer:
    """
    Analyzes Seated Rotation Exercise for lower back pain rehabilitation
    
    Exercise: Seated Spinal Rotation
    - Step 0: Sit straight, facing forward (neutral position)
    - Step 1: Rotate torso to the right (Style A)
    - Step 2: Return to center and rotate to left (Style B)
    
    Benefits: Improves spinal mobility, reduces stiffness
    """
    
    LANDMARKS = {
        'nose': 0,
        'left_shoulder': 11, 'right_shoulder': 12,
        'left_elbow': 13, 'right_elbow': 14,
        'left_wrist': 15, 'right_wrist': 16,
        'left_hip': 23, 'right_hip': 24,
        'left_knee': 25, 'right_knee': 26,
        'left_ankle': 27, 'right_ankle': 28
    }

    def __init__(self):
        self.rep_count = 0
        self.correct_reps = 0
        self.incorrect_reps = 0
        self.current_step = 0  # 0=neutral, 1=right rotation, 2=left rotation
        self.hold_timer = 0
        self.last_rotation = "none"  # Tracks which side was last rotated

    def calculate_angle(self, landmarks, p1, p2, p3):
        """Calculate angle at p2 vertex between p1-p2-p3"""
        if not landmarks or p1 not in landmarks or p2 not in landmarks or p3 not in landmarks:
            return None
            
        try:
            a = np.array([landmarks[p1]['x'], landmarks[p1]['y']])
            b = np.array([landmarks[p2]['x'], landmarks[p2]['y']])
            c = np.array([landmarks[p3]['x'], landmarks[p3]['y']])
            
            radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
            angle = abs(radians * 180.0 / np.pi)
            
            if angle > 180:
                angle = 360 - angle
                
            return angle
        except:
            return None

    def calculate_shoulder_rotation(self, landmarks):
        """
        Calculate torso rotation angle based on shoulder positions
        Returns: rotation angle (negative=left, positive=right, 0=center)
        """
        L = self.LANDMARKS
        
        if not landmarks:
            return None
        
        try:
            # Get shoulder and hip positions
            left_shoulder = landmarks[L['left_shoulder']]
            right_shoulder = landmarks[L['right_shoulder']]
            left_hip = landmarks[L['left_hip']]
            right_hip = landmarks[L['right_hip']]
            
            # Calculate shoulder line vector
            shoulder_vector = np.array([
                right_shoulder['x'] - left_shoulder['x'],
                right_shoulder['y'] - left_shoulder['y']
            ])
            
            # Calculate hip line vector (reference for rotation)
            hip_vector = np.array([
                right_hip['x'] - left_hip['x'],
                right_hip['y'] - left_hip['y']
            ])
            
            # Calculate angle between shoulder and hip lines
            # This tells us how much the torso has rotated
            shoulder_angle = np.arctan2(shoulder_vector[1], shoulder_vector[0])
            hip_angle = np.arctan2(hip_vector[1], hip_vector[0])
            
            rotation_angle = (shoulder_angle - hip_angle) * 180 / np.pi
            
            # Normalize to -180 to 180 range
            if rotation_angle > 180:
                rotation_angle -= 360
            elif rotation_angle < -180:
                rotation_angle += 360
            
            return rotation_angle
            
        except Exception as e:
            return None

    def check_sitting_posture(self, landmarks):
        """Check if user is in proper seated position"""
        L = self.LANDMARKS
        
        try:
            # Check hip height (should be similar to knee height when sitting)
            left_hip_y = landmarks[L['left_hip']]['y']
            right_hip_y = landmarks[L['right_hip']]['y']
            left_knee_y = landmarks[L['left_knee']]['y']
            right_knee_y = landmarks[L['right_knee']]['y']
            
            avg_hip_y = (left_hip_y + right_hip_y) / 2
            avg_knee_y = (left_knee_y + right_knee_y) / 2
            
            # When sitting, hips should be slightly above or level with knees
            height_diff = avg_hip_y - avg_knee_y
            
            # Check if sitting (height difference should be small)
            is_sitting = abs(height_diff) < 0.15
            
            # Check if hips are level
            hip_level = abs(left_hip_y - right_hip_y) < 0.05
            
            return is_sitting and hip_level
            
        except:
            return False

    def analyze_seated_rotation(self, landmarks):
        """
        Analyze seated rotation exercise with step-by-step guidance
        
        Returns:
            dict: Analysis results including correctness, feedback, step info
        """
        if not landmarks:
            return {
                "correct": False,
                "feedback": "No pose detected",
                "current_step": self.current_step,
                "step_complete": False,
                "instruction": "Position yourself in front of camera",
                "hold_progress": 0.0,
                "rotation_angle": None
            }
        
        L = self.LANDMARKS
        
        # Check if user is sitting properly
        is_sitting = self.check_sitting_posture(landmarks)
        
        if not is_sitting:
            return {
                "correct": False,
                "feedback": "Please sit on a chair",
                "current_step": self.current_step,
                "step_complete": False,
                "instruction": "Sit upright on a chair, facing camera",
                "hold_progress": 0.0,
                "rotation_angle": None
            }
        
        # Calculate rotation angle
        rotation_angle = self.calculate_shoulder_rotation(landmarks)
        
        if rotation_angle is None:
            return {
                "correct": False,
                "feedback": "Adjust camera to see shoulders and hips",
                "current_step": self.current_step,
                "step_complete": False,
                "instruction": "Move to see full upper body",
                "hold_progress": 0.0,
                "rotation_angle": None
            }
        
        feedback = []
        correct = False
        step_complete = False
        instruction = ""
        
        # ===== STEP 0: NEUTRAL POSITION (CENTER) =====
        if self.current_step == 0:
            instruction = "Sit straight, face forward, shoulders level"
            
            # Check if facing forward (rotation near 0)
            if abs(rotation_angle) < 15:
                feedback.append("✓ Good neutral position!")
                correct = True
                self.hold_timer += 1
                
                # Hold for 1 second (30 frames)
                if self.hold_timer >= 30:
                    feedback.append("Now rotate to your RIGHT →")
                    step_complete = True
                    self.current_step = 1  # Move to right rotation
                    self.hold_timer = 0
            else:
                feedback.append("Face forward, center your shoulders")
                self.hold_timer = 0
        
        # ===== STEP 1: ROTATE RIGHT (STYLE A) =====
        elif self.current_step == 1:
            instruction = "Rotate torso to the RIGHT (Style A position)"
            
            # Check for right rotation (positive angle, > 20 degrees)
            if rotation_angle > 20 and rotation_angle < 60:
                feedback.append("✓ Perfect RIGHT rotation!")
                correct = True
                self.hold_timer += 1
                
                # Hold for 2 seconds (60 frames)
                if self.hold_timer >= 60:
                    feedback.append("Great! Now return to center")
                    step_complete = True
                    self.current_step = 2  # Move to left rotation
                    self.hold_timer = 0
                    self.last_rotation = "right"
            elif rotation_angle > 60:
                feedback.append("Don't rotate too far!")
                self.hold_timer = max(0, self.hold_timer - 2)
            elif rotation_angle < 20:
                feedback.append("Rotate more to the right →")
                self.hold_timer = 0
            else:
                feedback.append("Keep rotating right")
                self.hold_timer = 0
        
        # ===== STEP 2: ROTATE LEFT (STYLE B) =====
        elif self.current_step == 2:
            instruction = "Rotate torso to the LEFT (Style B position)"
            
            # Check for left rotation (negative angle, < -20 degrees)
            if rotation_angle < -10 and rotation_angle > -90:
                feedback.append("✓ Perfect LEFT rotation!")
                correct = True
                self.hold_timer += 1
                
                # Hold for 2 seconds (60 frames)
                if self.hold_timer >= 60:
                    feedback.append("🎯 REP COMPLETE! Coin earned!")
                    step_complete = True
                    self.rep_count += 1
                    self.correct_reps += 1
                    self.current_step = 0  # Return to neutral
                    self.hold_timer = 0
                    self.last_rotation = "left"
            elif rotation_angle < -60:
                feedback.append("Don't rotate too far!")
                self.hold_timer = max(0, self.hold_timer - 2)
            elif rotation_angle > -20:
                feedback.append("Rotate more to the left ←")
                self.hold_timer = 0
            else:
                feedback.append("Keep rotating left")
                self.hold_timer = 0
        
        # Calculate hold progress (max 60 frames for rotation steps)
        max_hold = 30 if self.current_step == 0 else 60
        hold_progress = min(1.0, self.hold_timer / max_hold)
        
        return {
            "correct": correct,
            "feedback": " | ".join(feedback),
            "current_step": self.current_step,
            "step_complete": step_complete,
            "instruction": instruction,
            "hold_progress": hold_progress,
            "rotation_angle": rotation_angle,
            "is_sitting": is_sitting
        }

    def reset_exercise(self):
        """Reset exercise state"""
        self.current_step = 0
        self.hold_timer = 0
        self.last_rotation = "none"


# ========== USAGE EXAMPLE ==========
if __name__ == "__main__":
    analyzer = SeatedExerciseAnalyzer()
    
    # Example: Simulated landmarks for testing
    print("Seated Rotation Exercise Analyzer")
    print("=" * 50)
    print("\nExercise Flow:")
    print("Step 0: Sit straight (neutral) - Hold 1 sec")
    print("Step 1: Rotate RIGHT - Hold 2 sec")
    print("Step 2: Rotate LEFT - Hold 2 sec")
    print("Complete! → Back to Step 0")
    print("\n" + "=" * 50)