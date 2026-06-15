import numpy as np

class ExerciseAnalyzer:
    """Enhanced analyzer with step-by-step instruction support"""
    
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
        self.current_step = 0  # NEW: Track which step user is on
        self.step_completed = False
        self.state = "neutral"
        self.hold_timer = 0  # NEW: Timer for holding positions

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

    def analyze_bridge_stepped(self, landmarks):
        """
        Bridge Exercise with Step-by-Step Instructions
        Step 0: Lie on back, knees bent
        Step 1: Lift hips off ground
        Step 2: Hold position for 3 seconds
        """
        if not landmarks:
            return {
                "correct": False, 
                "feedback": "No pose detected",
                "current_step": self.current_step,
                "step_complete": False,
                "instruction": "Position yourself in front of camera"
            }
        
        L = self.LANDMARKS
        
        # Calculate metrics
        left_knee_angle = self.calculate_angle(landmarks, L['left_hip'], L['left_knee'], L['left_ankle'])
        right_knee_angle = self.calculate_angle(landmarks, L['right_hip'], L['right_knee'], L['right_ankle'])
        
        left_hip_y = landmarks[L['left_hip']]['y']
        right_hip_y = landmarks[L['right_hip']]['y']
        left_knee_y = landmarks[L['left_knee']]['y']
        
        avg_hip_y = (left_hip_y + right_hip_y) / 2
        hip_elevation = left_knee_y - avg_hip_y
        
        feedback = []
        correct = False
        step_complete = False
        instruction = ""
        
        # STEP 0: Starting position (lying down, knees bent)
        if self.current_step == 0:
            instruction = "Lie on your back with knees bent at 90 degrees"
            
            if left_knee_angle and right_knee_angle:
                avg_knee = (left_knee_angle + right_knee_angle) / 2
                
                # Check if user is in starting position
                if 80 <= avg_knee <= 120 and hip_elevation < 0.05:
                    feedback.append("✓ Good starting position! Ready to lift?")
                    correct = True
                    step_complete = True
                    self.hold_timer += 1
                    
                    # Move to next step after holding for 30 frames (~1 second)
                    if self.hold_timer >= 30:
                        self.current_step = 1
                        self.hold_timer = 0
                else:
                    feedback.append("Bend knees and lie flat")
                    self.hold_timer = 0
            else:
                feedback.append("Adjust camera to see your full body")
        
        # STEP 1: Lift hips
        elif self.current_step == 1:
            instruction = "Lift your hips up until body forms a straight line"
            
            if left_knee_angle and right_knee_angle:
                avg_knee = (left_knee_angle + right_knee_angle) / 2
                
                # Check if hips are lifted properly
                if hip_elevation >= 0.08 and 85 <= avg_knee <= 95:
                    feedback.append("✓ Perfect bridge position!")
                    correct = True
                    step_complete = True
                    self.hold_timer += 1
                    
                    # Move to holding phase after brief hold
                    if self.hold_timer >= 20:
                        self.current_step = 2
                        self.hold_timer = 0
                elif hip_elevation < 0.08:
                    feedback.append("Lift hips higher")
                    self.hold_timer = 0
                elif avg_knee < 85:
                    feedback.append("Bend knees slightly more")
                    self.hold_timer = 0
                elif avg_knee > 95:
                    feedback.append("Straighten legs slightly")
                    self.hold_timer = 0
            else:
                feedback.append("Position body to see full legs")
        
        # STEP 2: Hold position
        elif self.current_step == 2:
            instruction = "Hold this position for 3 seconds"
            
            if left_knee_angle and right_knee_angle:
                avg_knee = (left_knee_angle + right_knee_angle) / 2
                
                # Check if maintaining good form
                if hip_elevation >= 0.08 and 85 <= avg_knee <= 95:
                    self.hold_timer += 1
                    time_remaining = max(0, 90 - self.hold_timer) / 30.0
                    feedback.append(f"✓ Hold steady... {time_remaining:.1f}s")
                    correct = True
                    
                    # Complete the rep after 3 seconds (90 frames)
                    if self.hold_timer >= 90:
                        feedback.append("🎯 REP COMPLETE! Coin earned!")
                        step_complete = True
                        self.rep_count += 1
                        self.correct_reps += 1
                        self.current_step = 0  # Reset to start
                        self.hold_timer = 0
                else:
                    feedback.append("Maintain bridge position")
                    self.hold_timer = max(0, self.hold_timer - 2)  # Decay timer if form breaks
            else:
                feedback.append("Position body to see full legs")
                self.hold_timer = max(0, self.hold_timer - 2)
        
        return {
            "correct": correct,
            "feedback": " | ".join(feedback),
            "current_step": self.current_step,
            "step_complete": step_complete,
            "instruction": instruction,
            "hold_progress": min(1.0, self.hold_timer / 90.0),
            "knee_angle": (left_knee_angle + right_knee_angle) / 2 if left_knee_angle and right_knee_angle else None,
            "hip_elevation": hip_elevation
        }

    def analyze_cat_cow_stepped(self, landmarks):
        """
        Cat-Cow Exercise with Step-by-Step Instructions
        Step 0: Get into table-top position (hands and knees)
        Step 1: Arch back (Cow pose)
        Step 2: Round back (Cat pose)
        """
        if not landmarks:
            return {
                "correct": False,
                "feedback": "No pose detected",
                "current_step": self.current_step,
                "step_complete": False,
                "instruction": "Get on hands and knees"
            }
        
        L = self.LANDMARKS
        
        spine_angle = self.calculate_angle(
            landmarks, 
            L['left_shoulder'], 
            L['left_hip'], 
            L['left_knee']
        )
        
        feedback = []
        correct = False
        step_complete = False
        instruction = ""
        
        if spine_angle is None:
            return {
                "correct": False,
                "feedback": "Adjust camera to see full torso",
                "current_step": self.current_step,
                "step_complete": False,
                "instruction": "Position yourself in camera view"
            }
        
        # STEP 0: Table-top position
        if self.current_step == 0:
            instruction = "Get into table-top: hands under shoulders, knees under hips"
            
            if 145 <= spine_angle <= 155:
                feedback.append("✓ Good neutral position!")
                correct = True
                self.hold_timer += 1
                
                if self.hold_timer >= 30:
                    feedback.append("Ready to start! Move into COW pose (arch back)")
                    step_complete = True
                    self.current_step = 1
                    self.hold_timer = 0
            else:
                feedback.append("Adjust to neutral spine position")
                self.hold_timer = 0
        
        # STEP 1: Cow pose (arched back)
        elif self.current_step == 1:
            instruction = "COW: Drop belly, lift chest and head"
            
            if spine_angle > 160:
                feedback.append("✓ Good COW pose!")
                correct = True
                self.hold_timer += 1
                
                if self.hold_timer >= 45:
                    feedback.append("Now move to CAT pose!")
                    step_complete = True
                    self.current_step = 2
                    self.hold_timer = 0
            else:
                feedback.append("Arch back more (drop belly)")
                self.hold_timer = 0
        
        # STEP 2: Cat pose (rounded back)
        elif self.current_step == 2:
            instruction = "CAT: Round back, tuck chin to chest"
            
            if spine_angle < 140:
                feedback.append("✓ Perfect CAT pose!")
                correct = True
                self.hold_timer += 1
                
                if self.hold_timer >= 45:
                    feedback.append("🎯 REP COMPLETE! Coin earned!")
                    step_complete = True
                    self.rep_count += 1
                    self.correct_reps += 1
                    self.current_step = 1  # Go back to cow
                    self.hold_timer = 0
            else:
                feedback.append("Round back more (pull belly in)")
                self.hold_timer = 0
        
        return {
            "correct": correct,
            "feedback": " | ".join(feedback),
            "current_step": self.current_step,
            "step_complete": step_complete,
            "instruction": instruction,
            "hold_progress": min(1.0, self.hold_timer / 45.0),
            "spine_angle": spine_angle
        }

    def reset_exercise(self):
        """Reset exercise state"""
        self.current_step = 0
        self.hold_timer = 0
        self.step_completed = False