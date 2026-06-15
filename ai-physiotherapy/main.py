import cv2
import numpy as np
import json
import socket
import time
from datetime import datetime
from collections import deque

# Import modules
from pose_estimator import PoseEstimator
from exercise_analyzer_enhanced import ExerciseAnalyzer
from seated_exercise_analyzer import SeatedExerciseAnalyzer
from gamification import GamificationSystem
from data_manager import DataManager

class PhysiotherapySystemWithSeated:
    """
    Enhanced physiotherapy system with 3 exercises:
    1. Bridge (lying down)
    2. Cat-Cow (on hands and knees)
    3. Seated Rotation (NEW - on chair)
    """
    
    def __init__(self, exercise="seated", user_id="user_001"):
        self.pose_estimator = PoseEstimator()
        
        # Multiple analyzers for different exercises
        self.bridge_analyzer = ExerciseAnalyzer()
        self.seated_analyzer = SeatedExerciseAnalyzer()
        
        self.gamification = GamificationSystem()
        self.data_manager = DataManager(user_id)
        self.exercise = exercise
        
        # Network setup (send to Unity)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.unity_host = "192.168.100.5"
        self.unity_port = 5555
        
        # Performance tracking
        self.fps_history = deque(maxlen=30)
        
        print("\n" + "="*70)
        print("🏥 AI PHYSIOTHERAPY SYSTEM - WITH SEATED EXERCISE")
        print("="*70)
        print(f"Exercise: {exercise.upper()}")
        print(f"User ID: {user_id}")
        print(f"Unity Connection: {self.unity_host}:{self.unity_port}")
        print("\nAvailable Exercises:")
        print("  1. 'bridge' - Bridge exercise (lying down)")
        print("  2. 'cat_cow' - Cat-Cow stretch (hands and knees)")
        print("  3. 'seated' - Seated Rotation (NEW! on chair)")
        print("\nControls:")
        print("  'q' = Quit")
        print("  'r' = Reset counters")
        print("  's' = Switch exercise (cycles through all 3)")
        print("  '1' = Bridge")
        print("  '2' = Cat-Cow")
        print("  '3' = Seated Rotation")
        print("="*70 + "\n")

    def get_current_analyzer(self):
        """Get the appropriate analyzer for current exercise"""
        if self.exercise == "seated":
            return self.seated_analyzer
        else:
            return self.bridge_analyzer

    def run(self):
        """Main application loop"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ ERROR: Cannot access camera!")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Start session
        self.data_manager.start_session(self.exercise)
        print(f"✅ Session started - Begin {self.exercise.upper()} exercise!\n")
        
        last_coin_time = 0
        
        while True:
            start_time = cv2.getTickCount()
            
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame")
                break
            
            # Mirror for natural interaction
            frame = cv2.flip(frame, 1)
            
            # Step 1: Detect pose
            results, landmarks = self.pose_estimator.detect_pose(frame)
            
            # Step 2: Analyze based on current exercise
            if self.exercise == "bridge":
                analysis = self.bridge_analyzer.analyze_bridge_stepped(landmarks)
            elif self.exercise == "cat_cow":
                analysis = self.bridge_analyzer.analyze_cat_cow_stepped(landmarks)
            elif self.exercise == "seated":
                analysis = self.seated_analyzer.analyze_seated_rotation(landmarks)
            else:
                analysis = {
                    "correct": False,
                    "feedback": "Unknown exercise",
                    "current_step": 0,
                    "instruction": "Select valid exercise"
                }
            
            # Step 3: Update gamification
            if analysis["correct"]:
                score = self.gamification.rep_score(
                    True,
                    analysis.get("knee_angle") or analysis.get("rotation_angle")
                )
            
            # Step 4: Send to Unity
            self.send_to_unity(landmarks, analysis)
            
            # Step 5: Draw visualization
            frame = self.pose_estimator.draw_landmarks(frame, results)
            frame = self._draw_enhanced_ui(frame, analysis)
            
            # Step 6: Log data
            if analysis.get("step_complete") and analysis["current_step"] == 0:
                self.data_manager.log_rep({
                    "timestamp": datetime.now().isoformat(),
                    "correct": True,
                    "angles": analysis,
                    "score": self.gamification.session_score
                })
                
                current_time = time.time()
                if current_time - last_coin_time > 1.0:
                    print(f"🪙 COIN EARNED! Score: {self.gamification.session_score}")
                    last_coin_time = current_time
            
            # Calculate FPS
            end_time = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (end_time - start_time)
            self.fps_history.append(fps)
            
            # Display
            cv2.imshow('AI Physiotherapy System - WITH SEATED EXERCISE', frame)
            
            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n⏹️  Stopping session...")
                break
            elif key == ord('r'):
                print("\n🔄 Resetting counters...")
                self.reset_all_analyzers()
                self.gamification.session_score = 0
            elif key == ord('s'):
                # Cycle through exercises
                exercises = ["bridge", "cat_cow", "seated"]
                current_idx = exercises.index(self.exercise)
                self.exercise = exercises[(current_idx + 1) % len(exercises)]
                print(f"\n🔀 Switched to: {self.exercise.upper()}")
                self.reset_all_analyzers()
            elif key == ord('1'):
                self.exercise = "bridge"
                print(f"\n🔀 Switched to: BRIDGE")
                self.reset_all_analyzers()
            elif key == ord('2'):
                self.exercise = "cat_cow"
                print(f"\n🔀 Switched to: CAT-COW")
                self.reset_all_analyzers()
            elif key == ord('3'):
                self.exercise = "seated"
                print(f"\n🔀 Switched to: SEATED ROTATION")
                self.reset_all_analyzers()
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.sock.close()
        
        # End session
        analyzer = self.get_current_analyzer()
        session = self.data_manager.end_session(
            analyzer.correct_reps,
            analyzer.incorrect_reps,
            self.gamification.session_score
        )
        
        self._print_summary(session)

    def reset_all_analyzers(self):
        """Reset all exercise analyzers"""
        self.bridge_analyzer.rep_count = 0
        self.bridge_analyzer.correct_reps = 0
        self.bridge_analyzer.current_step = 0
        self.bridge_analyzer.hold_timer = 0
        
        self.seated_analyzer.rep_count = 0
        self.seated_analyzer.correct_reps = 0
        self.seated_analyzer.current_step = 0
        self.seated_analyzer.hold_timer = 0

    def send_to_unity(self, landmarks, analysis):
        """Send pose and analysis data to Unity via UDP"""
        try:
            data = {
                "landmarks": landmarks if landmarks else {},
                "feedback": {
                    "correct": analysis.get("correct", False),
                    "feedback": analysis.get("feedback", ""),
                    "instruction": analysis.get("instruction", ""),
                    "current_step": analysis.get("current_step", 0),
                    "step_complete": analysis.get("step_complete", False),
                    "hold_progress": analysis.get("hold_progress", 0.0),
                    "rotation_angle": analysis.get("rotation_angle", 0),
                    "knee_angle": analysis.get("knee_angle", 0),
                    "exercise_type": self.exercise  # NEW: Tell Unity which exercise
                },
                "timestamp": time.time()
            }
            
            message = json.dumps(data).encode('utf-8')
            self.sock.sendto(message, (self.unity_host, self.unity_port))
            
        except Exception as e:
            pass

    def _draw_enhanced_ui(self, frame, analysis):
        """Draw enhanced UI with step information"""
        h, w, _ = frame.shape
        
        color = (0, 255, 0) if analysis.get("correct") else (0, 0, 255)
        
        # Top banner - Instruction
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 120), color, -1)
        frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)
        
        instruction = analysis.get("instruction", "Follow the steps")
        cv2.putText(frame, instruction, (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
        
        # Feedback banner
        feedback = analysis.get("feedback", "")
        cv2.putText(frame, feedback, (20, 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        
        # Step indicator
        step_text = f"Step {analysis.get('current_step', 0) + 1}/3"
        cv2.putText(frame, step_text, (w - 200, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2, cv2.LINE_AA)
        
        # Progress bar
        hold_progress = analysis.get("hold_progress", 0.0)
        if hold_progress > 0:
            bar_width = 400
            bar_height = 30
            bar_x = (w - bar_width) // 2
            bar_y = 200
            
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                         (50, 50, 50), -1)
            
            fill_width = int(bar_width * hold_progress)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height),
                         (0, 255, 0), -1)
            
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                         (255, 255, 255), 2)
            
            progress_pct = int(hold_progress * 100)
            cv2.putText(frame, f"{progress_pct}%", (bar_x + bar_width + 20, bar_y + 22),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Rotation angle display (for seated exercise)
        if self.exercise == "seated" and analysis.get("rotation_angle") is not None:
            rotation = analysis.get("rotation_angle", 0)
            rotation_text = f"Rotation: {rotation:.1f}°"
            cv2.putText(frame, rotation_text, (w // 2 - 100, 260),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            # Visual rotation indicator
            center_x = w // 2
            center_y = 320
            radius = 40
            cv2.circle(frame, (center_x, center_y), radius, (100, 100, 100), 2)
            
            # Draw rotation arrow
            angle_rad = rotation * np.pi / 180
            end_x = int(center_x + radius * np.cos(angle_rad))
            end_y = int(center_y - radius * np.sin(angle_rad))
            cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y),
                          (0, 255, 255), 3, tipLength=0.3)
        
        # Stats panel
        stats_x = w - 280
        stats_y = 30
        
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (stats_x - 20, stats_y - 10),
                     (w - 20, stats_y + 250), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay2, 0.6, frame, 0.4, 0)
        
        cv2.putText(frame, f"Exercise:", (stats_x, stats_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, self.exercise.upper(), (stats_x, stats_y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        analyzer = self.get_current_analyzer()
        cv2.putText(frame, f"Reps: {analyzer.rep_count}",
                   (stats_x, stats_y + 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Score: {self.gamification.session_score}",
                   (stats_x, stats_y + 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        cv2.putText(frame, f"Streak: {self.gamification.streak}",
                   (stats_x, stats_y + 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 150, 0), 2)
        
        # FPS
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (20, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Unity connection
        cv2.circle(frame, (w - 30, h - 30), 10, (0, 255, 0), -1)
        cv2.putText(frame, "Unity", (w - 80, h - 22),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame

    def _print_summary(self, session):
        """Print session summary"""
        print("\n" + "="*70)
        print("📊 SESSION COMPLETE")
        print("="*70)
        
        if session:
            duration = (datetime.fromisoformat(session['end_time']) -
                       datetime.fromisoformat(session['start_time']))
            
            minutes = duration.seconds // 60
            seconds = duration.seconds % 60
            
            print(f"\n⏱️  Duration: {minutes}m {seconds}s")
            print(f"💪 Exercise: {session['exercise'].upper()}")
            print(f"\n📈 Performance:")
            print(f"   Total Reps: {session['total_correct']}")
            print(f"   ✅ Correct: {session['total_correct']}")
            
            print(f"\n🏆 Score: {session['score']}")
            print(f"🔥 Max Streak: {self.gamification.max_streak}")
        
        print("\n" + "="*70)
        print("💾 Session data saved to: sessions/")
        print("🎉 Great work! Keep up the progress! 💪\n")


if __name__ == "__main__":
    # Run system with seated exercise by default
    system = PhysiotherapySystemWithSeated(
        exercise="seated",
        user_id="user_001"
    )
    system.run()