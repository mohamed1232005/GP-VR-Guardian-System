# data_manager.py
import json
from datetime import datetime

class DataManager:
    def __init__(self, user_id="user_001", out_dir="sessions"):
        self.user_id = user_id
        self.out_dir = out_dir
        self.sessions = []
        self.current = None

    def start_session(self, exercise):
        self.current = {
            "user_id": self.user_id,
            "exercise": exercise,
            "start_time": datetime.now().isoformat(),
            "reps": [],
            "total_correct": 0,
            "total_incorrect": 0,
            "score": 0
        }

    def log_rep(self, rep_info):
        if self.current:
            self.current["reps"].append(rep_info)

    def end_session(self, total_correct, total_incorrect, score):
        if self.current:
            self.current.update({
                "end_time": datetime.now().isoformat(),
                "total_correct": total_correct,
                "total_incorrect": total_incorrect,
                "score": score
            })
            self.sessions.append(self.current)
            # save to file
            filename = f"{self.out_dir}/session_{self.user_id}_{len(self.sessions)}.json"
            with open(filename, 'w') as f:
                json.dump(self.current, f, indent=2)
            session = self.current
            self.current = None
            return session
