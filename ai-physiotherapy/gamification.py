# gamification.py
class GamificationSystem:
    def __init__(self):
        self.total_score = 0
        self.session_score = 0
        self.streak = 0
        self.max_streak = 0
        self.achievements = []

    def rep_score(self, is_correct, knee_angle=None):
        # base score
        score = 10 if is_correct else -5
        # bonus for near‑perfect knee angle
        if is_correct and knee_angle and 85 <= knee_angle <= 95:
            score += 5
        # update streak
        if is_correct:
            self.streak += 1
            self.max_streak = max(self.max_streak, self.streak)
        else:
            self.streak = 0
        self.session_score += score
        self.total_score += score
        return score

    def check_achievements(self, correct_reps, session_duration):
        new = []
        if correct_reps >= 10 and "First 10" not in self.achievements:
            new.append("First 10 Reps!")
            self.achievements.append("First 10")
        if self.streak >= 5 and "5 Streak" not in self.achievements:
            new.append("5 Perfect Streak!")
            self.achievements.append("5 Streak")
        if session_duration >= 300 and "5 Min Session" not in self.achievements:
            new.append("5 Minute Warrior!")
            self.achievements.append("5 Min Session")
        return new

    def feedback_color(self, is_correct):
        # Green for correct, red for incorrect
        return (0, 255, 0) if is_correct else (0, 0, 255)
