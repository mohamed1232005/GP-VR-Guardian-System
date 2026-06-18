# 🏥 TOSHFA — AI Physiotherapy System

**Real‑time, markerless physiotherapy coaching from a single webcam.**
TOSHFA watches the patient through a laptop camera, tracks the body with MediaPipe Pose (33 landmarks), checks each exercise **step‑by‑step** against joint‑angle rules, scores the session, and streams live pose + feedback to the Unity VR‑box app over the network so the exercise guidance appears on the walls in front of the patient.

This is the **physiotherapy ("TOSHFA") half** of the *Guardian + TOSHFA* graduation project — the laptop‑side "brain" that the Unity headset app talks to.

---

## 📑 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [The Three Exercises](#-the-three-exercises)
- [How It Works](#-how-it-works)
- [Project Structure](#-project-structure)
- [Requirements](#-requirements)
- [Installation & Setup](#-installation--setup)
- [Configuration](#-configuration)
- [Running the App](#-running-the-app)
- [Keyboard Controls](#-keyboard-controls)
- [Data Sent to Unity (UDP)](#-data-sent-to-unity-udp)
- [Session Logs](#-session-logs)
- [Scoring System](#-scoring-system)
- [Troubleshooting](#-troubleshooting)

---

## 🔎 Overview

TOSHFA turns an ordinary webcam into a rehabilitation coach. Every frame it:

1. Captures the camera image and mirrors it for natural movement.
2. Runs **MediaPipe Pose** to extract 33 body landmarks.
3. Passes the landmarks to the **analyzer** for the selected exercise, which validates the patient's posture **one guided step at a time** and produces feedback.
4. Updates the **gamification** layer (score, streaks, achievements).
5. Streams the pose + feedback to **Unity** as a JSON packet over UDP.
6. Logs every correct rep and saves a full **session report** as JSON.

The patient wears the phone in a VR box and follows the on‑wall instructions; the laptop simply watches and judges. The patient never needs to look at the laptop.

---

## ✨ Features

- **Markerless full‑body tracking** — MediaPipe Pose, `model_complexity=2`, 33 landmarks, no wearables or markers.
- **Three guided rehab exercises** with step‑by‑step coaching (not just rep counting).
- **Joint‑angle analysis** — angles computed directly from landmark geometry (e.g. hip–knee–ankle, shoulder‑vs‑hip rotation).
- **Hold timers & progress bar** — positions must be *held*, not flashed.
- **Gamification** — points, streaks, and unlockable achievements to keep patients engaged.
- **Live UDP bridge to Unity** — pose + feedback streamed in real time on port `5555`.
- **Automatic session logging** — every session saved as a structured JSON report.
- **On‑screen HUD** — instruction banner, feedback, step indicator, rotation dial, reps, score, streak, FPS, and a Unity‑connection light.
- **Instant exercise switching** — change exercises live with a single keypress.

---

## 🧘 The Three Exercises

| # | Exercise | Position | Guided Steps | What's Measured |
|---|----------|----------|--------------|-----------------|
| 1 | **Bridge** | Lying on back | `0` Lie down, knees bent ~90° → `1` Lift hips → `2` Hold ~3s | Knee angle (hip–knee–ankle) + hip elevation |
| 2 | **Cat‑Cow** | Hands & knees | Stepped arch/round of the spine | Spine/back posture from torso landmarks |
| 3 | **Seated Rotation** *(default)* | Sitting on a chair | `0` Sit straight (neutral) → `1` Rotate right → `2` Return & rotate left | Torso rotation (shoulder line vs. hip line) + seated‑posture check |

> **Seated Rotation** is the default exercise on launch (`exercise="seated"`). It targets lower‑back mobility and is the one wired end‑to‑end with the Unity wall images.

Each analyzer reports: `current_step`, `instruction`, `feedback`, `correct`, `step_complete`, `hold_progress`, and the relevant angle (`rotation_angle` or `knee_angle`).

---

## ⚙️ How It Works

```
┌──────────┐   frame   ┌────────────────┐  33 landmarks  ┌──────────────────┐
│  Webcam  │ ────────▶ │ PoseEstimator  │ ─────────────▶ │ Exercise Analyzer│
└──────────┘           │ (MediaPipe)    │                │ (step + angles)  │
                       └────────────────┘                └────────┬─────────┘
                                                                  │ analysis
                                          ┌───────────────────────┼───────────────────────┐
                                          ▼                       ▼                        ▼
                                  ┌───────────────┐      ┌─────────────────┐      ┌──────────────────┐
                                  │ Gamification  │      │  UDP → Unity     │      │  DataManager     │
                                  │ score/streak  │      │  pose + feedback │      │  session_*.json  │
                                  └───────────────┘      └─────────────────┘      └──────────────────┘
```

The on‑screen OpenCV window shows the live skeleton and HUD; the same feedback travels to the Unity headset over UDP so the exercise images and progress appear on the walls.

---

## 📁 Project Structure

```
ai-physiotherapy/
├── main.py                          # Entry point + main loop, HUD, controls, UDP sender
├── pose_estimator.py                # MediaPipe Pose wrapper → 33 landmarks
├── exercise_analyzer_enhanced.py    # Bridge + Cat-Cow stepped analyzers (ExerciseAnalyzer)
├── seated_exercise_analyzer.py      # Seated Rotation stepped analyzer (SeatedExerciseAnalyzer)
├── gamification.py                  # Scoring, streaks, achievements (GamificationSystem)
├── data_manager.py                  # Session logging → sessions/*.json (DataManager)
├── requirements.txt                 # opencv-python, mediapipe, numpy
└── sessions/                        # Session JSON reports (must exist — see Setup)
```

| File | Role |
|------|------|
| `main.py` | Orchestrates the whole pipeline: camera capture, calls the pose estimator and the right analyzer, draws the HUD, sends UDP packets, handles keyboard input, prints the session summary. Default run: Seated Rotation, `user_id="user_001"`. |
| `pose_estimator.py` | Wraps MediaPipe Pose. Returns a `{index: {x, y, z, visibility}}` dict of all 33 landmarks and draws the skeleton overlay. |
| `exercise_analyzer_enhanced.py` | `ExerciseAnalyzer` — step‑by‑step logic for **Bridge** (`analyze_bridge_stepped`) and **Cat‑Cow** (`analyze_cat_cow_stepped`), with a shared `calculate_angle` helper. |
| `seated_exercise_analyzer.py` | `SeatedExerciseAnalyzer` — **Seated Rotation** logic (`analyze_seated_rotation`), including torso rotation from shoulder‑vs‑hip vectors and a sitting‑posture check. |
| `gamification.py` | Points, streak/max‑streak tracking, achievement unlocks, feedback colors. |
| `data_manager.py` | Starts/ends sessions and writes one JSON report per session to `sessions/`. |

---

## 🧩 Requirements

| Component | Recommended |
|-----------|-------------|
| **Python** | **3.11** (pinned — see note below) |
| **mediapipe** | **0.10.14** (pinned — newer versions break on Windows) |
| **opencv-python** | latest |
| **numpy** | latest (1.x preferred with this mediapipe) |
| **Webcam** | Any laptop/USB camera |
| **OS** | Windows / macOS / Linux |

> ⚠️ **Important — pin your versions.** The very newest mediapipe builds (0.10.3x) have a known Windows bug where `mediapipe` loads but `mediapipe.solutions` silently fails (`module 'mediapipe' has no attribute 'solutions'`). **mediapipe 0.10.14 on Python 3.11** avoids this and is the tested combination for this project. `requirements.txt` currently lists the packages unpinned — the setup below installs the working pinned versions.

---

## 🛠 Installation & Setup

### Option A — Conda environment (recommended)

This is the tested setup and guarantees the working versions.

```bash
# 1. Create and activate the environment
conda create -n physio python=3.11 -y
conda activate physio

# 2. Install the tested dependencies
pip install mediapipe==0.10.14 opencv-python numpy

# 3. Go to the project folder
cd ai-physiotherapy

# 4. Make sure the sessions folder exists (session logs are written here)
mkdir sessions        # Windows:  mkdir sessions   |  macOS/Linux: mkdir -p sessions
```

The environment is named **`physio`**. Every time you come back, just run `conda activate physio` before launching.

### Option B — plain pip / venv

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate     macOS/Linux:  source .venv/bin/activate
pip install "mediapipe==0.10.14" opencv-python numpy
mkdir sessions
```

> 📂 **The `sessions/` folder must exist before you run** — `DataManager` writes `sessions/session_*.json` at the end of a session and won't create the folder for you.

---

## 🔧 Configuration

All runtime settings live at the top of `PhysiotherapySystemWithSeated.__init__` in **`main.py`**:

```python
# Network setup (send to Unity)
self.unity_host = "192.168.100.5"   # ← set this to your PHONE's IP
self.unity_port = 5555              # UDP port (must match the Unity receiver)
```

**Before running with the headset:**

1. On the phone: **Settings → Wi‑Fi → tap your network → note the IP** (e.g. `192.168.1.23`).
2. Set `self.unity_host` to that phone IP.
3. Keep `self.unity_port = 5555` (this must match the Unity UDP receiver).
4. The **phone and laptop must be on the same Wi‑Fi network.**

Other defaults (bottom of `main.py`):

```python
system = PhysiotherapySystemWithSeated(
    exercise="seated",     # "bridge" | "cat_cow" | "seated"
    user_id="user_001"     # used in session log filenames
)
```

---

## ▶️ Running the App

```bash
conda activate physio      # if you used the conda env
cd ai-physiotherapy
python main.py
```

A camera window titled **"AI Physiotherapy System - WITH SEATED EXERCISE"** opens with the live skeleton and HUD. The console prints the available exercises, controls, and a 🪙 message each time a rep earns a coin.

**Physical session setup:** phone in the VR box on the patient's face, laptop placed so its webcam sees the patient's **full upper body** while they sit facing it. The patient follows the images on the phone; the laptop watches and scores.

---

## ⌨️ Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit and save the session |
| `r` | Reset all counters and the session score |
| `s` | Switch exercise (cycles Bridge → Cat‑Cow → Seated) |
| `1` | Jump to **Bridge** |
| `2` | Jump to **Cat‑Cow** |
| `3` | Jump to **Seated Rotation** |

---

## 📡 Data Sent to Unity (UDP)

Each frame, `main.py` sends one JSON packet to `unity_host:5555`:

```json
{
  "landmarks": {
    "0":  { "x": 0.51, "y": 0.34, "z": -0.12, "visibility": 0.99 },
    "11": { "x": 0.42, "y": 0.55, "z": -0.08, "visibility": 0.98 }
    // ... all 33 MediaPipe Pose landmarks
  },
  "feedback": {
    "correct": true,
    "feedback": "✓ Good rotation!",
    "instruction": "Rotate your torso to the right",
    "current_step": 1,
    "step_complete": false,
    "hold_progress": 0.65,
    "rotation_angle": 24.3,
    "knee_angle": 0,
    "exercise_type": "seated"
  },
  "timestamp": 1718630400.123
}
```

The Unity receiver uses `exercise_type` and `current_step` to pick which wall image to show, and `hold_progress` to fill the on‑wall progress bar.

---

## 💾 Session Logs

When a session ends (`q`), `DataManager` writes a report to `sessions/`:

```
sessions/session_user_001_1.json
```

```json
{
  "user_id": "user_001",
  "exercise": "seated",
  "start_time": "2026-06-19T14:02:11.001",
  "end_time":   "2026-06-19T14:09:48.552",
  "reps": [
    { "timestamp": "...", "correct": true, "angles": { ... }, "score": 120 }
  ],
  "total_correct": 12,
  "total_incorrect": 0,
  "score": 145
}
```

A summary (duration, exercise, reps, score, max streak) is also printed to the console.

---

## 🏆 Scoring System

Defined in `gamification.py`:

| Event | Points |
|-------|--------|
| Correct rep | **+10** |
| Incorrect rep | **−5** |
| Near‑perfect knee angle (85°–95°) | **+5 bonus** |

Streaks increment on each correct rep and reset on a miss (`max_streak` is kept). Achievements unlock at milestones — **First 10 Reps**, **5 Perfect Streak**, and **5‑Minute Warrior** (5‑minute session).

---

## 🩺 Troubleshooting

| Symptom | Fix |
|---------|-----|
| `module 'mediapipe' has no attribute 'solutions'` | You're on a too‑new mediapipe. Use **mediapipe 0.10.14 on Python 3.11** (the conda setup above). |
| `❌ ERROR: Cannot access camera!` | Another app is using the webcam, or the wrong camera index. Close other camera apps; if needed change `cv2.VideoCapture(0)` to `1` in `main.py`. |
| `FileNotFoundError` when the session ends | The `sessions/` folder doesn't exist — create it (`mkdir sessions`). |
| Unity isn't receiving anything | Phone and laptop must be on the **same Wi‑Fi**; `unity_host` must be the **phone's IP**; port must be **5555**; allow Python through the firewall. |
| Pose not detected / jumpy | Improve lighting and make sure the **full upper body** is in frame; the seated exercise needs shoulders **and** hips visible. |
| Low FPS | `model_complexity=2` is accurate but heavier — lower it to `1` in `pose_estimator.py` for more speed. |

---

*Part of the Guardian + TOSHFA graduation project. This repository contains the TOSHFA physiotherapy (Python / webcam) component.*
