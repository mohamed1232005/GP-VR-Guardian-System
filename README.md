<img width="1063" height="630" alt="image" src="https://github.com/user-attachments/assets/9b91488c-1ca1-407e-b60a-43a4d2760a2b" /># Guardian System for Affordable Smartphone-Based VR

> **Team 25 — Graduation Project**  
> Faculty of Engineering | Computer Science & Artificial Intelligence  
> Academic Year 2025–2026

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [The Problem](#2-the-problem)
3. [System Architecture](#3-system-architecture)
4. [Three Implementation Pipelines](#4-three-implementation-pipelines)
   - [Pipeline 1 — ARCore + MediaPipe](#pipeline-1--arcore--mediapipe)
   - [Pipeline 2 — AI & Computer Vision](#pipeline-2--ai--computer-vision)
   - [Pipeline 3 — On-Device Guardian](#pipeline-3--on-device-guardian)
5. [TOSHFA — Use Case & Demonstration Application](#5-toshfa--use-case--demonstration-application)
6. [Technology Stack](#6-technology-stack)
7. [Data Flow](#7-data-flow)
8. [Performance Results](#8-performance-results)
9. [Repository Structure](#9-repository-structure)
10. [Getting Started](#10-getting-started)
11. [Known Issues & Bugs](#11-known-issues--bugs)
12. [Team](#12-team)
13. [Links](#13-links)

---

## 1. Project Overview

The **Guardian System for Affordable Smartphone-Based VR** is a real-time boundary-safety framework that turns any ARCore-compatible Android smartphone into a guardian-capable VR platform — no specialized hardware required.

High-end VR headsets such as the Meta Quest provide built-in guardian systems that detect physical boundaries and warn users before collisions. These systems rely on dedicated cameras, depth sensors, and proprietary processing that are unavailable on consumer smartphones. This project closes that gap.

The system uses the phone's built-in **RGB camera** and **IMU sensors** to:
- Detect and confirm the physical floor plane
- Define a safe play area and draw a virtual safety boundary in the VR scene
- Track the user's hands in 3D without a physical controller
- Detect proximity to real-world obstacles in real time
- Issue visual and audio safety warnings (Safe → Caution → Danger → Freeze) directly inside the VR experience

The framework is **modular**, **offline-capable**, and **hardware-agnostic** within the ARCore-certified Android ecosystem. TOSHFA — a lower-back VR rehabilitation module — serves as the primary demonstration application built on top of the Guardian Framework.

---

## 2. The Problem

Smartphone-based VR platforms lack built-in guardian systems capable of detecting real-world boundaries and warning users before collisions. Users remain physically vulnerable during immersive experiences while wearing a headset that completely occludes their vision.

| Impact | Description |
|--------|-------------|
| **Physical collisions** | Users walk into walls, furniture, and obstacles with no warning |
| **Risk of injury** | Bodily harm from collisions undermines user trust and safety |
| **Blocked adoption** | Movement-based VR cannot be safely deployed in clinics, schools, or rehabilitation centers without safety guarantees |

> **Core Gap:** Affordable smartphone VR lacks the spatial safety and interaction capabilities of premium VR systems.

---

## 3. System Architecture

The system is organized into five hierarchical layers:

```
┌─────────────────────────────────────────────────────────┐
│                     USER LAYER                          │
│         User wearing a Smartphone VR Headset            │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│                   HARDWARE LAYER                        │
│     Android Smartphone  │  Camera  │  IMU Sensors       │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│                  APPLICATION LAYER                      │
│                   Unity Application                     │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│             GUARDIAN FRAMEWORK CORE  ★                  │
│   ┌──────────────────┬──────────────────────────────┐   │
│   │ Safety Management│  Boundary Monitoring         │   │
│   ├──────────────────┼──────────────────────────────┤   │
│   │ Hand Interaction │  Warning System              │   │
│   └──────────────────┴──────────────────────────────┘   │
│                                                         │
│   Implementation Pipelines (supporting components):     │
│   ┌──────────────┬───────────────────┬──────────────┐   │
│   │  Pipeline 1  │    Pipeline 2     │  Pipeline 3  │   │
│   │  ARCore +    │  AI & Computer    │  On-Device   │   │
│   │  MediaPipe   │  Vision           │  Guardian    │   │
│   └──────────────┴───────────────────┴──────────────┘   │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│                USE CASE / APPLICATION                   │
│          TOSHFA Rehabilitation System                   │
│     (Demonstration application built on the framework)  │
└─────────────────────────────────────────────────────────┘
```

> **The Guardian Framework is the project.** The three pipelines are alternative implementation approaches within it. TOSHFA is the demonstration use case deployed on top of the framework.

---

## 4. Three Implementation Pipelines

### Pipeline 1 — ARCore + MediaPipe

**The primary pipeline. Powers TOSHFA and the main production build.**

| Component | Implementation |
|-----------|---------------|
| Floor Detection | ARCore Plane API + RANSAC fallback (`floor_detector.py`) |
| Hand Tracking | MediaPipe Hands — 21 3D landmarks, markerless |
| Gesture Recognition | Pointing gesture (EMA-smoothed, 3-frame debouncing state machine) |
| Controller-Free Interaction | Fingertip dot → boundary point placement in Unity |
| Communication | TCP (hand data, JSON) + UDP (camera frames, results) |

**Performance:** ~30 FPS for depth estimation; ~10–16 FPS end-to-end with the full Python backend

**Key files:**
```
server.py              — Python backend entry point (TCP/UDP server, port 9999)
hand_tracker.py        — MediaPipe hand landmark detection + gesture classifier
floor_detector.py      — ARCore plane detection + RANSAC depth-based fallback
GuardianSystem.cs      — Unity C# frontend: hand skeleton, fingertip, boundary drawing
```

**How it works:**
1. Unity streams camera frames to Python via UDP
2. `hand_tracker.py` runs MediaPipe Hands and detects pointing gestures
3. Results (gesture type, 21 landmarks, confidence) stream back via TCP as JSON
4. `GuardianSystem.cs` renders the hand skeleton, red fingertip dot, and neon safety boundary

---

### Pipeline 2 — AI & Computer Vision

**The depth-intelligence pipeline. Focuses on environmental understanding.**

| Component | Implementation |
|-----------|---------------|
| Depth Estimation | ARCore Depth API + MiDaS monocular depth (adaptive resolution scaling) |
| Floor Segmentation | RANSAC plane fitting on depth point cloud |
| Environment Understanding | YOLOv11n object detection → proximity zones (Safe / Caution / Danger) |
| Intelligent Boundary Generation | Automatic boundary box from detected floor plane |
| Sensor Fusion | Kalman filter — IMU + depth + pose |

**Performance:** ~10 FPS (heavy multi-model load; frame caching every 3rd frame reduces CPU load by ~67%)

**Key models:**
```
MiDaS           — Monocular depth estimation (depth map from single RGB frame)
YOLOv11n        — Lightweight real-time obstacle detection (on-device / edge GPU)
SegFormer       — Semantic floor segmentation
MediaPipe Pose  — 33-landmark body pose (BlazePose)
```

**Smart caching strategy:** Every 3rd frame is fully processed. Frames 1 and 2 of each triplet reuse the last valid depth map and detection result — reducing inference calls by ~67% without significant accuracy loss at human movement speeds.

**Safety state machine:**
```
SAFE ──→ CAUTION ──→ DANGER ──→ FREEZE
  ↑__________|___________|_________|
               (proximity clears)
```

---

### Pipeline 3 — On-Device Guardian

**The standalone pipeline. Runs entirely on the phone with no external server.**

| Component | Implementation |
|-----------|---------------|
| Mobile AI Inference | TensorFlow Lite (quantized models) |
| Floor Detection | Pure Python RANSAC — no ARCore dependency |
| Hand Tracking | MediaPipe on-device |
| Standalone Deployment | No Wi-Fi, no laptop, no external server required |
| Real-Time Processing | Optimized for mid-range Android (Snapdragon 600–800 series) |

**Performance:** ~12 FPS standalone on device

**Key advantages over Pipeline 1/2:**
- Works offline and in environments without Wi-Fi
- Eliminates TCP/UDP network latency (~30–80 ms saving)
- Single APK deployment — no server setup required
- Suitable for clinical and field deployment

---

## 5. TOSHFA — Use Case & Demonstration Application

**TOSHFA** is the lower-back VR rehabilitation module — the primary demonstration application built on top of the Guardian Framework. It is not a separate application; it is the `VR_LowerBackRehabMain` scene that runs inside the Guardian safe-space lifecycle.

### Scene Flow

```
Welcome Gate
    → AR Floor Detection (ARCore confirms physical floor)
    → Safe-Space + Guardian Boundary (room dimensions confirmed)
    → VR Lobby
    → [START REHAB] → VR_LowerBackRehabMain (TOSHFA)
    → Session Complete → Reset
```

### The Exercise: Step & Reach (Lower-Back Routine)

TOSHFA guides users through a five-phase lower-back rehabilitation routine:

```
RELIEVE → RELEASE → STABILIZE → STRENGTHEN → COMPLETE (Cool Down)
```

Phases cycle **within each rep** as a movement-pace guide. Session progression is **rep-driven** (10 reps default) with a timer fallback.

### Body-Pose Integration

The laptop/PC webcam runs **MediaPipe BlazePose** (`pose_tracking.py`), streaming raw 33-joint `BODY_POSE` data over TCP to Unity. Unity's `BodyPoseReceiver` derives rehabilitation metrics:

| Flag | Default | Behavior |
|------|---------|----------|
| `useBodyPoseWhenTracked` | `true` | CV reps drive progress while tracking is healthy |
| `useRealRepsForProgression` | `true` | Real reps (not the clock) advance the 5-phase session |
| `repProgressTimeoutMultiplier` | `2.5` | Safety timeout — stalled tracking still completes the session |
| `totalReps` | `10` | Reps for a full session |
| `secondsPerRep` | `5.0` | Tempo / timer-fallback pacing |

**Form feedback (status line priority order):**
1. Amber — Form warning (e.g., spine-lean violation)
2. Cyan — Hold timing ("Hold X.X s", quantized to 0.1 s)
3. Green — Good form confirmed
4. Tracking lost — rep-linked message ("N/10 reps — step back into view")

### Guardian Fail-Safe Integration

TOSHFA is fully wired into the Guardian boundary safety system:
- Subscribes to `GuardianFailSafeController` pause/resume events
- On pause: warning sound, amber "PAUSED — {reason}" status, emissives dimmed
- Self-healing: if fail-safe clears without firing resume, `Update()` auto-resumes
- Phase toasts are hard-gated off while paused — no announcements during safety events

### Session Data Export

On completion (progress = 100%):
- Exports a JSON session summary to `persistentDataPath/sessions/`
- In-VR stats panel switches from live analytics to a final summary card
- Triggered by `BodyPoseReceiver.SessionEnded` event

### Known TOSHFA-Specific Fix

When `ManualStereoCameraRig` kept ARCore alive during stereo rendering, ARCore re-localized continuously and the boundary box drifted off the physical floor. **Fix:** `FloorDetectionController` now creates a single `ARAnchor` at the confirmed floor and parents both the boundary box and safe-space root under it — pinning them to real-world coordinates regardless of ARCore re-localization.

---

## 6. Technology Stack

### Unity & Mobile Layer
| Tool | Purpose |
|------|---------|
| Unity 2022.3 LTS | VR/AR application framework |
| Android Platform (API 29+) | Target deployment OS |
| ARCore / AR Foundation | Floor detection, depth API, pose tracking |
| ARCore XR Plugin | Unity integration layer for ARCore |

### Computer Vision & AI Layer
| Library | Version | Purpose |
|---------|---------|---------|
| MediaPipe | Latest | Hand tracking (21 landmarks), body pose (33 landmarks) |
| Depth Anything V2 | — | Monocular depth estimation (alternative to MiDaS) |
| SegFormer | — | Semantic floor segmentation |
| OpenCV | 4.x | Frame preprocessing, camera calibration |
| NumPy | 1.x | Array operations, depth map processing |
| RANSAC (custom) | — | Robust floor plane fitting from depth point clouds |

### Communication Layer
| Protocol | Usage |
|----------|-------|
| UDP (port 9998) | Camera frame streaming: Unity → Python (JPEG compressed) |
| TCP (port 9999) | AI results streaming: Python → Unity (JSON) |
| Local Wi-Fi | Same network required for Pipelines 1 & 2 |

---

## 7. Data Flow

```
Phone Camera (30 FPS RGB)
       │
       ▼
TCP → Python Backend
       ├── MiDaS / Depth Anything V2 → Depth Map
       ├── MediaPipe Hands            → 21 Landmarks + Gesture
       ├── RANSAC                     → Floor Y=0 Reference
       └── YOLOv11n                   → Obstacle Proximity Labels
       │
       ▼
Safety State Machine
   SAFE / CAUTION / DANGER / FREEZE
       │
       ▼ UDP (JSON)
Unity VR Application
       ├── Boundary Renderer    → Neon safety box on floor
       ├── Hand Skeleton        → 21-bone hand overlay in AR
       ├── Fingertip Dot        → Red pointing indicator
       └── Alert System         → Visual + audio safety cues
```

**Latency targets:**
- End-to-end (sensor → Unity alert): ≤ 100 ms
- Per-frame processing: 18.9 ms – 34.0 ms (Python backend)
- UDP packet round-trip: < 5 ms on local Wi-Fi

---

## 8. Performance Results

| Pipeline | FPS | Notes |
|----------|-----|-------|
| **Pipeline 1** — ARCore + MediaPipe | **~30 FPS** | Depth estimation alone; ~10–16 FPS end-to-end with full backend |
| **Pipeline 2** — AI & Computer Vision | **~10 FPS** | Heavy multi-model load; 3-frame cache reduces CPU by 67% |
| **Pipeline 3** — On-Device Guardian | **~12 FPS** | Fully standalone; no server dependency |

**Evaluation metrics:**

| Metric | Target | Status |
|--------|--------|--------|
| Boundary detection error | ≤ 10 cm | Achieved |
| End-to-end latency | ≤ 100 ms | Achieved (avg ~52 ms) |
| Frame rate (Python backend) | ≥ 8 FPS | Achieved (avg 9.1 FPS) |
| Safety state accuracy | ≥ 90% | Achieved in test scenarios |
| Floor detection confidence | 100% | Confirmed across 578 test frames |

---

## 9. Repository Structure

```
GP-VR-Guardian-System/
│
├── server.py                    # Main Python backend — TCP/UDP server (port 9999)
├── hand_tracker.py              # MediaPipe Hands — landmark detection + gesture classifier
├── floor_detector.py            # ARCore plane + RANSAC floor fitting fallback
├── pose_tracking.py             # MediaPipe BlazePose — 33-joint body pose (TOSHFA)
├── depth_estimator.py           # MiDaS / Depth Anything V2 integration
├── safety_logic.py              # Safety state machine (Safe/Caution/Danger/Freeze)
├── requirements.txt             # Python dependencies
│
├── Unity/  (managed via Unity Cloud — see Links)
│   ├── Assets/
│   │   ├── Scripts/
│   │   │   ├── GuardianSystem.cs                    # Core: hand skeleton, boundary drawing
│   │   │   ├── FloorDetectionController.cs          # ARCore floor detection + ARAnchor fix
│   │   │   ├── GuardianFailSafeController.cs        # Boundary pause/resume events
│   │   │   ├── BodyPoseReceiver.cs                  # TOSHFA: body pose metrics
│   │   │   ├── LowerBackRehabStudioSceneController.cs # TOSHFA: main rehab controller
│   │   │   ├── VRExperienceFlowController.cs        # Master state machine
│   │   │   └── RehabStatsPanelController.cs         # TOSHFA: live stats + summary card
│   │   └── Scenes/
│   │       ├── SampleScene.unity                    # Main AR scene
│   │       └── VR_LowerBackRehabMain.unity          # TOSHFA rehabilitation scene
│   └── ...
│
└── README.md
```

**Branch structure:**

| Branch | Purpose |
|--------|---------|
| `main` | Stable Python backend (server, hand tracking, communication) |
| `Mohamed` | Depth estimation, pose estimation, modules integration |
| `Hamza` | Unity scenes, boundary drawing, floor detection |
| `Ameen` | Python-Unity pipeline, VR lobby, Android setup |
| `Yousef` | FPS optimization, ARCore Unity project, integration trials |
| `hamza_boundary` | Boundary drawing algorithm development |
| `hamza_points3D` | 3D point cloud processing experiments |

> Unity scene files are managed separately in **Unity Cloud (Plastic SCM)** to avoid binary merge conflicts.

---

## 10. Getting Started

### Prerequisites

**Hardware:**
- Android smartphone, ARCore-certified, Android 9.0+ (API 29)
- Laptop or desktop with a webcam (for TOSHFA body-pose tracking)
- Both devices on the **same local Wi-Fi network**

**Software:**
- Python 3.9+
- Unity 2022.3 LTS + Unity Hub
- Android SDK / USB debugging enabled on the phone

---

### Step 1 — Clone the Python Backend

```bash
git clone https://github.com/mohamed1232005/GP-VR-Guardian-System.git
cd GP-VR-Guardian-System
pip install -r requirements.txt
```

Start the server:
```bash
python server.py
```
The terminal will print `READY — waiting for mobile connection...` once the server is listening on port 9999. **Keep this terminal open throughout the session.**

---

### Step 2 — Pull the Unity Scene from Unity Cloud

The Unity project is managed in Unity Cloud (Plastic SCM):

1. Open **Unity Hub** and sign in with your Unity account
2. Go to **Projects → Add → Clone from Unity Cloud**
3. Paste the project link (see [Links](#13-links))
4. Wait for the scene to sync, then open **SampleScene**

---

### Step 3 — Configure the Server IP in Unity

1. In the Unity **Hierarchy**, select the `GuardianSystem` GameObject
2. In the **Inspector**, under **NETWORK**, update **Server IP** to your laptop's local IP address (e.g., `192.168.1.10`)
3. Confirm your phone and laptop are on the same Wi-Fi network

---

### Step 4 — Build and Run on Android

1. Connect your Android phone via USB and enable **USB Debugging**
2. In Unity: **File → Build Settings → Android → Switch Platform**
3. Click **Build and Run** — Unity compiles and installs the APK automatically
4. The app will launch. When the screen shows `AR Camera ✓ Net ✓`, the full pipeline is live

---

### Step 5 — Run a Session

1. Point the phone camera at the floor — ARCore will detect the floor plane
2. Confirm the safe area when prompted
3. The green neon safety boundary appears on the floor in AR
4. Walk toward any edge — the system transitions through Safe → Caution → Danger → Freeze
5. For TOSHFA: press **START REHAB** in the VR lobby to enter the rehabilitation scene

---

## 11. Known Issues & Bugs

| ID | Description | Severity | Status | Fix |
|----|-------------|----------|--------|-----|
| BUG-01 | FPS drops to ~8.5 when running depth + pose simultaneously | Medium | Resolved | Pose tracker disabled in MVP; re-enable via `server.py` line 106 |
| BUG-02 | Floor detection delayed 3–10 s on textureless / dark floors | Medium | Mitigated | 5-second timeout activates simplified Y=0 raycasting fallback |
| BUG-03 | Hand tracking loses detection at image boundary (<5% edge margin) | Low | Known | Expand camera FOV usage; add boundary proximity warning |
| BUG-04 | UDP packet loss causes safety state freeze in Unity | Medium | Mitigated | UDP retry logic + last-known-state fallback in Unity receiver |
| BUG-05 | Skipped frames sent undefined `result` variable (NameError) | Medium | Resolved | `last_result` cache added; skipped frames send last valid result |
| BUG-06 | Boundary box drifts during TOSHFA stereo mode (ARCore re-localization) | High | Resolved | `FloorDetectionController` now anchors boundary to `ARAnchor` at confirmed floor |

---

## 12. Team

**Team 25 — Guardian System for Affordable Smartphone-Based VR**

| Member | Role | Key Contributions |
|--------|------|-------------------|
| **Mohamed Ehab Yousri** | Team Lead | System architecture, monocular depth estimation (ARCore Depth API, adaptive resolution, depth caching), FPS optimization, integration trials, Unity scene building |
| **Hamza Abdelmoreed** | Integration Lead | All-modules integration (lead), boundary drawing & area detection, ground level estimation (RANSAC), Unity scenes development, Python-Unity pipeline |
| **Amin Gamal** | Co-Integration Lead | Python-Unity pipeline, VR lobby environment, Android dev environment setup, ARCore XR Plugin configuration, module development, FPS optimization |
| **Yousef Selim** | FPS & ARCore Lead | FPS optimization across all modules (depth, pose, hand tracking), ARCore Unity project, pose estimation trial (MediaPipe + rtmpose-s), integration trial |

All members contributed to: system architecture design, requirements analysis, technical documentation, and co-authoring a research paper on pose estimation for lower-back pain ("TOSHFA") published on arXiv.

---

## 13. Links

| Resource | Link |
|----------|------|
| **GitHub Repository (Python Backend)** | https://github.com/mohamed1232005/GP-VR-Guardian-System |
| **Unity Cloud Project (Unity Scenes)** | https://cloud.unity.com/home/organizations/11270618913273/projects/c0e40f1e-f1ae-49ad-9c5b-c16b028dc078 |
| **Demo Video** | https://drive.google.com/drive/folders/14uRaXD2cKZxU0FoLaBoZ7k0BDWAV8DPG |
| **Presentation** | https://canva.link/lnd040bvpzaufzr |

---

## License

This project was developed as a graduation project for academic purposes.  
© 2026 Team 25 — All rights reserved.

---

*Built with ARCore · MediaPipe · Unity · Python · TensorFlow Lite*
