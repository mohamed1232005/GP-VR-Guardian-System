# Guardian System for Affordable Smartphone-Based Virtual Reality

> A real-time, controller-free **spatial-safety framework** that turns any ARCore-compatible Android phone into a guardian-capable VR platform — **no extra hardware**, using only the phone's built-in **RGB camera** and **IMU**.

**Team 25 · Graduation Project 2025–2026**
**Zewail City of Science and Technology — School of Computational Sciences and Artificial Intelligence (CSAI)**

### Project Snapshot

| | |
|---|---|
| **Platform** | Android 9.0+ (API 29+), ARCore-certified |
| **Engine** | Unity 2022.3 LTS |
| **AR stack** | ARCore / AR Foundation / ARCore XR Plugin |
| **AI / CV** | MediaPipe · Depth Anything V2 · SegFormer-B0 · RANSAC · Unity Sentis |
| **Backend** | Python 3.9+ (Pipelines 1 & 2); on-device C# (Pipeline 3) |
| **Use case** | TOSHFA — lower-back VR rehabilitation |
| **Status** | Final submission — three working pipelines |

---

## Table of Contents

1. [Project Description](#1-project-description)
2. [Team Members](#2-team-members)
3. [Supervisor & Sponsor](#3-supervisor--sponsor)
4. [Problem Statement](#4-problem-statement)
5. [Objectives & Scope](#5-objectives--scope)
6. [Features](#6-features)
7. [System Architecture](#7-system-architecture)
8. [The Three Implementation Pipelines](#8-the-three-implementation-pipelines)
9. [TOSHFA — Demonstration Use Case](#9-toshfa--demonstration-use-case)
10. [Technologies Used](#10-technologies-used)
11. [Repository Structure & Branching Strategy](#11-repository-structure--branching-strategy)
12. [Environment Requirements](#12-environment-requirements)
13. [Setup Instructions](#13-setup-instructions)
14. [Deployment Instructions](#14-deployment-instructions)
15. [Usage Guide](#15-usage-guide)
16. [API Documentation](#16-api-documentation)
17. [Data & Session Schema](#17-data--session-schema)
18. [Performance & Results](#18-performance--results)
19. [Testing & Validation](#19-testing--validation)
20. [Screenshots & Demo](#20-screenshots--demo)
21. [Troubleshooting](#21-troubleshooting)
22. [Reproducing & Verifying the Build](#22-reproducing--verifying-the-build)
23. [Contributions](#23-contributions)
24. [Project Links](#24-project-links)
25. [License](#25-license)

---

## 1. Project Description

High-end VR headsets such as the Meta Quest ship with built-in **guardian systems** that detect physical boundaries and warn users before collisions. These rely on dedicated depth cameras and proprietary processing that affordable smartphone VR simply does not have. **This project closes that gap.**

The Guardian System adds guardian-style spatial safety to **affordable smartphone-based VR** using only the phone's existing sensors. While wearing a headset that fully blocks their view, the user is continuously protected: the system detects and confirms the physical floor, draws a virtual safety boundary inside the VR scene, tracks the user's hands in 3D without a controller, and issues escalating **Safe → Caution → Danger → Freeze** warnings in real time.

The framework is **modular**, **hardware-agnostic** within the ARCore ecosystem, and is delivered through **three interchangeable pipelines** that implement the same guardian core with different perception strategies and compute locations. It is demonstrated through **TOSHFA**, a lower-back VR rehabilitation application built directly on top of the guardian safe-space lifecycle.

---

## 2. Team Members

| Name | ID | Program |
|------|------|---------|
| Ameen Gamal | 202202219 | CSAI |
| Hamza Abdelmoreed | 202201508 | CSAI |
| Mohamed Ehab Yousry | 202201236 | CSAI |
| Yousef Selim Shawky | 202201255 | CSAI |

> *Bachelor of Science in Computational Sciences and Artificial Intelligence (CSAI).*

---

## 3. Supervisor & Sponsor

| Role | Name |
|------|------|
| **Supervisor** | Dr. Mayada Mansour Ali Hadhoud |
| **Sponsor / Industry Partner** | [VRapeutic](https://myvrapeutic.com) — VR-based therapeutic & rehabilitation solutions |

---

## 4. Problem Statement

Smartphone-based VR platforms lack built-in guardian systems capable of detecting real-world boundaries and warning users before collisions. While wearing a headset that fully occludes their vision, users remain physically vulnerable during immersive, movement-based experiences.

| Impact | Description |
|--------|-------------|
| **Physical collisions** | Users walk into walls, furniture, and obstacles with no warning. |
| **Risk of injury** | Collisions cause bodily harm and erode user trust in smartphone VR. |
| **Blocked adoption** | Movement-based VR cannot be safely deployed in clinics, schools, or rehabilitation centres without spatial-safety guarantees. |
| **Interaction gap** | Without controllers, affordable VR offers no natural way to interact with the scene. |

**Core gap:** affordable smartphone VR lacks the spatial-safety and natural-interaction capabilities of premium VR systems — and it must be solved **without adding hardware**.

---

## 5. Objectives & Scope

**Objectives**

- Deliver guardian-style spatial safety on affordable smartphones using **only the built-in camera and IMU**.
- Detect and confirm the physical floor markerlessly and anchor a stable virtual boundary to it.
- Provide **controller-free** hand interaction for placing and confirming the safe area.
- Drive **multimodal** (visual + audio + haptic) warnings through a clear safety state machine.
- Keep the design **modular** so the guardian core can be reused across VR applications.
- Validate the framework with a real application — **TOSHFA** lower-back rehabilitation.

**Scope**

- Implementation, integration, and evaluation of the three pipelines and the TOSHFA use case.
- **Technical-feasibility scope:** the project demonstrates a working, evaluated system. It is **not** a clinical trial; TOSHFA was not tested on patients and carries no medical claims.
- Target platform is **Android / ARCore**; iOS / ARKit is identified as future work.

---

## 6. Features

- **Markerless floor detection** — ARCore plane detection and depth-based RANSAC plane fitting confirm the physical floor as the `Y = 0` reference.
- **Anchored virtual boundary** — a locked neon boundary is pinned to the real floor with an `ARAnchor`, eliminating ARCore re-localization drift in stereo VR.
- **Controller-free hand interaction** — MediaPipe Hands tracks **21 3D landmarks**; a fingertip ray places, edits, and confirms boundary points.
- **Escalating safety warnings** — a **Safe → Caution → Danger → Freeze** state machine drives synchronized visual, audio, and haptic alerts.
- **Three interchangeable pipelines** — networked, AI-perception, and fully on-device implementations of the same guardian core.
- **Offline on-device mode** — Pipeline 3 runs entirely on the phone with no laptop, Wi-Fi, or external server.
- **Body-pose rep tracking** — MediaPipe Pose (33 landmarks) drives exercise counting and form feedback in TOSHFA.
- **Guardian-integrated application layer** — TOSHFA pauses and resumes automatically based on guardian safety events.

---

## 7. System Architecture

The system is organized into hierarchical layers. The **Guardian Framework Core is the project**; the three pipelines are alternative implementations of that core, and **TOSHFA** is the demonstration application deployed on top of it.

```
┌──────────────────────────────────────────────────────────────┐
│  USER LAYER          User wearing a smartphone VR headset      │
├──────────────────────────────────────────────────────────────┤
│  HARDWARE LAYER      Android phone │ RGB camera │ IMU sensors  │
├──────────────────────────────────────────────────────────────┤
│  APPLICATION LAYER   Unity 2022.3 LTS VR/AR application        │
├──────────────────────────────────────────────────────────────┤
│  GUARDIAN FRAMEWORK CORE  *                                    │
│   ┌───────────────────┬────────────────────────────────────┐  │
│   │ Safety Management │ Boundary Monitoring                │  │
│   │ Hand Interaction  │ Warning System (Safe→…→Freeze)     │  │
│   └───────────────────┴────────────────────────────────────┘  │
│   Implementation pipelines:                                    │
│   ┌──────────────┬───────────────────┬─────────────────────┐  │
│   │  Pipeline 1  │    Pipeline 2     │     Pipeline 3      │  │
│   │  ARCore +    │  AI & Computer    │  On-Device Guardian │  │
│   │  MediaPipe   │  Vision (Sentis)  │  (C# / Sentis)      │  │
│   └──────────────┴───────────────────┴─────────────────────┘  │
├──────────────────────────────────────────────────────────────┤
│  USE CASE            TOSHFA — Lower-Back Rehabilitation         │
└──────────────────────────────────────────────────────────────┘
```

**Runtime data flow (networked pipelines)**

```
Phone camera ──JPEG/UDP──▶ Python backend ──(MediaPipe / Depth / RANSAC)──▶
   perception result ──JSON/TCP──▶ Unity ──▶ boundary + warnings + interaction
```

For Pipeline 3 the entire loop runs **inside Unity on the device** — frames never leave the phone.

---

## 8. The Three Implementation Pipelines

All three pipelines implement the **same guardian core** (floor → boundary → warnings) but differ in *how* they perceive the environment and *where* computation runs. Each pipeline has a **Unity scene** (managed in Unity Cloud / Plastic SCM) and a corresponding **Python branch** on GitHub.

### Pipeline 1 — ARCore + MediaPipe  *(primary / production build)*

The networked pipeline that powers TOSHFA. ARCore (Unity) performs floor-plane detection; a Python backend runs **MediaPipe Hands** (21 landmarks) and gesture recognition; Unity renders the hand skeleton, fingertip ray, and safety boundary. Camera frames stream out over UDP and JSON results return over TCP.

**Flow:** ARCore detects plane → user confirms area with fingertip → boundary anchored → MediaPipe hand interaction + warning monitoring → TOSHFA.

| | |
|---|---|
| **Unity scene** | [`MVP-Guardian-Unity/ARCore_pipeline_with_simple_toshfa`](https://cloud.unity.com/home/organizations/11270618913273/projects/3af6106f-603f-411a-9087-d805c64f2b72/plastic-scm/organizations/11270618913273/repositories/MVP-Guardian-Unity%2FARCore_pipeline_with_simple_toshfa) |
| **Python branch** | [`ARcore_simple_toshfa`](https://github.com/mohamed1232005/GP-VR-Guardian-System/tree/ARcore_simple_toshfa) |
| **Floor detection** | ARCore Plane API (+ RANSAC fallback) |
| **Interaction** | MediaPipe Hands — 21 landmarks, fingertip pointing |
| **Compute** | Phone client + Python host over local Wi-Fi |
| **Render rate** | ~60 FPS (lightest on-device load) |

### Pipeline 2 — AI & Computer Vision

The depth-intelligence pipeline, focused on environmental understanding. **Depth Anything V2** provides monocular depth, **SegFormer-B0** performs semantic floor segmentation, and a custom **RANSAC** routine fits the floor plane from the resulting point cloud. The AI models are deployed as ONNX and executed through **Unity Inference Engine (Sentis)**; the Python side handles MediaPipe and geometry.

**Flow:** frame → depth + floor segmentation → point cloud → RANSAC floor plane → boundary + warning monitoring.

| | |
|---|---|
| **Unity scene** | [`MVP-Guardian-Unity/AI_Guardian_Pipeline_2_hamza`](https://cloud.unity.com/home/organizations/11270618913273/projects/3af6106f-603f-411a-9087-d805c64f2b72/plastic-scm/organizations/11270618913273/repositories/MVP-Guardian-Unity%2FAI_Guardian_Pipeline_2_hamza/branch/%2Fmain/tree/) |
| **Python branch** | [`AI-Guardian-pipeline-2--final-version`](https://github.com/mohamed1232005/GP-VR-Guardian-System/tree/AI-Guardian-pipeline-2--final-version) |
| **Depth** | Depth Anything V2 (monocular) |
| **Segmentation** | SegFormer-B0 (floor) |
| **Floor fitting** | RANSAC plane fitting on depth point cloud |
| **Render rate** | 27.5 – 30 FPS |

### Pipeline 3 — On-Device Guardian  *(C# / Sentis, backend-free)*

A fully **on-device** implementation written in native **C#** using **Unity Inference Engine (Sentis)** — no external Python backend. Depth Anything V2 runs locally through Sentis and a C# RANSAC routine defines the safe play area, keeping inference local and low-latency for offline and clinical deployment. This scene also hosts the **TOSHFA** scene.

**Flow:** on-device depth (Sentis) → C# RANSAC floor → boundary + warning monitoring → TOSHFA — all on the phone.

| | |
|---|---|
| **Unity scene (+ TOSHFA)** | [`MVP-Guardian-Unity/Warning_System_Yousef-hAMZA`](https://cloud.unity.com/home/organizations/11270618913273/projects/3af6106f-603f-411a-9087-d805c64f2b72/plastic-scm/organizations/11270618913273/repositories/MVP-Guardian-Unity%2FWarning_System_Yousef-hAMZA/branch/%2Fmain%2FFix%20CODE/tree/) |
| **Python branch (prototype)** | [`Yousef-final-python-On-Device-Pipeline-3`](https://github.com/mohamed1232005/GP-VR-Guardian-System/tree/Yousef-final-python-On-Device-Pipeline-3) |
| **Inference** | Depth Anything V2 via Unity Sentis (on-device) |
| **Floor detection** | RANSAC implemented entirely in C# |
| **Compute** | Standalone APK — no Wi-Fi, no laptop, no server |
| **Render rate** | ~35 FPS |

---

## 9. TOSHFA — Demonstration Use Case

**TOSHFA** is the lower-back VR rehabilitation module — the primary application built on top of the Guardian Framework. It runs **inside the guardian safe-space lifecycle**: the boundary must be detected and locked before the rehabilitation scene starts.

**Scene flow**

```
Welcome Gate → AR Floor Detection → Safe-Space + Guardian Boundary
            → VR Lobby → [START REHAB] → TOSHFA Rehab Scene
            → Session Complete → Reset
```

**Rehabilitation routine.** A guided lower-back session uses **MediaPipe Pose (33 body landmarks)** to drive rep tracking. Progression is **rep-driven** (10 reps by default) with a timer fallback, and on-screen **form feedback** (form warning, hold timing, good-form confirmation) guides the user through each movement.

**Guardian fail-safe integration.** TOSHFA subscribes to the boundary fail-safe controller: on a safety event the session pauses (warning sound, amber "PAUSED" status, dimmed visuals) and auto-resumes when the area is clear. On completion, a **JSON session summary** is exported and the in-VR stats panel switches to a final summary card.

---

## 10. Technologies Used

**Unity & Mobile**

| Tool | Purpose |
|------|---------|
| Unity 2022.3 LTS | VR/AR application framework |
| Android (API 29+) | Target deployment platform |
| ARCore / AR Foundation | Floor detection, depth, pose tracking |
| ARCore XR Plugin · XR Plugin Management | Unity ↔ ARCore integration |
| Unity Inference Engine (Sentis) | On-device ONNX inference (Pipelines 2 & 3) |
| TextMeshPro · Input System | In-VR UI and input |

**AI & Computer Vision**

| Library / Model | Purpose |
|-----------------|---------|
| MediaPipe Hands | Hand tracking — 21 3D landmarks |
| MediaPipe Pose (BlazePose) | Body pose — 33 landmarks (TOSHFA) |
| Depth Anything V2 | Monocular depth estimation |
| SegFormer-B0 | Semantic floor segmentation |
| RANSAC (custom) | Robust floor-plane fitting |
| OpenCV · NumPy · SciPy | Frame processing & geometry |

**Communication**

| Protocol | Usage |
|----------|-------|
| UDP | Camera-frame streaming: Unity → Python (JPEG) |
| TCP | Result streaming: Python → Unity (JSON) |
| Local Wi-Fi | Shared network for Pipelines 1 & 2 |

---

## 11. Repository Structure & Branching Strategy

Source code is split across **GitHub** (Python backends) and **Unity Cloud / Plastic SCM** (Unity scenes and binary assets, kept out of Git to avoid large-binary merge conflicts). Each pipeline lives on a dedicated branch so its history is isolated and reproducible.

**Python backend (representative layout)**

```
GP-VR-Guardian-System/
├── server.py            # TCP/UDP server entry point
├── hand_tracker.py      # MediaPipe Hands — landmarks + gesture classifier
├── floor_detector.py    # ARCore plane + RANSAC floor fitting
├── pose_tracking.py     # MediaPipe Pose — 33-joint body pose (TOSHFA)
├── safety_logic.py      # Safety state machine (Safe/Caution/Danger/Freeze)
├── transport.py         # UDP/TCP transport layer
├── config.py            # Ports, IPs, thresholds
├── gesture_recognizer.task
└── requirements.txt
```

**Branch ↔ pipeline ↔ scene mapping**

| Pipeline | GitHub branch (Python) | Unity Cloud scene |
|----------|------------------------|-------------------|
| Pipeline 1 + initial TOSHFA | `ARcore_simple_toshfa` | `ARCore_pipeline_with_simple_toshfa` |
| Pipeline 2 | `AI-Guardian-pipeline-2--final-version` | `AI_Guardian_Pipeline_2_hamza` |
| Pipeline 3 + TOSHFA | `Yousef-final-python-On-Device-Pipeline-3` | `Warning_System_Yousef-hAMZA` *(branch `/main/Fix CODE`)* |

> Unity scenes are version-controlled in **Unity Cloud (Plastic SCM)** under the **`MVP-Guardian-Unity`** repository. All members contribute through Git / Plastic commits, giving a clear, per-member contribution history for both code and scenes.

---

## 12. Environment Requirements

**Hardware**

- ARCore-certified Android smartphone, **Android 9.0+ (API 29+)**, with RGB camera and IMU, in a low-cost VR viewer (e.g. Cardboard).
- Laptop/desktop with a webcam — required only for Pipelines 1 & 2 and for TOSHFA body-pose tracking.
- Both devices on the **same local Wi-Fi** (not required for Pipeline 3).

**Software**

- **Python 3.9+** (Pipelines 1 & 2 backends).
- **Unity 2022.3 LTS** + Unity Hub.
- Android SDK / Android Build Support module, with **USB debugging** enabled.

**Required Unity packages**

- AR Foundation, ARCore XR Plugin, XR Plugin Management
- Unity Inference Engine (Sentis) — Pipelines 2 & 3
- TextMeshPro, Input System

**Python dependencies** (`requirements.txt`, Pipelines 1 & 2)

```
mediapipe>=0.10
opencv-python
numpy
scipy
```

> Pipeline 3 requires **no Python dependencies** — all inference runs on-device through Unity Sentis.

---

## 13. Setup Instructions

**1. Clone the backend and install dependencies**

```bash
git clone https://github.com/mohamed1232005/GP-VR-Guardian-System.git
cd GP-VR-Guardian-System
git checkout ARcore_simple_toshfa        # or the branch for your target pipeline
pip install -r requirements.txt
```

**2. Get the Unity scene from Unity Cloud**

1. Open **Unity Hub** and sign in.
2. **Projects → Add → Clone from Unity Cloud**.
3. Paste the scene link for your pipeline (see [§11](#11-repository-structure--branching-strategy)).
4. Wait for the sync to finish, then open the scene in **Unity 2022.3 LTS**.

**3. Configure the network (Pipelines 1 & 2)**

- Start the backend: `python server.py` — wait for `READY — waiting for mobile connection…`.
- In Unity, select the **`GuardianSystem`** object → Inspector → **NETWORK** → set **Server IP** to the host PC's local IP (e.g. `192.168.1.10`).
- Confirm phone and PC share the same Wi-Fi. Default ports: **UDP 9998** (frames) / **TCP 9999** (results) — configurable in `config.py`.

---

## 14. Deployment Instructions

**Unity Player Settings (Android / ARCore)**

- **Minimum API Level:** 29 (Android 9.0).
- **Scripting Backend:** IL2CPP · **Target Architecture:** ARM64.
- **Graphics API:** OpenGLES3 (remove *Auto*; ARCore requires OpenGLES3 or Vulkan).
- **XR Plug-in Management → Android:** enable **ARCore**.

**Build & run**

1. Connect the phone via USB and enable **USB Debugging**.
2. **File → Build Settings → Android → Switch Platform**.
3. Click **Build and Run** — Unity compiles and installs the APK on the device.
4. When the app shows `AR Camera OK / Net OK`, the pipeline is live.

**On-device deployment (Pipeline 3).** Build the `Warning_System_Yousef-hAMZA` scene to a single standalone APK. No server, Wi-Fi, or host PC is required — install and run.

---

## 15. Usage Guide

1. Place the phone in the VR viewer and look at the floor — ARCore detects the floor plane.
2. Confirm the safe area: the **yellow preview** boundary locks into a **green/cyan** boundary anchored to the real floor.
3. Move around — approaching an edge escalates the warning state (see below) with visual, audio, and haptic feedback.
4. Use your hand (fingertip ray) to point and interact; press **START REHAB** in the VR lobby to enter TOSHFA.
5. Follow the guided lower-back routine (10 reps by default); a **session summary** appears on completion.

**Safety warning state machine**

| State | Trigger | Feedback |
|-------|---------|----------|
| **SAFE** | Well inside the boundary | Normal play; boundary subtle |
| **CAUTION** | Approaching an edge | Boundary glows; soft audio cue |
| **DANGER** | Very near / at the edge | Strong visual + audio + haptic alert |
| **FREEZE** | Crossing / contact | Scene freezes and dims until the user returns inside |

> Thresholds (distances that trigger each state) are configurable in `config.py` (networked) or the scene inspector (on-device).

---

## 16. API Documentation

The Python backend and the Unity client communicate over a lightweight, **two-channel socket protocol**. There is no REST/HTTP layer.

| Channel | Direction | Transport | Payload |
|---------|-----------|-----------|---------|
| Frame stream | Unity → Python | **UDP** (default `9998`) | JPEG-compressed RGB camera frame |
| Result stream | Python → Unity | **TCP** (default `9999`) | JSON perception + safety result |

**Result message (Python → Unity, JSON) — representative schema**

```jsonc
{
  "frame_id": 1287,
  "hand": {
    "detected": true,
    "gesture": "pointing",          // none | pointing | pinch
    "confidence": 0.94,
    "landmarks": [ {"x":0.51,"y":0.42,"z":-0.08} ]   // 21 entries
  },
  "floor": {
    "valid": true,
    "normal": [0.01, 0.99, 0.02],   // plane normal
    "height": 0.0                    // Y = 0 reference
  },
  "safety": {
    "state": "CAUTION",             // SAFE | CAUTION | DANGER | FREEZE
    "nearest_distance_m": 0.46
  },
  "pose": {                          // present in TOSHFA sessions
    "detected": true,
    "landmarks": [ ]                 // 33 entries (BlazePose)
  }
}
```

**Frame packet (Unity → Python, UDP).** Raw JPEG bytes of the current camera frame (downscaled, e.g. 320×240) for low-latency streaming. Configuration (ports, IP, resolution, thresholds) is centralized in `config.py`.

---

## 17. Data & Session Schema

The system uses **no relational database**. Runtime state is held in memory; the only persisted artefact is a **TOSHFA session summary** exported as JSON to `persistentDataPath/sessions/` on completion.

```jsonc
{
  "session_id": "2026-06-19T14:32:10",
  "exercise": "lower_back_step_and_reach",
  "total_reps": 10,
  "completed_reps": 10,
  "duration_seconds": 312,
  "form_warnings": 3,
  "guardian_pauses": 1,
  "completed": true
}
```

---

## 18. Performance & Results

| Pipeline | Unity Render FPS | Notes |
|----------|------------------|-------|
| **Pipeline 1** — ARCore + MediaPipe | **~60 FPS** | Lightest on-device load; primary production build |
| **Pipeline 2** — AI & Computer Vision | **27.5 – 30 FPS** | Heavier multi-model perception |
| **Pipeline 3** — On-Device Guardian | **~35 FPS** | Standalone; no network round-trip |

| Metric | Target | Result |
|--------|--------|--------|
| Boundary detection error | ≤ 10 cm | Achieved |
| End-to-end latency (sensor → alert) | ≤ 100 ms | Achieved (avg ~52 ms) |
| Floor detection confidence | High | Confirmed across 578 test frames |
| Safety-state accuracy | ≥ 90% | Achieved in test scenarios |

---

## 19. Testing & Validation

The system was evaluated against the functional safety requirements rather than on human subjects (technical-feasibility scope).

| Test | What it checks | Result |
|------|----------------|--------|
| **Boundary lock** | Boundary stays fixed to the real floor under head movement | Pass — anchored, no drift |
| **Warning escalation** | Correct Safe → Caution → Danger → Freeze transitions near edges | Pass |
| **Hand interaction** | Fingertip ray places/confirms boundary points | Pass |
| **Floor detection** | Plane found on varied floors / lighting | Pass; Y=0 fallback on failure |
| **TOSHFA transition** | Session starts only inside a locked safe area | Pass |
| **Guardian pause/resume** | TOSHFA pauses on safety event, resumes when clear | Pass |

---

## 20. Screenshots & Demo

Image assets live in **`docs/screenshots/`** and the full walkthrough is in the demo video (see [Project Links](#24-project-links)). Key captures included with the submission:

| Capture | Shows |
|---------|-------|
| `floor_scan` | AR floor scanning with the yellow preview boundary |
| `boundary_locked` | Locked green/cyan boundary anchored to the real floor |
| `hand_tracking` | 21-joint hand skeleton with the fingertip interaction ray |
| `warning_states` | Caution / Danger / Freeze warning visuals |
| `vr_lobby` | VR lobby with the START REHAB (TOSHFA) button |
| `toshfa_session` | TOSHFA rehabilitation scene with rep tracking |

---

## 21. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Floor not detected | Dark or textureless floor | Move to a well-lit, textured area; a Y=0 fallback activates after ~5 s |
| `Net` fails / no data | Wrong Server IP, different Wi-Fi, server not running, or firewall | Verify same Wi-Fi, correct Server IP, `server.py` running, and that the ports are allowed |
| Hand lost near screen edge | Hand outside camera frame | Keep the hand centred in view |
| Boundary jitter / freeze | Transient UDP packet loss | App holds the last safe state; restart the server if it persists |
| Low frame rate | Background apps / heavy pipeline | Close other apps; Pipeline 1 is the lightest |
| ARCore build fails | Graphics API or API level | Set Graphics API to OpenGLES3 and Minimum API Level to 29 |

---

## 22. Reproducing & Verifying the Build

A clean reproduction of the **primary build (Pipeline 1)**:

```bash
# 1. Backend
git clone https://github.com/mohamed1232005/GP-VR-Guardian-System.git
cd GP-VR-Guardian-System
git checkout ARcore_simple_toshfa
pip install -r requirements.txt
python server.py            # expect: READY — waiting for mobile connection…

# 2. Unity
#   - Unity Hub → Clone from Unity Cloud → ARCore_pipeline_with_simple_toshfa
#   - Open in Unity 2022.3 LTS, set Server IP on the GuardianSystem object
#   - File → Build Settings → Android → Build and Run
```

**Verification checklist**

- [ ] Backend prints `READY` and accepts the phone connection.
- [ ] App shows `AR Camera OK / Net OK` on launch.
- [ ] Floor is detected and the boundary locks (green/cyan).
- [ ] Hand skeleton and fingertip ray appear and track.
- [ ] Warnings escalate correctly near the boundary edge.
- [ ] START REHAB launches TOSHFA inside the locked area.

> All dependencies and assets are linked in this README; following the steps above reproduces a working build end to end.

---

## 23. Contributions

All members contributed to system architecture, requirements analysis, technical documentation, and integration. Per-member focus areas:

| Member | Role | Key contributions |
|--------|------|-------------------|
| **Mohamed Ehab Yousry** | Team Lead | System architecture, monocular depth estimation & depth caching, FPS optimization, Unity scene building, integration |
| **Hamza Abdelmoreed** | Integration Lead | All-modules integration, boundary drawing & area detection, RANSAC ground estimation, Unity scenes, Python–Unity pipeline |
| **Ameen Gamal** | Co-Integration Lead | Python–Unity pipeline, VR lobby, Android dev environment, ARCore XR Plugin configuration, FPS optimization |
| **Yousef Selim Shawky** | FPS & ARCore Lead | FPS optimization across modules, ARCore Unity project, pose estimation, on-device Pipeline 3 |

> Contribution history is reflected in the per-pipeline branches (GitHub) and the `MVP-Guardian-Unity` repository (Unity Cloud / Plastic SCM). Every member contributed through tracked commits.

---

## 24. Project Links

| Resource | Link |
|----------|------|
| GitHub repository | https://github.com/mohamed1232005/GP-VR-Guardian-System |
| Unity Cloud organization (all scenes) | https://cloud.unity.com/home/organizations/11270618913273/projects |
| Pipeline 1 scene | [`ARCore_pipeline_with_simple_toshfa`](https://cloud.unity.com/home/organizations/11270618913273/projects/3af6106f-603f-411a-9087-d805c64f2b72/plastic-scm/organizations/11270618913273/repositories/MVP-Guardian-Unity%2FARCore_pipeline_with_simple_toshfa) |
| Pipeline 2 scene | [`AI_Guardian_Pipeline_2_hamza`](https://cloud.unity.com/home/organizations/11270618913273/projects/3af6106f-603f-411a-9087-d805c64f2b72/plastic-scm/organizations/11270618913273/repositories/MVP-Guardian-Unity%2FAI_Guardian_Pipeline_2_hamza/branch/%2Fmain/tree/) |
| Pipeline 3 + TOSHFA scene | [`Warning_System_Yousef-hAMZA`](https://cloud.unity.com/home/organizations/11270618913273/projects/3af6106f-603f-411a-9087-d805c64f2b72/plastic-scm/organizations/11270618913273/repositories/MVP-Guardian-Unity%2FWarning_System_Yousef-hAMZA/branch/%2Fmain%2FFix%20CODE/tree/) |

---

## 25. License

This project was developed as a graduation project for academic purposes.
© 2026 Team 25 — All rights reserved.

---

*Built with ARCore · MediaPipe · Depth Anything V2 · SegFormer · Unity Sentis · Python.*
