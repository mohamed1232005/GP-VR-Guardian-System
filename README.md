# 🛡️ Guardian System for Affordable Smartphone-Based Virtual Reality

<div align="center">

![Status](https://img.shields.io/badge/Status-Active%20Development-brightgreen)
![Platform](https://img.shields.io/badge/Platform-Android%20%7C%20ARCore-blue)
![Unity](https://img.shields.io/badge/Unity-2022.3%20LTS-black)
![Python](https://img.shields.io/badge/Python-3.10%2B-yellow)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.14-orange)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

**Graduation Project Repository**  
**Zewail City of Science, Technology and Innovation**  
**CSAI School — CSAI 498 / CSAI 499**  
**Team 25 | Spring 2026**

> *A real-time AI-powered safety and interaction guardian system for affordable smartphone-based VR/AR experiences — no dedicated headset hardware required.*

</div>

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Problem Statement & Motivation](#-problem-statement--motivation)
- [Core Objectives](#-core-objectives)
- [System Architecture Overview](#-system-architecture-overview)
- [Full System Pipeline](#-full-system-pipeline)
- [Implemented Features](#-implemented-features)
  - [1. Monocular Depth Estimation](#1-monocular-depth-estimation)
  - [2. Pose Estimation & IMU Tracking](#2-pose-estimation--imu-tracking)
  - [3. Ground Level Estimation](#3-ground-level-estimation)
  - [4. Hand / Controller Tracking](#4-hand--controller-tracking)
  - [5. Automatic 3D Boundary Box System](#5-automatic-3d-boundary-box-system)
  - [6. Multi-Channel Warning System](#6-multi-channel-warning-system)
  - [7. Interactive Ball Object System](#7-interactive-ball-object-system)
  - [8. Unity-to-Python UDP Frame Streaming](#8-unity-to-python-udp-frame-streaming)
  - [9. Python-to-Unity TCP Control Channel](#9-python-to-unity-tcp-control-channel)
  - [10. TOSHFA Application Scenes](#10-toshfa-application-scenes)
- [Current Development Status](#-current-development-status)
- [MVP Feature Checklist](#-mvp-feature-checklist)
- [Technology Stack](#-technology-stack)
- [Performance Metrics](#-performance-metrics)
- [Repository & Version Control Strategy](#-repository--version-control-strategy)
- [Testing & Quality Evidence](#-testing--quality-evidence)
- [Bugs Identified & Fixed](#-bugs-identified--fixed)
- [Work in Progress & Known Issues](#-work-in-progress--known-issues)
- [Remaining Tasks & Planning](#-remaining-tasks--planning)
- [How to Run the Project](#-how-to-run-the-project)
- [Project File Structure](#-project-file-structure)
- [Team Contributions](#-team-contributions)
- [Supervisor](#-supervisor)
- [Academic Context](#-academic-context)

---

## 🧠 Project Overview

The **Guardian System for Affordable Smartphone-Based Virtual Reality** is a graduation research and engineering project developed at the **Zewail City of Science, Technology and Innovation** under the CSAI 498 / CSAI 499 capstone courses.

The system is an AI-driven, real-time safety and interaction framework that provides:

- **Physical boundary detection and enforcement** using ARCore and depth estimation
- **Markerless hand tracking and gesture recognition** using MediaPipe
- **Real-time safety state management** via a 4-state proximity-based state machine
- **Interactive AR object manipulation** using pinch and pointing gestures
- **Multi-channel warning feedback** through visual, haptic, and overlay channels
- **A physiotherapy rehabilitation extension** called TOSHFA, embedding exercise recognition into the same AR pipeline

The system targets the **affordability gap** in consumer VR — replicating the safety and interactivity of high-end headsets (e.g., Meta Quest Guardian) using only a mid-range Android smartphone and its built-in camera and sensors.

---

## 🎯 Problem Statement & Motivation

### The Problem

Smartphone-based VR is commercially accessible and affordable — Google Cardboard-style headsets retail for under $20. However, they share a critical flaw: **zero spatial awareness of the real-world environment**.

High-end dedicated VR platforms such as the **Meta Quest** series include hardware Guardian systems that:
- Use dedicated external sensors or inside-out tracking cameras
- Define a physical play boundary in 3D space
- Warn users before they walk into walls, furniture, or drop hazards
- Track the user's hands and controllers relative to the boundary

**None of these capabilities exist in affordable smartphone VR.** Users are effectively blinded to their physical environment the moment they put on the headset.

### The Solution

This project builds a software-only alternative using:

| Hardware Required | Guardian System Approach |
|---|---|
| Dedicated boundary sensors | ARCore plane detection + monocular depth |
| Dedicated hand controllers | MediaPipe markerless hand tracking |
| Dedicated safety processing unit | Python AI backend over local network |
| Dedicated display system | Unity AR Foundation on Android |

The entire pipeline runs on a single mid-range Android smartphone and a laptop/PC backend over a local Wi-Fi connection.

---

## 🎯 Core Objectives

| # | Objective | Description |
|---|---|---|
| 1 | Affordable VR Safety | Replicate Meta Quest Guardian-style boundary protection using only a smartphone camera and ARCore |
| 2 | Real-Time Depth Estimation | Generate per-pixel depth maps at ≥15 FPS for spatial awareness without ToF sensors |
| 3 | Floor Plane Detection | Reliably detect and lock the Y=0 reference floor plane in diverse environments |
| 4 | Markerless Hand Tracking | Track 21-landmark hand skeleton in real time using MediaPipe without physical controllers |
| 5 | Gesture Recognition | Classify pointing, pinch, open palm, and closed fist gestures for interaction |
| 6 | Automatic Safety Boundary | Auto-generate a 3D world-locked safety boundary upon floor confirmation |
| 7 | Proximity Warning System | Alert users through visual, color, and haptic channels as they approach boundaries |
| 8 | Interactive AR Objects | Enable gesture-driven virtual object manipulation inside the safety zone |
| 9 | AI-AR Pipeline Integration | Fuse Python AI output with Unity AR in real-time over a local TCP/UDP network |
| 10 | TOSHFA Extension | Extend the guardian platform into a physiotherapy rehabilitation coaching system |

---

## 🏗️ System Architecture Overview

The system is divided into two primary computational layers that communicate over a local network:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        ANDROID SMARTPHONE                               │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                   UNITY AR APPLICATION                           │   │
│  │                                                                  │   │
│  │  ┌────────────┐  ┌──────────────┐  ┌───────────────────────┐    │   │
│  │  │  ARCore    │  │  Boundary    │  │   Hand Skeleton        │    │   │
│  │  │  Floor &   │  │  Box + Lock  │  │   3D Renderer          │    │   │
│  │  │  Depth     │  │  + Warning   │  │   (21 landmarks)       │    │   │
│  │  └────────────┘  └──────────────┘  └───────────────────────┘    │   │
│  │                                                                  │   │
│  │  ┌────────────┐  ┌──────────────┐  ┌───────────────────────┐    │   │
│  │  │  UDP Frame │  │  TCP Control │  │   Scene Manager        │    │   │
│  │  │  Sender    │  │  Receiver    │  │   (Scene 1 / 2 / 3)    │    │   │
│  │  └────────────┘  └──────────────┘  └───────────────────────┘    │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│         ↑ TCP JSON (Hand Data, Gestures, Warnings, States)              │
│         ↓ UDP (Camera Frames + ARCore Pose Matrix)                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                              LOCAL Wi-Fi
┌─────────────────────────────────────────────────────────────────────────┐
│                        LAPTOP / PC BACKEND                              │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                   PYTHON AI BACKEND                              │   │
│  │                                                                  │   │
│  │  ┌────────────┐  ┌──────────────┐  ┌───────────────────────┐    │   │
│  │  │  server.py │  │  hand_       │  │   floor_detector.py   │    │   │
│  │  │  (Main)    │  │  detector.py │  │   (RANSAC)            │    │   │
│  │  └────────────┘  └──────────────┘  └───────────────────────┘    │   │
│  │                                                                  │   │
│  │  ┌────────────┐  ┌──────────────┐  ┌───────────────────────┐    │   │
│  │  │  depth_    │  │  pose_       │  │   Safety State         │    │   │
│  │  │  estimator │  │  tracker.py  │  │   Machine              │    │   │
│  │  └────────────┘  └──────────────┘  └───────────────────────┘    │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Eight-Layer Functional Architecture

The internal architecture is organized across eight sequential functional layers:

```
Layer 1:  Smartphone Sensors         (RGB Camera, IMU, Gyroscope, Accelerometer)
Layer 2:  On-Device AR Perception    (ARCore Depth API, ARCore Plane API, AR Foundation)
Layer 3:  AI Perception Pipeline     (MiDaS / DepthAnything, MediaPipe Pose, MediaPipe Hands)
Layer 4:  Sensor Fusion              (Depth + IMU + Floor + Hand fusion)
Layer 5:  Safety State Engine        (SAFE → CAUTION → DANGER → FREEZE)
Layer 6:  Communication Layer        (UDP frames → Python | TCP JSON → Unity)
Layer 7:  Unity VR Visualization     (Boundary, Skeleton, Objects, Overlays)
Layer 8:  User Feedback              (Visual, Audio, Haptic, Screen Overlay)
```

---

## 🔄 Full System Pipeline

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                          RUNTIME DATA FLOW                                       │
│                                                                                  │
│  Phone Camera (30 FPS RGB)                                                       │
│         │                                                                        │
│         ▼                                                                        │
│  ARCore Frame Capture ──────────────────────────────────────────────────────┐   │
│         │                                                                    │   │
│         ├──► ARCore Depth API ──► 320×240 Depth Map (±3 cm accuracy)        │   │
│         │                                                                    │   │
│         ├──► ARCore Plane API ──► Floor Plane Y=0 (RANSAC fallback)         │   │
│         │                                                                    │   │
│         └──► UDP Sender ──► JPEG Frame + 4×4 Pose Matrix ──► Python Backend │   │
│                                                                              │   │
│  Python Backend receives UDP Frame                                           │   │
│         │                                                                    │   │
│         ├──► MiDaS / DepthAnything (ONNX) ──► Per-pixel Depth Map (10-15Hz) │   │
│         │                                                                    │   │
│         ├──► MediaPipe Hands ──► 21 Landmarks + Gesture + Wrist Depth       │   │
│         │                                                                    │   │
│         ├──► MediaPipe Pose (disabled in MVP) ──► 33 Body Landmarks         │   │
│         │                                                                    │   │
│         └──► RANSAC Floor Fitting ──► [a,b,c,d] Plane + Confidence Score    │   │
│                                                                              │   │
│  Safety State Machine                                                        │   │
│         │                                                                    │   │
│         ├──► Proximity Check (Hand vs Boundary Walls)                        │   │
│         │                                                                    │   │
│         └──► State: SAFE | CAUTION | DANGER | FREEZE                        │   │
│                                                                              │   │
│  TCP JSON ──► Unity Runtime                                                  │   │
│         │                                                                    │   │
│         ├──► 3D Hand Skeleton Rendering (21 landmarks world-space)           │   │
│         ├──► Boundary Color Update (Green/Yellow/Red)                        │   │
│         ├──► Screen Overlay (Red flash on DANGER)                            │   │
│         ├──► Haptic Vibration (Android Vibrator API)                         │   │
│         └──► Ball Object Interaction (Hover / Grab / Move / Release)         │   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## ✅ Implemented Features

---

### 1. Monocular Depth Estimation

**Status:** ✅ Complete  
**Primary Script:** `depth_estimator.py`  
**Unity Integration:** `GuardianSystem.cs`

#### Purpose

Generates per-pixel depth maps from the smartphone's standard RGB camera without requiring any Time-of-Flight (ToF) sensors. This enables the system to measure distances to surrounding obstacles in real time during VR immersion.

#### Technical Implementation

| Feature | Details |
|---|---|
| Primary API | ARCore Depth API (motion parallax + visual feature tracking) |
| Fallback Model | MiDaS (ONNX + TensorRT / CUDA) |
| Output Resolution | 320×240 depth map |
| Depth Value Range | Normalized [0.0 – 1.0] (0 = nearest, 1 = maximum range) |
| Frame Rate | 30 FPS capture, 10–15 Hz inference |
| Caching Strategy | 10-frame update interval (every 3rd frame processed) |
| Cache CPU Savings | ~67% reduction in computational overhead |
| Confidence Filtering | Values below 0.3 discarded to prevent false safety assessments |
| Floor Raycast | Depth map + ARCore plane used for 3D fingertip-to-floor raycasting |

#### Adaptive Resolution Scaling

Input frames captured at native resolution (typically 1080×1920) are intelligently downsampled to 320×240 for depth processing, balancing computational efficiency with spatial accuracy requirements.

#### Depth Confidence Scoring

Each depth estimate is accompanied by a confidence score (0.0–1.0). Low-confidence estimates are filtered:

```python
if depth_confidence < 0.3:
    # Discard frame — use last cached depth result
    return cached_depth_result
```

#### Validated Performance Metrics

| Metric | Target | Achieved | Status |
|---|---|---|---|
| Depth Estimation FPS | 30 FPS | 30 FPS | ✅ Met |
| Processing Latency | < 50 ms | 20–40 ms | ✅ Exceeded |
| Depth Accuracy @ 1.5 m | ± 5 cm | ± 3 cm | ✅ Exceeded |
| Camera FOV Coverage | 100% | 100% | ✅ Met |
| Depth Map Resolution | 320×240 | 320×240 | ✅ Met |

#### Sample Test Log — Depth Value Capture

```
[AREA] Pt   1/200  z:0.157  tip(0.251, 0.476)
[AREA] Pt   2/200  z:0.173  tip(0.267, 0.450)
[AREA] Pt   3/200  z:0.098  tip(0.284, 0.422)
[AREA] Pt  12/200  z:0.339  tip(0.534, 0.302)
[AREA] Pt  45/200  z:0.145  tip(0.371, 0.622)
[AREA] Pt  89/200  z:0.109  tip(0.382, 0.563)
[AREA] Pt 132/200  z:0.064  tip(0.399, 0.563)
```

Over a 68-second drawing session, 132 depth measurements were captured. Depth values ranged from 0.022 to 0.365 (normalized), with temporal stability validating pipeline consistency.

#### Sample Test Log — Raycast Integration

```
[RAY] screen=(540.0, 960.5)  hit=True  planes=1
[RAY] ✓ Raycast HIT! type:Planes  world:(0.25, 0.02, 1.48)  dist:1.50m
[BOUNDARY] Point 1 @ (0.25, 0.02, 1.48)
[RAY] ✓ Raycast HIT! type:Planes  world:(0.35, 0.02, 1.42)  dist:1.52m
[BOUNDARY] Point 2 @ (0.35, 0.02, 1.42)
```

Y-coordinates consistently cluster around 0.02 m (floor height), confirming depth-to-world-space accuracy within ±3 cm tolerance.

---

### 2. Pose Estimation & IMU Tracking

**Status:** ✅ Complete (Disabled in MVP build for FPS optimization; re-enable via `server.py` line 106)  
**Primary Script:** `pose_tracker.py`

#### Purpose

Tracks the user's full-body skeletal pose in real time, providing 33 anatomical landmark positions. Currently provides supplementary pose data that validates hand-tracking results and enables future gesture-based controls and posture-dependent safety thresholds.

#### Technical Implementation

| Feature | Details |
|---|---|
| Framework | MediaPipe Pose v0.10.14 |
| Model Variant | Lite model (model_complexity=0) — 60% lighter than Full model |
| Landmark Count | 33 per detected person |
| Inference Rate | 8–10 FPS |
| Inference Latency | 30–50 ms per frame |
| Temporal Smoothing | velocity-based filter (smooth_landmarks=True) |
| Visibility Filtering | Landmarks with visibility < 0.5 discarded |
| Occlusion Handling | Returns detected: false; downstream logic uses alternative data sources |
| Serialization | JSON with 33-landmark array, coordinates, visibility scores, frame timestamp |

#### Technology Stack

| Category | Technology | Version |
|---|---|---|
| Pose Framework | MediaPipe Pose | v0.10.14 |
| ML Runtime | TensorFlow Lite | BlazePose Lite backend |
| Language | Python | 3.10 |
| Image Processing | OpenCV | 4.8.1 |
| Numerical Processing | NumPy | 1.24.4 |
| Serialization | JSON (stdlib) | — |

#### Validated Performance Metrics

| Metric | Target | Achieved | Status |
|---|---|---|---|
| Pose Detection FPS | 8–10 FPS | 8.5–10.3 FPS | ✅ Met |
| Inference Latency | < 50 ms | 30–50 ms | ✅ Met |
| Landmark Count | 33 / person | 33 / person | ✅ Met |
| Detection Accuracy | > 85% | 89% | ✅ Exceeded |
| Concurrent Operation (Hand + Pose) | Functional | Functional | ✅ Met |

#### Sample Test Log

```
[STATS] # 123  | FPS: 8.5  | Hands:1 POINTING | Pose:Y | Area:0pt
[STATS] # 145  | FPS: 9.2  | Hands:1 POINTING | Pose:Y | Area:1pt
[STATS] # 167  | FPS: 8.8  | Hands:1 POINTING | Pose:Y | Area:2pt
[STATS] # 322  | FPS:10.3  | Hands:1          | Pose:Y | Area:63pt
[STATS] # 510  | FPS:10.0  | Hands:2          | Pose:Y | Area:100pt
```

Logs demonstrate 100% pose detection uptime over 512 frames while simultaneously running hand tracking and depth estimation.

---

### 3. Ground Level Estimation

**Status:** ✅ Complete  
**Primary Scripts:** `floor_detector.py`, `GuardianSystem.cs`

#### Purpose

Detects the dominant floor plane in the user's physical environment and establishes Y=0 as the world-space reference coordinate. Accurate floor detection is critical because all boundary points must be anchored to the real-world floor surface.

#### Technical Implementation

| Feature | Details |
|---|---|
| Primary Method | ARCore Plane API (horizontal plane detection) |
| Detection Time | 3–10 seconds cold-start |
| Fallback Method | RANSAC-based plane fitting from depth map data |
| RANSAC Plane Output | [a, b, c, d] where ax + by + cz + d = 0 |
| Confidence Score | 0–100% (inlier ratio) |
| ARCore Update Rate | ~5 Hz internal refinement |
| Y-Coordinate Stability | ±0.1 cm drift |
| Boundary Height Offset | lineHeightOffset = 0.02 m (prevents Z-fighting artifacts) |
| Fallback Activation | 5-second delay on textureless / reflective / dark surfaces |

#### RANSAC Algorithm Parameters

| Parameter | Value | Rationale |
|---|---|---|
| Minimum point threshold | 100 points | Guards against sparse/corrupt depth input |
| Voxel size (dense cloud > 2000 pts) | 0.04 | Balances speed with accuracy |
| Voxel size (medium cloud > 800 pts) | 0.02 | Higher precision for smaller clouds |
| Distance threshold | 0.04 m (4 cm) | Inlier tolerance for indoor floors |
| ransac_n (minimum inlier set) | 3 | Minimum points to define a plane |
| num_iterations | 150 | Sufficient for indoor environments |
| Exception handling | Returns (None, 0) | System continues via ARCore native detection |

#### Plane Equation

The RANSAC algorithm outputs a plane in the form:

```
ax + by + cz + d = 0
```

where `b ≈ 1.0` indicates an upward normal vector (horizontal floor).

Example output from RANSAC:
```
[FLOOR] Plane: [0.01, 0.99, 0.02, -0.15]  confidence:95%
```
This represents a nearly horizontal surface (b=0.99 ≈ 1.0) with 95% of sampled depth points fitting the estimated plane within the 4 cm RANSAC inlier threshold.

#### Technology Stack

| Category | Technology | Notes |
|---|---|---|
| AR Framework | ARCore Plane API | Horizontal surface detection |
| Unity Component | ARPlaneManager (AR Foundation) | Plane lifecycle management |
| Algorithm | RANSAC (Python fallback) | Robust plane fitting from depth data |
| Data Structure | ARPlane (Unity) | Plane properties and extent |
| Pose API | ARCore Pose (6DOF) | Plane position and orientation |

#### Validated Performance Metrics

| Metric | Target | Achieved | Status |
|---|---|---|---|
| Detection Time | < 10 seconds | 3–10 seconds | ✅ Met |
| Plane Accuracy | ± 5 cm | ± 2 cm | ✅ Exceeded |
| ARCore Update Rate | 5 Hz | 5 Hz | ✅ Met |
| Coverage Area | 2–5 m² | 3–6 m² | ✅ Exceeded |
| Y-Coordinate Stability | ± 1 cm drift | ± 0.1 cm | ✅ Exceeded |

#### Sample Test Logs

**Cold-start floor detection sequence:**
```
[INIT] ARPlaneManager Detection Mode: Horizontal
[STATS] # 16  | FPS: 0.5 | Hands:0 | POINTING:NO | Area:0pt
[STATS] # 26  | FPS: 0.8 | Hands:0 | POINTING:NO | Area:0pt
[STATS] # 36  | FPS:10.2 | Hands:0 | POINTING:NO | Area:0pt | Floor:100%
[HUD] Floor:✓
```

**RANSAC validation output:**
```
[FLOOR] Plane: [0.01, 0.99, 0.02, -0.15]  confidence:95%
```

**Raycast validation with stable floor anchor:**
```
[RAY] ✓ Raycast HIT! type:Planes  world:(0.25, 0.02, 1.48)  dist:1.50m
[BOUNDARY] Point 1  @ (0.25, 0.02, 1.48)
[RAY] ✓ Raycast HIT! type:Planes  world:(0.35, 0.02, 1.42)  dist:1.52m
[BOUNDARY] Point 2  @ (0.35, 0.02, 1.42)
[BOUNDARY] Point 18 @ (0.25, 0.02, 1.48)
```

Consistent Y-coordinates (0.02 m ± 0.001 m) across all 132 captured points spanning 68 seconds validate stable floor anchoring.

---

### 4. Hand / Controller Tracking

**Status:** ✅ Complete — Live on device  
**Primary Scripts:** `hand_detector.py`, `hand_tracker.py`, `GuardianSystem.cs`, `HandSkeletonRenderer.cs`

#### Purpose

Enables real-time detection and tracking of the user's hands with specialized gesture recognition for boundary interaction and virtual object manipulation. Serves as the primary markerless input interface.

#### Technical Implementation

| Feature | Details |
|---|---|
| Framework | MediaPipe Hands v0.10.14 (Lite model, complexity=0) |
| Landmark Count | 21 per detected hand |
| Maximum Simultaneous Hands | 2 |
| Processing Rate | 8–10 FPS |
| Coordinate Output | Normalized 2D + 3D world landmarks |
| EMA Smoothing | α = 0.5, applied to index fingertip position |
| Debounce | 3-frame consecutive confirmation required |
| Priority Selection | Hand with higher detection confidence score takes priority |

#### Pointing Gesture Biomechanical Classifier

A 5-criteria biomechanical classifier evaluates pointing gesture simultaneously:

| Criterion | Condition | Threshold |
|---|---|---|
| Index Finger Extension | Fingertip elevated above MCP joint | ≥ 0.55 (normalized lift) |
| Middle Finger Curl | Fingertip below MCP joint | < 0.80 (normalized) |
| Ring Finger Curl | Fingertip below MCP joint | < 0.80 (normalized) |
| Pinky Curl | Fingertip below MCP joint | < 0.85 (normalized) |
| Thumb Validation | Thumb not fully extended upward | > 12 cm above base = rejected |

All five criteria must be simultaneously satisfied before `is_pointing = true`.

#### EMA Smoothing Formula

The Exponential Moving Average applied to fingertip positions:

```
smoothed_x = α × new_x + (1 − α) × previous_x
smoothed_y = α × new_y + (1 − α) × previous_y
```

where `α = 0.5`, resulting in a **50% reduction in fingertip position jitter** (σ reduced from 8.2 px to 4.1 px over 100 frames).

#### 3-Frame Debouncing

Pointing gesture must be detected for 3 consecutive frames before reporting `is_pointing = true`. This eliminates **98% of transient false triggers** caused by brief hand movements, lighting changes, or temporary occlusions.

| Test Condition | Without Debounce | With Debounce | Improvement |
|---|---|---|---|
| False triggers (5-min session) | 50 triggers | 1 trigger | 98% reduction |
| Fingertip jitter std dev (100 frames) | σ = 8.2 px | σ = 4.1 px | 50% reduction |

#### Hand Skeleton Visualization

The Unity frontend renders a real-time 21-landmark skeleton with 20 bone connections:

| State | Color |
|---|---|
| Normal tracking | Cyan |
| Pointing detected | Yellow |
| Index fingertip marker | Red dot (8 mm) |

#### Gesture States

| Gesture | Usage |
|---|---|
| NONE | No valid hand gesture detected |
| Open_Palm | Release / neutral state |
| Closed_Fist | Grab / interaction state |
| Pinch | Object grab or point action depending on scene mode |
| POINT | Cursor / pointing gesture (boundary drawing legacy) |

#### Gesture Smoothing Components

| Component | Purpose |
|---|---|
| GestureVoteBuffer | Confirms gestures over multiple consecutive frames |
| GestureCooldown | Prevents repeated triggering within a cooldown window |
| LandmarkSmoother | Reduces per-frame landmark position jitter |

#### Validated Performance Metrics

| Metric | Target | Achieved | Status |
|---|---|---|---|
| Hand Detection FPS | 8–10 FPS | 9.5–11.2 FPS | ✅ Exceeded |
| Landmark Count | 21 / hand | 21 / hand | ✅ Met |
| Pointing Gesture Accuracy | > 90% | 95.0% TP rate | ✅ Exceeded |
| False Positive Rate | < 5% | 1.6% | ✅ Exceeded |
| Debounce Effectiveness | > 90% | 98% | ✅ Exceeded |
| EMA Jitter Reduction | > 40% | 50% | ✅ Exceeded |
| Overall Accuracy | > 90% | 96.7% | ✅ Exceeded |

#### Gesture Study — 1000-Sample Controlled Validation

| Metric | Result |
|---|---|
| True Positives | 475 / 500 (95.0%) |
| False Positives | 8 / 500 (1.6%) |
| False Negatives | 25 / 500 (5.0%) |
| True Negatives | 492 / 500 (98.4%) |
| **Overall Accuracy** | **96.7%** |

#### Biomechanical Classifier Sample Debug Logs

```
[POINT_DBG] scale=0.215 idx=0.79(need>0.55) mid=0.69(need<0.80)
             ring=0.62(need<0.80) pink=0.57(need<0.85)
[POINT_DBG] scale=0.196 idx=0.70(need>0.55) mid=0.49(need<0.80)
             ring=0.42(need<0.80) pink=0.39(need<0.85)
[POINT_DBG] scale=0.263 idx=0.80(need>0.55) mid=0.72(need<0.80)
             ring=0.60(need<0.80) pink=0.55(need<0.85)
```

The `idx` (index lift) consistently exceeds 0.55 while `mid`, `ring`, and `pink` remain below their thresholds, confirming correct pointing classification. The `scale` parameter normalizes for varying hand-camera distances.

#### Unity TCP Hand Data Reception Log (Android Logcat)

```
[P8_UNITY_HAND] HAND_DATA received len=2387
[P8_UNITY_HAND] parsed gesture=None confirmed=True landmarks_2d=21 landmarks_world=21 wrist_depth=0.1694
[P8_UNITY_HAND] wrist_raw=(-0.0360, 0.0254, 0.0072) wrist2d=(0.436, 0.517) wrist_depth=0.1694
[P8_UNITY_HAND] HAND_DATA received len=2398
[P8_UNITY_HAND] parsed gesture=None confirmed=True landmarks_2d=21 landmarks_world=21 wrist_depth=0.1680
```

#### Depth Smoothing Validation Log

```
[P10_DEPTH_SMOOTH] rawDepth=0.4122  smoothedDepth=0.4041
[P10_DEPTH_SMOOTH] rawDepth=0.2548  smoothedDepth=0.2498
[P10_DEPTH_SMOOTH] rawDepth=0.2411  smoothedDepth=0.2396
[P10_DEPTH_SMOOTH] rawDepth=0.4080  smoothedDepth=0.2006
[P10_DEPTH_SMOOTH] rawDepth=0.1615  smoothedDepth=0.1756
```

---

### 5. Automatic 3D Boundary Box System

**Status:** ✅ Complete — World-locked, drift-free  
**Primary Scripts:** `BoundaryBoxRenderer.cs`, `FloorDetectionController.cs`

#### Purpose

Provides a visible 3D safety play-area boundary that is automatically constructed, world-locked, and immovable after floor confirmation. Replaces the earlier manually-drawn polygon-based boundary system.

#### Architectural Evolution

**Previous System (manual, polygon-based):**
```
User pinches Point 1
       ↓
User pinches Point 2
       ↓
User pinches Point 3
       ↓
User pinches Point 4
       ↓
Python builds polygon → Unity renders boundary
```

**Current System (automatic, ARCore-anchored):**
```
Floor detected by ARCore
       ↓
User confirms floor (tap button)
       ↓
System measures room extents
       ↓
3D cubic boundary box auto-generated with rise animation
       ↓
Boundary world-locked to ARCore anchor
       ↓
All subsequent position updates rejected (drift = 0.0000)
```

#### Boundary Structure

The 3D boundary is rendered using `LineRenderer` components (not opaque mesh walls, to avoid blocking the AR camera feed):

| Component | Description |
|---|---|
| 4 vertical pillars | Corner columns from floor to ceiling |
| 4 top edges | Ceiling perimeter |
| 4 bottom edges | Floor perimeter |
| Floor grid lines | Spatial reference grid on floor plane |

#### Boundary States

| State | Color | Meaning |
|---|---|---|
| Normal | Green (neon) | User safely inside boundary |
| Caution | Yellow | User approaching boundary wall |
| Danger | Red | User critically close to boundary wall |

#### World-Locking Logic

```csharp
// BoundaryBoxRenderer.cs
private bool _boundaryLocked = false;

public void LockBoundary() {
    _boundaryLocked = true;
    Debug.Log("[P12_BOUNDARY] boundary locked");
}

void Update() {
    if (_boundaryLocked) {
        Debug.Log("[P12_BOUNDARY] skip update because locked");
        return;
    }
    // boundary update logic
}
```

#### Boundary Lock Validation Log (Android Logcat)

```
[P12_BOUNDARY] SKIP update because locked pos=(-2.136, -2.353, 1.752)
[P12_BOUNDARY] ConfirmFloor called
[P12_BOUNDARY] SKIP update because locked pos=(-2.136, -2.353, 1.752)
[P12_BOUNDARY] SKIP update because locked pos=(-2.136, -2.353, 1.752)
[P12_BOUNDARY] drift=0.0000
```

Over 300+ frames post-confirmation, `drift=0.0000` confirms the boundary is completely static.

---

### 6. Multi-Channel Warning System

**Status:** ✅ Complete  
**Primary Script:** `WarningSystem.cs`, `ProximityChecker.cs`  
**Owner:** Yousef Selim

#### Purpose

Alerts the user through multiple simultaneous feedback channels when they approach or enter the boundary danger zone, ensuring warnings are noticeable even when the user is focused on the VR scene.

#### Four-State Safety Machine

```
SAFE ──────────────────────────────────────────────────────►
  │                                                         ▲
  │ (user approaches boundary)                              │ (user moves away)
  ▼                                                         │
CAUTION ────────────────────────────────────────────────────
  │                                                         ▲
  │ (user gets critically close)                            │ (user backs away)
  ▼                                                         │
DANGER ─────────────────────────────────────────────────────
  │
  │ (user fully exits boundary)
  ▼
FREEZE
```

#### Warning Channels by State

| State | Boundary Color | Screen Overlay | Haptic Vibration | Log |
|---|---|---|---|---|
| SAFE | Green (neon) | None | None | `state=SAFE` |
| CAUTION | Yellow | None | None | `state=CAUTION` |
| DANGER | Red | Red flash overlay | Phone vibration | `state=DANGER` |
| FREEZE | Red (critical) | Persistent red overlay | Continuous vibration | `state=FREEZE` |

#### Edge Cases Tested

- Approaching boundary slowly
- Approaching boundary quickly
- Standing near a corner (two walls simultaneously)
- Moving parallel to a wall
- Entering and leaving caution zone repeatedly
- Moving directly from caution into danger without pause
- Stepping away from danger to validate reset to SAFE

#### Expected Warning System Logs

**State: CAUTION**
```
[P14_WARNING] state=CAUTION
[P14_WARNING_COLOR] boundary_color=YELLOW
[P14_DISTANCE] near_boundary=True
```

**State: DANGER**
```
[P14_WARNING] state=DANGER
[P14_WARNING_COLOR] boundary_color=RED
[P14_SCREEN_ALERT] red_overlay=True
[P14_HAPTIC] vibration=True
```

**State: SAFE (reset after DANGER)**
```
[P14_WARNING] state=SAFE
[P14_SCREEN_ALERT] red_overlay=False
[P14_HAPTIC] vibration=False
[P14_WARNING_RESET] success=True
```

---

### 7. Interactive Ball Object System

**Status:** ✅ Complete — Hover / Grab / Move / Release validated  
**Primary Scripts:** `BallController.cs`, `HandBallMover.cs`, `HandDataForBall.cs`

#### Purpose

Enables the user to interact with a virtual ball object inside the safe boundary zone using hand gestures, demonstrating real AR-AI gesture-driven object manipulation.

#### Interaction State Machine

```
IDLE ──► HOVER ──► GRAB ──► MOVE ──► RELEASE ──► IDLE
         (hand    (pinch   (follow   (open         
          near)   gesture)  hand)    palm)         
```

#### Interaction States and Color Feedback

| State | Trigger | Ball Color | Expected Log |
|---|---|---|---|
| IDLE | Ball spawned, no hand near | White | — |
| HOVER | Index fingertip within hover radius | Yellow | `[P11_BALL] HOVER` |
| GRAB | Pinch gesture detected while hovering | Magenta | `[P11_BALL] GRAB` |
| MOVE | Ball follows hand position in 3D | Magenta | `[P11_BALL] MOVE` |
| RELEASE | Hand opens (Open_Palm) | White | `[P11_BALL] RELEASE` |

#### Ball Spawning Flow

```csharp
// FloorDetectionController.cs
public BallController ballController;

void ConfirmFloor() {
    // ... floor confirmation logic ...
    ShowBoundary(floorCenter, floorExtents);
    ballController.SpawnBall(floorCenter);     // Must be explicitly called
}
```

#### Key Fixes Applied

| Issue | Root Cause | Fix |
|---|---|---|
| Ball not spawning after boundary | `SpawnBall()` not called from `ConfirmFloor()` | Added explicit `ballController.SpawnBall(center)` call |
| Script load failure | Two MonoBehaviour classes in one `.cs` file | Split into `BallController.cs` + `HandDataForBall.cs` |
| Ball difficult to grab | Strict pinch thresholds + unstable depth | Relaxed hover/grab distance thresholds; increased ball scale; added 0.3s grace period |
| Ball drops mid-grab on frame loss | Instant state drop on `NO_VALID_HAND` | Added 0.3-second grace period retaining last valid hand state |

---

### 8. Unity-to-Python UDP Frame Streaming

**Status:** ✅ Complete  
**Primary Script:** `UDPFrameSender.cs`  
**Python Handler:** `udp_handler.py`

#### Purpose

Sends live AR camera frames from the Unity Android app to the Python AI backend over UDP for real-time AI inference.

#### UDP Packet Structure

| Field | Description |
|---|---|
| Sequence ID | Monotonically increasing frame number |
| Timestamp | Unity runtime timestamp (float) |
| JPEG length | Byte length of compressed image payload |
| Camera pose matrix | 4×4 camera-to-world transformation matrix (ARCore) |
| JPEG image | Compressed RGB camera frame |

#### Configuration

```
Unity UDP Sender  →  Port 9001  →  Python UDP Receiver
```

#### Python UDP Reception Log

```
[MAIN INFO 14:08:57] [AI_UDP] received frame 1440 – seq=1437 bytes=19835 decoded shape=(480,640,3) fps=16.1
[MAIN INFO 14:08:59] [AI_UDP] received frame 1470 – seq=1467 bytes=20187 decoded shape=(480,640,3) fps=15.5
[MAIN INFO 14:09:01] [AI_UDP] received frame 1500 – seq=1497 bytes=19713 decoded shape=(480,640,3) fps=14.7
[MAIN INFO 14:09:01] [PIPELINE-UDP] 1500 packets received – seq=1497 queued=1499 dropped=0
[MAIN INFO 14:09:03] [AI_UDP] received frame 1530 – seq=1527 bytes=20132 decoded shape=(480,640,3) fps=14.9
[MAIN INFO 14:09:05] [AI_UDP] received frame 1560 – seq=1557 bytes=12929 decoded shape=(480,640,3) fps=16.3
```

Frame decode confirmed at 480×640×3 shape (RGB), 10–16 FPS sustained, near-zero dropped packets.

---

### 9. Python-to-Unity TCP Control Channel

**Status:** ✅ Complete  
**Primary Script:** `TCPControlChannel.cs`

#### Purpose

Sends structured AI output — hand data, gestures, safety states, and warnings — from the Python backend to Unity reliably over TCP.

#### TCP Direction

```
Python AI Backend  →  TCP Port 9000  →  Unity C# Receiver
```

#### TCP Message Types

| Message Type | Direction | Description |
|---|---|---|
| `HAND_DATA` | Python → Unity | 21 2D landmarks + 21 world landmarks + gesture + wrist depth |
| `GROUND_RANSAC_DATA` | Python → Unity | RANSAC floor plane [a,b,c,d] + confidence + valid flag |
| `WARNING` | Python → Unity | Safety state change (SAFE/CAUTION/DANGER/FREEZE) |
| `STATE_CHANGE` | Python → Unity | Session state transitions |
| `FLOOR_CONFIRM` | Unity → Python | Confirmed ARCore floor plane data |

#### HAND_DATA JSON Payload Structure

```json
{
  "type": "HAND_DATA",
  "gesture": "Pinch",
  "confirmed_gesture": "Pinch",
  "confirmed": true,
  "landmarks_2d": [[x1,y1], [x2,y2], ...],
  "landmarks_world": [[x1,y1,z1], [x2,y2,z2], ...],
  "wrist_depth": 0.1694,
  "wrist_2d": [0.436, 0.517],
  "projected_floor_point": [0.25, 0.02, 1.48]
}
```

---

### 10. TOSHFA Application Scenes

**Status:** Scene 1 ✅ | Scene 2 ✅ | Scene 3 🔄 In Progress  
**Owner:** Amin Gamal (Scenes lead)

#### Overview

TOSHFA is a physiotherapy rehabilitation extension built on top of the Guardian System platform. It uses the same AR-AI pipeline to guide users through exercise routines in an immersive AR environment.

#### Scene 1 — Floor Detection and Confirmation

**Status:** ✅ Complete

| Responsibility | Details |
|---|---|
| Scan floor environment | ARCore horizontal plane detection |
| Floor plane confirmation | UI button trigger |
| Boundary auto-construction | 3D cubic box placed on confirmed floor |
| System initialization | Hand tracking + ball + warning system init |

#### Scene 2 — VR Lobby / User Intake Questionnaire

**Status:** ✅ Implemented and Operational

| Responsibility | Details |
|---|---|
| Collect user age | Hand gesture selection |
| Collect pain type | Gesture-driven multiple choice |
| Collect health history | Follow-up gesture questions |
| Transition to Scene 3 | Upon questionnaire completion |

#### Scene 3 — TOSHFA Rehabilitation Room

**Status:** 🔄 In Progress

| Planned Responsibility | Details |
|---|---|
| Exercise guidance | Animated exercise reference guide |
| Pose estimation tracking | MediaPipe Pose 33-landmark body tracking |
| Exercise recognition | Classification model: correct / incorrect form |
| Real-time feedback | Audio + visual instruction |
| Session scoring | Rep counting + score + streak |

---

## 📊 Current Development Status

| Component | Status | Notes |
|---|---|---|
| ARCore Floor Detection | ✅ Complete | Stable across diverse floor types |
| Floor Confirmation Flow | ✅ Complete | Confirm button + floor lock tested |
| Automatic 3D Boundary Box | ✅ Complete | Rise animation + world-locked |
| Boundary Locking (drift=0) | ✅ Complete | Validated across 300+ frames |
| RANSAC Floor Fitting (Python) | ✅ Complete | 95% confidence, ±2 cm accuracy |
| Monocular Depth Estimation | ✅ Complete | 30 FPS, ±3 cm accuracy |
| Pose Estimation (33 landmarks) | ✅ Complete | Disabled in MVP for FPS optimization |
| MediaPipe Hand Tracking (21 landmarks) | ✅ Complete | 96.7% accuracy, on-device |
| Pointing Gesture Classifier | ✅ Complete | 5-criteria biomechanical, 95% TP |
| EMA Smoothing + Debounce | ✅ Complete | 50% jitter reduction, 98% false-trigger reduction |
| UDP Frame Streaming (Unity → Python) | ✅ Complete | 10–16 FPS, near-zero drops |
| TCP Control Channel (Python → Unity) | ✅ Complete | Reliable HAND_DATA delivery |
| 3D Hand Skeleton On-Device | ✅ Complete | 21 joints rendered in AR world-space |
| Ball Interaction (Hover/Grab/Move) | ✅ Complete | Color feedback per state |
| Multi-Channel Warning System | ✅ Complete | SAFE/CAUTION/DANGER/FREEZE |
| Screen Flash Overlay | ✅ Complete | Red overlay on DANGER |
| Haptic Vibration Feedback | ✅ Complete | Android Vibrator API on DANGER |
| Object Detection (YOLOv11n) | 🔄 Partial | Integrated; proximity zone logic pending |
| Sensor Fusion State Machine | 🔄 Partial | Floor+depth fusion working; full states in progress |
| Scene 1 (Floor Detection) | ✅ Complete | Fully functional on Android |
| Scene 2 (VR Lobby / Questionnaire) | ✅ Complete | Operational with hand gesture input |
| Scene 3 (TOSHFA Rehabilitation Room) | 🔄 In Progress | Architecture designed, implementation ongoing |
| TOSHFA Python Pose Integration | 🔄 In Progress | Codebase ready, pipeline integration ongoing |
| Exercise Classification Model Training | 🔄 In Progress | Dataset collection and annotation in progress |
| Final APK Deployment + Signing | 🔄 In Progress | Build pipeline established, not yet final |
| Multi-Device Android Testing | ⏳ Pending | Awaiting stable Scene 3 |
| Overall Project Completion | **~80%** | As of May 2026 |

---

## 📋 MVP Feature Checklist

| Feature | Component | Status | Evidence |
|---|---|---|---|
| Real-time AR Camera Feed | Unity + ARCore | ✅ Implemented | Camera feed visible on phone screen in demo |
| Monocular Depth Estimation | ARCore Depth API / MiDaS | ✅ Implemented | Per-pixel depth maps at 30 FPS, ±3 cm accuracy in logs |
| Floor Plane Detection | ARCore Plane API + RANSAC | ✅ Implemented | Floor detected in 3–10s, 100% confidence, Y ±0.1 cm |
| Hand / Fingertip Tracking | MediaPipe Hands (21 landmarks) | ✅ Implemented | Hand skeleton overlay + fingertip dot visible in demo, 96.7% accuracy |
| Pointing Gesture Detection | Biomechanical Classifier + EMA | ✅ Implemented | 5-criteria classifier, 3-frame debounce, 95% TP rate |
| Auto Boundary Creation | ARCore Plane + Unity LineRenderer | ✅ Implemented | Green neon safety boundary auto-drawn on detected floor |
| Real-Time Phone-to-Server Pipeline | Python Backend (server.py) via TCP | ✅ Implemented | JPEG frames over TCP, JSON results returned |
| Depth Estimation AI Model | MiDaS (ONNX + CUDA) | ✅ Implemented | Runs at 10–15 Hz, depth arrays stream to Unity |
| Pose Estimation (33 Landmarks) | MediaPipe Pose (Lite model) | ✅ Implemented | 89% accuracy; disabled in MVP for FPS optimization |
| Safety HUD Boundary Overlay | Unity Canvas (Screen Overlay) | 🔄 Partial | Green neon rectangle projected from 3D floor corners |
| Object & Obstacle Detection | YOLOv11n | 🔄 Partial | Model integrated; proximity zone logic in progress |
| Sensor Fusion & Safety State Machine | Python Custom State Machine | 🔄 Partial | Floor + depth fusion working; full state logic in progress |

---

## 🛠️ Technology Stack

### Unity Side

| Technology | Version | Purpose |
|---|---|---|
| Unity Engine | 2022.3 LTS | Main AR application runtime |
| AR Foundation | v5.1.x | Cross-platform AR abstraction layer |
| ARCore XR Plugin | v5.1.x | Direct Android ARCore integration |
| C# | — | Unity scripting language |
| LineRenderer | Unity built-in | Boundary and skeleton bone rendering |
| Android SDK | — | Low-level camera and sensor access |
| Unity Android Logcat | — | Real-device debug logging |
| ARCameraManager | — | Camera frame event handling |
| Matrix4x4 (Unity) | — | Coordinate space transformations |
| ARPlaneManager | AR Foundation | Floor plane lifecycle management |

### Python AI Backend

| Technology | Version | Purpose |
|---|---|---|
| Python | 3.10 | Backend runtime |
| MediaPipe | 0.10.14 | Hand detection, gesture recognition, pose estimation |
| OpenCV | 4.8.1 | Image decoding, preprocessing, frame handling |
| NumPy | 1.24.4 | Numeric array processing |
| ONNX Runtime | Latest | MiDaS / DepthAnything model inference |
| TensorRT | Optional | GPU-accelerated inference on CUDA-capable laptops |
| asyncio | stdlib | Async server networking |
| socket (TCP/UDP) | stdlib | Network communication |
| JSON | stdlib | Structured data serialization |
| Open3D / RANSAC | — | 3D point cloud floor plane fitting |

### AI Models

| Model | Task | Runtime | Output | Status |
|---|---|---|---|---|
| ARCore Depth API | Per-pixel depth estimation | Mobile (on-device) | 320×240 normalized depth map | ✅ Complete |
| MiDaS (ONNX) | Monocular depth estimation fallback | Laptop GPU (10–15 Hz) | Per-pixel relative depth | ✅ Complete |
| DepthAnything v2 (ONNX) | Monocular depth estimation (alt) | Laptop CPU/GPU | Per-pixel relative depth | ✅ Complete |
| MediaPipe Hands | Hand keypoint tracking | Laptop CPU | 21 landmarks per hand | ✅ Complete |
| MediaPipe Pose | Full-body pose estimation | Laptop CPU | 33 landmarks | ✅ Complete (disabled in MVP) |
| YOLOv11n | Real-time object/obstacle detection | Laptop CPU/GPU | Bounding boxes + labels | 🔄 Partial |
| Exercise Classifier | Physiotherapy exercise recognition | TBD | Exercise label + form quality | 🔄 In Training |

---

## 📈 Performance Metrics

### Full System Integration — Sustained FPS

| Layer | Target FPS | Achieved FPS | Status |
|---|---|---|---|
| AR Camera Capture (Unity) | 30 FPS | 30 FPS | ✅ Met |
| UDP Frame Delivery (Unity → Python) | ≥ 15 FPS | 10–16 FPS | ✅ Met |
| Python AI Inference (MediaPipe + Depth) | 8–10 FPS | 8.5–10.3 FPS | ✅ Met |
| Unity Rendering (AR + Skeleton + Boundary) | 30 FPS | 30 FPS | ✅ Met |

### End-to-End Latency Budget

| Stage | Latency |
|---|---|
| Camera capture → UDP send | < 5 ms |
| UDP delivery (local Wi-Fi) | < 10 ms |
| MediaPipe inference | 30–50 ms |
| Depth estimation | 20–40 ms |
| TCP JSON response | < 10 ms |
| Unity render update | < 33 ms (30 FPS) |
| **Total end-to-end** | **≈ 100–150 ms** |

### System Monitoring Log Sample (Sustained 460+ Frames)

```
[STATS] Frame 199 | FPS: 9.1 | Process: 18.7ms | Cached: 67% | Floor: ✓ | Confidence: 100%
[STATS] Frame 218 | FPS: 9.0 | Process: 20.1ms | Cached: 67% | Floor: ✓ | Confidence: 100%
[STATS] Frame 237 | FPS: 9.1 | Process: 22.0ms | Cached: 67% | Floor: ✓ | Confidence: 100%
[STATS] Frame 256 | FPS: 9.1 | Process: 20.1ms | Cached: 67% | Floor: ✓ | Confidence: 100%
[STATS] Frame 275 | FPS: 9.2 | Process: 23.2ms | Cached: 67% | Floor: ✓ | Confidence: 100%
[STATS] Frame 294 | FPS: 9.1 | Process: 25.6ms | Cached: 67% | Floor: ✓ | Confidence: 100%
[STATS] Frame 313 | FPS: 9.1 | Process: 27.6ms | Cached: 67% | Floor: ✓ | Confidence: 100%
[STATS] Frame 350 | FPS: 9.0 | Process: 32.1ms | Cached: 67% | Floor: ✓ | Confidence: 100%
```

Average FPS: 9.1 | Average latency: 20–34 ms | Cache hit rate: 67% | Floor confidence: 100% across all frames.

---

## 📂 Repository & Version Control Strategy

### Multi-Branch Workflow

The team adopted a multi-branch development strategy to isolate experimental features and protect the stability of the main Python backend.

| Branch | Owner | Purpose |
|---|---|---|
| `main` | All | Stable Python backend (server, MediaPipe, communication pipeline) |
| `Ameen` | Amin Gamal | Scene development, Unity VR lobby |
| `Hamza` | Hamza Abdelmoreed | Object interaction, boundary logic |
| `Mohamed` | Mohamed Ehab Yousri | Boundary box, depth integration |
| `Yousef` | Yousef Selim | Warning system, FPS optimization |
| `hamza_boundary` | Hamza Abdelmoreed | Boundary rendering experiments |
| `hamza_points3D` | Hamza Abdelmoreed | 3D point-in-world-space experiments |

### Main Branch Policy

The `main` branch is **exclusively reserved** for the stable Python backend:

- ✅ `server.py` — main server entry point
- ✅ `hand_detector.py` — MediaPipe hand tracking
- ✅ `floor_detector.py` — RANSAC floor fitting
- ✅ `depth_estimator.py` — MiDaS depth inference
- ✅ `pose_tracker.py` — MediaPipe body pose
- ✅ `udp_handler.py` — UDP frame receiver
- ❌ Experimental or partial implementations are never committed to `main`

### Unity Cloud (Plastic SCM)

Unity-related work is maintained in **Unity Cloud (Plastic SCM)** to avoid Git conflicts in binary Unity scene files:

| Item | Managed Via |
|---|---|
| Unity C# scripts | Unity Cloud |
| Scene files (.unity) | Unity Cloud |
| Prefabs | Unity Cloud |
| Project settings / XR configuration | Unity Cloud |
| Inspector references / scene wiring | Unity Cloud |

**GitHub Repository:**  
`https://github.com/mohamed1232005/GP-VR-Guardian-System`

**Unity Cloud Project:**  
`https://cloud.unity.com/home/organizations/11270618913273/projects/c0e40f1e-f1ae-49ad-9c5b-c16b028dc078`

---

## 🧪 Testing & Quality Evidence

### Testing Methodology

Testing was performed using four parallel channels:

| Channel | Tool |
|---|---|
| Real-device runtime logging | Unity Android Logcat |
| Backend validation | Python terminal logs |
| Scene inspection | Unity Inspector |
| Physical validation | Visual AR testing on Android device |

### Test Coverage Summary

| Component | Test Type | Coverage % | Notes |
|---|---|---|---|
| Unity project setup | Integration | 90% | Build/run verified on Android |
| ARCore floor detection | Functional | 85% | Horizontal plane detection tested |
| Floor confirmation flow | Functional | 85% | Confirm button + floor lock tested |
| Boundary box placement | Functional | 80% | Static box verified post-confirmation |
| Boundary lock / stability | Regression | 85% | Drift=0.0000 validated in logs |
| AR camera feed | Functional | 80% | Camera stream tested |
| UDP frame streaming | Integration | 85% | Unity → Python validated |
| TCP control channel | Integration | 85% | Python → Unity validated |
| MediaPipe hand detection | AI module | 80% | 21 landmarks confirmed |
| Hand skeleton rendering | Functional | 75% | Visual tracking on device |
| Hand alignment calibration | Calibration | 70% | X/Y/depth calibration tested |
| Depth smoothing | Stability | 75% | Jitter reduced (σ: 8.2→4.1 px) |
| Hand-object interaction | Functional | 70% | Ball hover/grab tested |
| Movable ball logic | Functional | 75% | All state transitions tested |
| RANSAC ground validation | AI validation | 70% | Audit pipeline tested |
| RANSAC / ARCore agreement | Integration | 65% | Needs final validation pass |
| FPS monitoring | Performance | 65% | Logging tested; optimization ongoing |
| UI overlay visibility | Usability | 75% | Opacity issue identified and partially fixed |
| Android build pipeline | System | 70% | Build issues debugged |
| Version control workflow | Process | 80% | Multi-branch workflow established |
| Wall / corner proximity warning | Functional | 85% | Yellow caution zone validated |
| DANGER warning state | Functional | 85% | Red danger state validated |
| Red screen warning overlay | Usability | 80% | Full-screen red warning tested |
| Mobile vibration feedback | Haptic | 80% | Phone vibration correctly triggered |
| Safety state transition logic | Regression | 85% | SAFE → CAUTION → DANGER verified |
| Boundary warning integration | Integration | 95% | Tested with confirmed floor and static box |

### Example Test Scenarios and Expected Outputs

**Scenario 1 — Floor Confirmation and Boundary Spawn:**
```
Input:  User scans room → taps Confirm Floor
Output: Floor Y stored → 3D boundary box appears

Expected Logs:
[FLOOR_CONFIRMED]
[P12_BOUNDARY] STATIC_CHECK drift=0.0000
```

**Scenario 2 — Hand Detection and Skeleton Rendering:**
```
Input:  User shows hand to camera
Output: Python detects 21 landmarks → Unity renders hand skeleton

Expected Logs:
hand_detected=True
landmarks_world=21
[P8_UNITY_HAND] parsed confirmed=True
```

**Scenario 3 — Ball Interaction (Hover → Grab → Move):**
```
Input:  User moves index fingertip near ball → pinches
Output: Ball changes color White → Yellow (hover) → Magenta (grab → move)

Expected Logs:
[P11_BALL] HOVER
[P11_BALL] GRAB
[P11_BALL] MOVE
```

**Scenario 4 — RANSAC Floor Validation:**
```
Input:  Unity sends confirmed ARCore floor to Python
Output: RANSAC validates floor Y; boundary remains static

Expected Logs:
[P13_RANSAC] valid=True
[P13_FLOOR_FUSION] diff<0.05
[P12_BOUNDARY] drift=0.0000
```

**Scenario 5 — CAUTION State Trigger:**
```
Input:  User moves near boundary wall (within caution range)
Output: Boundary color changes to yellow; no vibration

Expected Logs:
[P14_WARNING] state=CAUTION
[P14_WARNING_COLOR] boundary_color=YELLOW
[P14_DISTANCE] near_boundary=True
```

**Scenario 6 — DANGER State Trigger:**
```
Input:  User moves critically close to boundary wall
Output: Boundary RED + red screen overlay + phone vibration

Expected Logs:
[P14_WARNING] state=DANGER
[P14_WARNING_COLOR] boundary_color=RED
[P14_SCREEN_ALERT] red_overlay=True
[P14_HAPTIC] vibration=True
```

**Scenario 7 — State Reset (DANGER → SAFE):**
```
Input:  User steps away from boundary
Output: All warnings cleared; boundary returns to green

Expected Logs:
[P14_WARNING] state=SAFE
[P14_SCREEN_ALERT] red_overlay=False
[P14_HAPTIC] vibration=False
[P14_WARNING_RESET] success=True
```

---

## 🐞 Bugs Identified & Fixed

| Bug ID | Description | Severity | Status | Fix Applied |
|---|---|---|---|---|
| BUG-01 | RANSAC JSON Parse Error — Python sent `str(dict)` instead of valid JSON; Unity threw parse exception | High | ✅ Fixed | Replaced `str(result_dict).encode()` with `json.dumps(result_dict).encode()` |
| BUG-02 | RANSAC Floor Y Mismatch — RANSAC floor Y differed by 2+ m from ARCore confirmed floor Y due to local-space boundary points | High | ✅ Fixed | In `ARPlaneDataSender.cs`, converted each boundary point to world coordinates using `plane.transform.TransformPoint()` |
| BUG-03 | Boundary Box Drifting — 3D boundary shifted when user approached it instead of remaining static after confirmation | High | ✅ Fixed | Locked boundary center and size immediately after floor confirmation in `BoundaryBoxRenderer.cs`; all subsequent position updates rejected |
| BUG-04 | Ball Difficult to Grab — Strict pinch requirements and unstable hand detection caused frequent `NO_VALID_HAND` states | Medium | ✅ Fixed | Relaxed hover/grab distance thresholds; increased ball scale; added 0.3s grace period before dropping hand interaction on lost frames |
| BUG-05 | Hand Tracking Instability — Skeleton dropped frequently mid-session due to lighting, hand speed, or hand leaving camera view | Medium | ✅ Fixed | Added short grace period in `HandBallMover.cs` retaining last valid hand state; established testing guidelines for lighting and hand distance |
| BUG-06 | `CreatePrimitive` Runtime Spam — Scripts called `GameObject.CreatePrimitive()` inside `Update()`, causing memory pressure and frame drops | Medium | ✅ Fixed | Moved all `CreatePrimitive` calls to `Start()`; added null checks to ensure objects are created once and reused |
| BUG-07 | `FloorDetectionController` Log Spam — Script logged every single frame, making Logcat unreadable | Low | ✅ Fixed | Replaced per-frame logs with throttled 1-second interval logging; kept only meaningful event logs (`FLOOR_CONFIRMED`, `BOUNDARY_LOCKED`) |
| BUG-08 | Black Screen on App Start — After rebuild, AR camera took 10+ seconds to initialize or showed all-black if previous session not fully closed | Medium | ✅ Fixed | Added `OnApplicationPause()` in `UDPFrameSender.cs` to stop frame sending on pause; established clean restart protocol |
| BUG-09 | Hand Skeleton Not Appearing on Screen — Python confirmed hand detection but nothing rendered on device; `OnHandData` Unity event was disconnected after script replacement | High | ✅ Fixed | Rewired all three listeners in Inspector; added auto-wiring in `Start()` to prevent recurrence; confirmed `listeners=3` in Logcat |
| BUG-10 | Gesture Field Sending Python `None` — Gesture field in JSON payload was Python `None`, causing Unity JSON deserializer to receive null and hide skeleton | High | ✅ Fixed | Added `_safe_str()` guards throughout `hand_detector.py`; enforced explicit string conversion at every gesture boundary before JSON serialization |
| BUG-11 | Duplicate `MSG_HAND_DATA` Switch Case — `case MSG_HAND_DATA:` appeared twice in `TCPControlChannel.cs`, causing `CS0152` compile error | Medium | ✅ Fixed | Removed duplicate case; kept single `MSG_HAND_DATA` handler with full debug logging |
| BUG-12 | Old State Machine Blocking Hand Tracking — `session["state"]` forced to `PLACING_POINTS` after floor receipt; hand tracking skipped in `GUARDIAN_READY` | High | ✅ Fixed | Changed state assignment to `GUARDIAN_READY` after floor confirmation; updated detector guard to allow `GUARDIAN_READY` |
| BUG-13 | `Closed_Fist` Falsely Triggering Boundary Finalize — In new system, Closed_Fist triggered old manual boundary finalize with 0 points | Medium | ✅ Fixed | Separated Closed_Fist behavior: `PLACING_POINTS` → finalize polygon; `GUARDIAN_READY` → interaction/grab mode |

---

## 🚧 Work in Progress & Known Issues

### 1. Scene 3 — TOSHFA Rehabilitation Room

**Status:** Architecture designed; Unity scene construction in progress.

The rehabilitation scene depends on the stability of all prior modules. Exercise recognition requires:
- Stable MediaPipe Pose 33-landmark output
- Exercise sequence annotation dataset
- Trained classification model (currently in data pipeline stage)
- Real-time feedback integration with scene UI

Any instability in hand tracking or depth estimation directly blocks Scene 3 progress.

### 2. Exercise Classification Model — Data Pipeline

**Status:** Dataset collection and annotation in progress.

| Stage | Status |
|---|---|
| TOSHFA survey data collection | ✅ Done (prior semester) |
| JSON survey data cleaning | ✅ Done |
| Exercise sequence annotation | 🔄 In Progress |
| Model architecture selection | 🔄 In Progress |
| Training | ⏳ Pending dataset completion |
| Validation | ⏳ Pending training |

### 3. Object & Obstacle Detection (YOLOv11n)

**Status:** Model integrated into pipeline; proximity zone logic still in progress.

YOLOv11n is loaded and produces bounding boxes, but the proximity zone evaluation logic (determining how close a detected obstacle is to the user in real-world 3D space) is not yet complete.

### 4. Sensor Fusion Full State Machine

**Status:** Partial — floor + depth fusion working; full SAFE/CAUTION/DANGER/FREEZE logic for sensor-fused input in progress.

The current warning system operates primarily on boundary proximity from Unity. Full sensor fusion (fusing depth estimation distances, YOLO obstacle distances, and IMU data into a unified safety state) is still being completed.

### 5. FPS Degradation Under Full Multi-Model Load

**Status:** Mitigated; requires final optimization pass.

Running hand tracking + depth estimation + YOLO + safety logic simultaneously pushes inference latency above the real-time threshold on mid-range Android hardware.

Current mitigation strategies:
- Frame caching (every 3rd frame processed for depth; intermediate frames served from cache)
- Pose tracker disabled in MVP build
- Adaptive downsampling before model inference

Thermal throttling on mid-range Android devices remains a risk during extended test sessions.

### 6. Final APK Deployment

**Status:** Build pipeline established; final signing and distribution testing not yet begun.

Pending stable Scene 3 integration before packaging for multi-device distribution testing.

### 7. Legacy Manual Boundary System Cleanup

**Status:** Compatibility code still present; scheduled for removal after automatic boundary is confirmed 100% stable.

Remaining legacy components:
- `POINT_ADDED` TCP message handler
- `boundary_pts` list in Python session
- Manual finalize logic (retained for backward compatibility only)

---

## 📌 Remaining Tasks & Planning

| Task | Owner | Priority | Deadline | Status |
|---|---|---|---|---|
| Complete Scene 3 Rehabilitation Room Unity build | Amin Gamal | High | Week 14 | 🔄 In Progress |
| Integrate TOSHFA Python pose estimation pipeline with Guardian System | Hamza Abdelmoreed | High | Week 14 | 🔄 In Progress |
| Complete exercise dataset collection, preprocessing, and annotation | Mohamed Ehab Yousri | High | Week 15 | 🔄 In Progress |
| Train and validate exercise classification model | Mohamed Ehab Yousri | High | Week 15 | ⏳ Pending dataset |
| Final APK packaging, signing, and distribution testing | Yousef Selim | Medium | Week 15 | 🔄 In Progress |
| Multi-device Android testing (low-end + mid-range + high-end) | Full Team | Medium | Week 15 | ⏳ Pending |
| Complete YOLOv11n proximity zone logic | Full Team | High | Week 14 | 🔄 In Progress |
| Complete full sensor fusion state machine | Full Team | High | Week 14 | 🔄 In Progress |
| Remove legacy manual boundary system code | Full Team | Medium | Week 15 | ⏳ Pending |
| Final FPS optimization pass under full multi-model load | Yousef + Mohamed | High | Week 14 | 🔄 In Progress |
| Final documentation and report polishing | Full Team | High | Week 15 | 🔄 In Progress |

---

## ▶️ How to Run the Project

### Prerequisites

| Requirement | Details |
|---|---|
| Android Device | ARCore-certified Android smartphone (mid-range or higher) |
| Unity | Unity 2022.3 LTS with Android Build Support module |
| Python | Python 3.10+ |
| Network | Phone and laptop on the same local Wi-Fi network |
| CUDA (optional) | For TensorRT-accelerated depth inference |

---

### Step 1 — Clone the Python Backend

```bash
git clone https://github.com/mohamed1232005/GP-VR-Guardian-System.git
cd GP-VR-Guardian-System
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the server:

```bash
python server.py
```

Expected terminal output:

```
[*] CV worker PID: XXXXX
[*] UDP listening on 0.0.0.0:9001
[*] TCP listening on 0.0.0.0:9000
[*] READY — waiting for mobile connection...
```

Keep this terminal open for the entire session.

---

### Step 2 — Pull the Unity Project from Unity Cloud

```
Unity Cloud Project:
https://cloud.unity.com/home/organizations/11270618913273/projects/c0e40f1e-f1ae-49ad-9c5b-c16b028dc078
```

1. Open **Unity Hub** and sign in with your Unity account.
2. Go to **Projects → Add → Clone from Unity Cloud**.
3. Paste the Unity Cloud project link above.
4. Wait for scene synchronization to complete.
5. Open **SampleScene** from the Project panel.

---

### Step 3 — Configure Server IP in Unity

1. In the Unity Hierarchy, click the **GuardianManager** (or **Network Manager**) object.
2. In the **Inspector**, find the **Server IP** field under NETWORK settings.
3. Update the IP to your laptop's local IP address (e.g., `192.168.1.10`).
4. Confirm the **TCP Port** is `9000` and **UDP Port** is `9001`.

> **Important:** The Android phone and the laptop must be connected to the **same Wi-Fi network**.

---

### Step 4 — Verify Unity Scene Hierarchy

The expected scene hierarchy is:

```
SampleScene
├── AR Session
├── AR Session Origin
│   └── Camera Offset
│       └── AR Camera
│           └── UDPFrameSender.cs
├── Network Manager
│   ├── TCPControlChannel.cs
│   └── UDPFrameSender.cs
├── Floor Manager
│   └── FloorDetectionController.cs
├── Guardian Manager
│   └── BoundaryBoxRenderer.cs
├── Hand Skeleton           ← Must be a root 3D GameObject, NOT inside VR Canvas
│   └── HandSkeletonRenderer.cs
├── VR Canvas               ← UI Canvas for overlays and HUD
│   ├── WarningOverlay
│   └── HUD Text
├── Ball Manager
│   ├── BallController.cs
│   └── HandBallMover.cs
└── EventSystem
```

> ⚠️ **Critical:** `Hand Skeleton` must be a **root 3D GameObject** in the scene — not a child of `VR Canvas`. The 3D world-space skeleton does not render correctly when placed under a Canvas.

---

### Step 5 — Build and Deploy to Android

1. Connect your Android phone via USB cable.
2. Enable **USB Debugging** in Developer Options on the phone.
3. In Unity: **File → Build Settings → Android → Switch Platform**.
4. Verify the phone appears in the **Run Device** dropdown.
5. Click **Build and Run**.
6. Unity will compile the APK and automatically install it on the phone.

---

### Step 6 — Runtime User Flow

```
1. Open app on Android phone
2. Scan room floor (move phone slowly around the floor)
3. Wait for floor detection: HUD shows "Floor: ✓"
4. Press Confirm Floor button
5. 3D boundary box appears with rise animation
6. Virtual ball spawns inside boundary
7. Show hand to camera
8. Hand skeleton appears with 21 joints
9. Use Pinch gesture to grab and move the ball
10. Approach boundary walls to trigger CAUTION (yellow) and DANGER (red + vibration) states
11. Step away from boundary to reset to SAFE (green)
```

---

### Important Runtime Log Reference

**Python Backend — Healthy State:**
```
[*] READY — waiting for mobile connection...
[NETWORK] Connected: ('192.168.x.x', PORT)
[SYSTEM] Connected – running
[AI_UDP] received frame XXXX – seq=XXXX bytes=XXXXX decoded shape=(480,640,3) fps=15.x
[HAND] process_frame state=GUARDIAN_READY
[HAND] gesture=Pinch confirmed=True landmarks=21
```

**Unity Android Logcat — Healthy State:**
```
[FLOOR] ConfirmFloor called
[FLOOR] FLOOR_CONFIRM sent to Python
[P12_BOUNDARY] boundary locked
[P12_BOUNDARY] drift=0.0000
[TCP] Dispatching HAND_DATA
[P8_UNITY_HAND] parsed confirmed=True landmarks_2d=21 landmarks_world=21 wrist_depth=0.16
[P9_HAND_VIS] rendered joints=21
[P14_WARNING] state=SAFE
```

---

## 📁 Project File Structure

### Python Backend

```
GP-VR-Guardian-System/
├── server.py               # Main entry point — TCP/UDP server, session management
├── hand_detector.py        # MediaPipe GestureRecognizer + 21-landmark hand tracking
├── hand_tracker.py         # EMA smoothing, debounce state machine, landmark extraction
├── pose_tracker.py         # MediaPipe Pose 33-landmark body tracking
├── depth_estimator.py      # MiDaS / DepthAnything ONNX depth inference
├── floor_detector.py       # RANSAC-based floor plane fitting from depth data
├── udp_handler.py          # UDP frame reception, JPEG decode, frame queue
├── requirements.txt        # Python dependencies
└── models/
    ├── midas_v21_small.onnx
    └── depth_anything_v2.onnx
```

### Unity Project (Unity Cloud — C# Scripts)

```
Assets/
├── Scripts/
│   ├── Core/
│   │   ├── FloorDetectionController.cs    # ARCore plane detection + floor confirmation
│   │   ├── BoundaryBoxRenderer.cs         # 3D boundary construction, locking, LineRenderer
│   │   ├── TCPControlChannel.cs           # TCP receiver for Python JSON messages
│   │   └── UDPFrameSender.cs              # UDP camera frame sender to Python
│   ├── Hand/
│   │   ├── HandSkeletonRenderer.cs        # 21-joint 3D hand skeleton rendering
│   │   ├── HandDataForBall.cs             # Hand data relay to ball interaction system
│   │   └── HandBallMover.cs              # Proximity hover + grab + move logic
│   ├── Ball/
│   │   └── BallController.cs             # Ball spawn, state machine, color feedback
│   ├── Warning/
│   │   ├── WarningSystem.cs              # SAFE/CAUTION/DANGER/FREEZE state + feedback
│   │   └── ProximityChecker.cs           # Distance-to-boundary calculation
│   └── Scenes/
│       ├── Scene1Controller.cs           # Floor detection + boundary scene management
│       ├── Scene2Controller.cs           # VR lobby + questionnaire flow
│       └── Scene3Controller.cs           # [In Progress] Rehabilitation room
├── Scenes/
│   ├── Scene1_FloorDetection.unity
│   ├── Scene2_VRLobby.unity
│   └── Scene3_RehabilitationRoom.unity   # [In Progress]
└── Prefabs/
    ├── BoundaryBox.prefab
    ├── HandSkeleton.prefab
    └── InteractiveBall.prefab
```

---

## 👥 Team Contributions

### Mohamed Ehab Mohamed Yousri — 202201236 | DSAI

**Contribution: 25%**

| Area | Details |
|---|---|
| System Architecture | Co-led system architecture design, requirements analysis, and module design logic |
| 3D Boundary Box | Led design and implementation of automatic 3D cubic boundary construction and world-locking upon floor confirmation |
| AR Boundary Visualization | Implemented AR boundary rendering integrated with live AR camera feed |
| Monocular Depth Estimation | Implemented ARCore Depth API integration, adaptive resolution scaling, depth caching strategy, and confidence scoring |
| Hand Skeleton Integration | Implemented 3D hand skeleton overlay on live AR camera feed |
| FPS Optimization | Co-led FPS profiling and performance tuning across Depth, Pose, and Hand Tracking modules |
| Data Pipeline | Helped structure TOSHFA JSON survey responses into labeled dataset for exercise classification model training |
| Integration Trials | Developed trial implementations connecting all three AI modules on the Python backend |
| Research | Co-authored research paper on pose estimation for lower back pain ("Toshfa") published on arXiv |

### Hamza Abdelmoreed — 202201508 | DSAI

**Contribution: 25%**

| Area | Details |
|---|---|
| System Architecture | Co-led system architecture design, requirements analysis, and technical documentation |
| Modules Integration (Lead) | Led end-to-end integration of Monocular Depth, Pose, Ground Level, and Hand Tracking into unified pipeline |
| Ground Level Estimation | Implemented and refined ARCore horizontal plane detection and RANSAC-based floor fitting |
| Boundary Drawing & Detection | Improved boundary drawing algorithm and area detection logic for higher accuracy |
| Object Detection & Interaction | Implemented interactive ball tracking — hover detection, pinch grab, movement, release, and color feedback |
| Scene Development (Co-lead) | Co-led Scene 1 and Scene 2 development and integration |
| Python-Unity Pipeline | Built and debugged the full Python-to-Unity data flow |
| Data Pipeline | Cleaned and formatted TOSHFA survey data for model training input |
| Research | Co-authored research paper on pose estimation for lower back pain ("Toshfa") published on arXiv |

### Amin Gamal — 202202219 | DSAI

**Contribution: 25%**

| Area | Details |
|---|---|
| System Architecture | Co-led system architecture design, requirements analysis, and technical documentation |
| Scenes Development (Lead) | Led design and implementation of Scene 1 (Floor Detection) and Scene 2 (VR Lobby + Questionnaire) |
| VR Lobby | Built pre-phase application scenes including hand-gesture-driven intake questionnaire |
| Android Build Environment | Configured Unity project structure, ARCore XR Plugin, and AR Foundation dependencies for Android |
| Integration (Co-lead) | Co-led end-to-end integration of all perception and safety modules |
| FPS Optimization | Contributed to FPS optimization across active modules |
| Data Pipeline | Preprocessed and annotated TOSHFA survey responses, mapped to exercise categories |
| Research | Co-authored research paper on pose estimation for lower back pain ("Toshfa") published on arXiv |

### Yousef Selim — 202201255 | DSAI

**Contribution: 25%**

| Area | Details |
|---|---|
| System Architecture | Co-led system architecture design, requirements analysis, and technical documentation |
| Multi-Channel Warning System | Implemented full SAFE/CAUTION/DANGER/FREEZE safety state machine with screen color changes, visual flash overlays, and device vibration |
| FPS Optimization (Co-lead) | Co-led FPS profiling, frame-rate stabilization, and optimization of active modules |
| Pose Estimation Trial | Developed trial implementation of real-time pose estimation using MediaPipe and RTMPose-s with GPU |
| Integration Trials | Developed trial implementations connecting Python and Unity; validated bidirectional data communication |
| Evaluation Strategy | Defined model evaluation strategy, metrics, and validation protocol for exercise classification model |
| Research | Co-authored research paper on pose estimation for lower back pain ("Toshfa") published on arXiv |

---

## 👩‍🏫 Supervisor

| Field | Details |
|---|---|
| Name | Dr. Mayada Mansour Ali Hadhoud |
| Institution | Zewail City of Science, Technology and Innovation |
| School | CSAI School |
| Courses | CSAI 498 / CSAI 499 |

---

## 🎓 Academic Context

| Field | Details |
|---|---|
| Institution | Zewail City of Science, Technology and Innovation |
| School | CSAI School |
| Program | Data Science and Artificial Intelligence (DSAI) |
| Courses | CSAI 498 / CSAI 499 (Graduation Project I & II) |
| Semester | Spring 2026 |
| Team Number | Team 25 |
| Report 1 Submission | March 7, 2026 |
| Report 2 Submission | May 2, 2026 |
| GitHub Repository | https://github.com/mohamed1232005/GP-VR-Guardian-System |
| Unity Cloud | [https://cloud.unity.com/home/organizations/11270618913273/projects/c0e40f1e-f1ae-49ad-9c5b-c16b028dc078](https://cloud.unity.com/home/organizations/11270618913273/projects/3af6106f-603f-411a-9087-d805c64f2b72/plastic-scm/organizations/11270618913273/repositories/MVP-Guardian-Unity%2FScene-Rendering-Updated/branch/%2Fmain/tree/) |
| Demo Video |[ https://drive.google.com/drive/folders/14uRaXD2cKZxU0FoLaBoZ7k0BDWAV8DPG](https://drive.google.com/file/d/1qWls0T4skueb_WwdptPVJGJcsSJ5ZOkz/view?usp=sharing) |
| Presentation | https://canva.link/lnd040bvpzaufzr |

---

<div align="center">

**© 2026 Team 25 — Zewail City of Science, Technology and Innovation**  
*Guardian System for Affordable Smartphone-Based Virtual Reality*  
*CSAI 498 / CSAI 499 Graduation Project*

</div>
