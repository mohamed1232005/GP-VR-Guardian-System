<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Unity-2022.3+-black?logo=unity&logoColor=white" />
  <img src="https://img.shields.io/badge/Android-API%2030+-brightgreen?logo=android&logoColor=white" />
  <img src="https://img.shields.io/badge/Depth%20Anything-V2-orange" />
  <img src="https://img.shields.io/badge/SegFormer-B0-purple" />
  <img src="https://img.shields.io/badge/MediaPipe-HandLandmarker-red" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

<h1 align="center">🛡️ AI Guardian — VR Safety Boundary System</h1>

<p align="center">
  <strong>Smartphone-Based AR/VR Safety-Boundary Generation Using Monocular AI Vision</strong><br/>
  Graduation Project · Zewail City of Science and Technology · CSAI · June 2026
</p>

---

## 📖 Table of Contents

1. [What Is This?](#what-is-this)
2. [Why "AI, Not ARCore"?](#why-ai-not-arcore)
3. [System Architecture](#system-architecture)
4. [AI/ML Stack](#aiml-stack)
5. [Project Structure](#project-structure)
6. [Pipeline Deep Dive](#pipeline-deep-dive)
7. [Networking Protocol](#networking-protocol)
8. [Lock Logic & Safety Gates](#lock-logic--safety-gates)
9. [Hand Tracking](#hand-tracking)
10. [Post-Lock Optimization](#post-lock-optimization)
11. [Session Metrics & Artifacts](#session-metrics--artifacts)
12. [Configuration Reference](#configuration-reference)
13. [Getting Started](#getting-started)
14. [Performance Results](#performance-results)
15. [Limitations & Future Work](#limitations--future-work)
16. [References](#references)

---

## What Is This?

**AI Guardian** turns a commodity Android smartphone into a low-cost mixed-reality safety-boundary generator — the same kind of "play area guardian" found in dedicated VR headsets — using only monocular AI vision.

A Unity client streams compressed RGB frames from the phone's camera over UDP to a Python AI server on a PC. For every incoming frame the server:

1. Estimates a metric depth map (Depth Anything V2)
2. Semantically identifies floor pixels (SegFormer-B0 / geometric fallback)
3. Back-projects floor pixels into 3D camera space
4. Fits a horizontal floor plane via custom NumPy RANSAC
5. Accumulates trusted floor cells in a world-space grid over time
6. Locks the largest stable, safe rectangle and streams it to Unity
7. Tracks hands with MediaPipe for ray-cast interaction

The locked boundary is rendered as a green cube in Unity AR space; as the user approaches a wall, that face highlights and a vibration warning triggers.

> **Representative result (Phase 9.1, Samsung SM-A346E):** 242 s session, 567 frames, locked at frame 516 — **zero errors, zero warnings**.

---

## Why "AI, Not ARCore"?

This is the central contribution of the project. The table below shows the exact division of responsibility:

| Capability | Source |
|---|---|
| Floor detection | Depth Anything V2 + SegFormer + NumPy RANSAC **(this project)** |
| Metric depth from a single camera | Depth Anything V2 **(this project)** |
| Safety boundary geometry | Custom plane-fit + accumulator **(this project)** |
| Hand tracking & interaction ray | MediaPipe Tasks HandLandmarker **(this project)** |
| Camera image + intrinsics | ARCore (used as a sensor only) |
| Camera-to-world pose | ARCore (used as a sensor only) |
| Post-lock spatial anchor | ARCore (used after lock to freeze the cube) |

ARCore is retained only as a camera, a pose source, and a post-lock anchor. All spatial perception and the safety boundary itself are produced by the AI pipeline.

---

## System Architecture

```
┌──────────────────────────────────────┐     UDP 9001 (JPEG + JSON metadata)
│         Smartphone / Unity           │ ─────────────────────────────────────►
│  ARCameraFrameProvider               │                                        │
│    → resize 320×240                  │                                        ▼
│    → JPEG encode                     │         ┌──────────────────────────────────────┐
│    → attach: intrinsics, pose,       │         │         Python AI Server             │
│              frame_id, rotation      │         │                                      │
└──────────────────────────────────────┘         │  UDPReceiver (port 9001)             │
         ▲                                       │    ↓                                 │
         │  TCP 9002 (JSON guardian /            │  AIPipelineWorker (background thread)│
         │  hand results)                        │    ↓                                 │
         └───────────────────────────────────────│  Depth Anything V2                  │
                                                 │    ↓                                 │
                                                 │  SegFormer / Geo fallback            │
                                                 │    ↓                                 │
                                                 │  Back-projection → RANSAC            │
                                                 │    ↓                                 │
                                                 │  FloorAccumulator (world grid)       │
                                                 │    ↓                                 │
                                                 │  Lock gate → TCPServer (port 9002)   │
                                                 │    ↓                                 │
                                                 │  MediaPipe HandLandmarker            │
                                                 └──────────────────────────────────────┘
```

**Latest-frame policy.** The pipeline keeps only the most recent pending frame and drops older ones. This prevents queue build-up when the heavy models (DepthAnything + SegFormer, ~350 ms/frame) cannot keep up with the 9 FPS UDP stream.

---

## AI/ML Stack

| Component | Library / Model | Role |
|---|---|---|
| Depth estimation | `Depth-Anything-V2-Metric-Indoor-Small-hf` via HuggingFace Transformers | Converts RGB → metric depth map (metres) |
| Floor segmentation | `nvidia/segformer-b0-finetuned-ade-512-512` via HuggingFace Transformers | Identifies floor pixels in ADE20K class space |
| Hand tracking | MediaPipe Tasks `HandLandmarker` | 21-landmark hand skeleton + handedness + confidence |
| Plane fitting | Custom NumPy RANSAC (no SciPy, no Open3D, no scikit-learn) | Robust horizontal plane from 3D floor points |
| Deep learning runtime | PyTorch | Model inference backend |
| Image processing | OpenCV, PIL | Frame decode, colour conversion, resize |
| Numerical computing | NumPy | All geometry and RANSAC arithmetic |

> **Explicitly not used:** SciPy, Open3D, scikit-learn, ONNX/TensorRT, any WebSocket library.

---

## Project Structure

```
AI-GUARDIAN-SERVER/
│
├── server.py                   # Main entry point — starts UDP, TCP, pipeline
├── config.py                   # All tuneable constants (one file)
├── requirements.txt
│
├── models/
│   ├── ai_pipeline_worker.py   # Core background worker thread (Phase 7A–14)
│   ├── depth_model.py          # Depth Anything V2 wrapper
│   ├── depth_worker.py         # Depth inference helper
│   ├── floor_segmenter.py      # SegFormer floor segmentation
│   ├── dummy_guardian.py       # Offline/dummy boundary for testing
│   ├── hand_tracker.py         # MediaPipe HandLandmarker wrapper
│   └── hand_landmarker.task    # MediaPipe model asset
│
├── geometry/
│   ├── backproject.py          # Pinhole back-projection (pixel → 3D)
│   ├── boundary_generator.py   # Rectangular boundary from RANSAC inliers
│   ├── floor_accumulator.py    # World-space multi-frame occupancy grid
│   ├── floor_from_depth.py     # Geometric floor detection (no segmentation)
│   ├── plane_fit.py            # Custom RANSAC + LSQ plane fit
│   └── rotation_utils.py       # Frame + intrinsics rotation helpers
│
├── networking/
│   ├── protocol.py             # Message serialisation
│   ├── tcp_server.py           # TCP result server (port 9002)
│   ├── udp_receiver.py         # UDP frame receiver (port 9001)
│   └── unity_lock_listener.py  # Unity→Python lock-ACK listener (port 9991)
│
├── toshfa/                     # Downstream rehabilitation app integration
│   ├── pose_landmarker_lite.task
│   └── toshfa_pose_session.py
│
├── artifacts/                  # Auto-generated per session
│   ├── session_metrics.json
│   ├── session_metrics.csv
│   └── session_summary.txt
│
└── debug/                      # Optional debug images and geometry probes
```

---

## Pipeline Deep Dive

### Phase 1 — Depth Estimation

`Depth-Anything-V2-Metric-Indoor-Small-hf` is loaded via `AutoImageProcessor` and `AutoModelForDepthEstimation`. The BGR frame is converted to RGB, passed to the processor, and the predicted depth tensor is bicubic-upsampled to the original resolution and clipped to `[MIN_DEPTH_M, MAX_DEPTH_M]` (default 0.1–10 m).

### Phase 2 — Floor Segmentation (Hybrid)

The pipeline runs in a **geometry-first, hybrid** mode controlled by `FLOOR_DETECTION_MODE`:

- **Geometric path** (`floor_from_depth.py`): fits a plane to the lower image ROI using depth alone. Reliable on glossy or textureless floors where SegFormer struggles.
- **Semantic path** (SegFormer-B0 on ADE20K): identifies the ADE20K floor class dynamically at startup. Used as supporting evidence, not as authority.
- **Hybrid fusion**: the geometric plane defines the band of accepted heights; semantic scores modulate the confidence. If floor pixel ratio falls below threshold, the frame is rejected with `floor_too_small`.

### Phase 3 — Back-Projection

Each accepted floor pixel `(u, v)` is lifted to 3D using the standard pinhole model and the (rotation-adjusted) camera intrinsics `(fx, fy, cx, cy)`:

```
X = (u - cx) * D(u,v) / fx
Y = (v - cy) * D(u,v) / fy
Z = D(u,v)
```

Points are in the OpenCV camera convention (X right, Y down, Z forward). A Y-axis flip converts to Unity camera convention. The camera-to-world matrix (sent by Unity in column-major order) then rotates these into world space.

### Phase 4 — RANSAC Floor-Plane Fitting

Custom NumPy RANSAC (no external solver):

1. Sample 3 back-projected points → candidate plane normal `n̂` and offset `d`
2. Count inliers: points with point-to-plane distance ≤ `RANSAC_INLIER_THRESHOLD_M` (default 0.05 m)
3. Best plane after `RANSAC_ITERATIONS` (default 200) iterations → refine with LSQ over inliers
4. Gate: `plane_min_inliers`, `plane_min_inlier_ratio`, `plane_max_rmse`, `plane_min_horizontal_score`

The horizontal score is the dot product of the fitted normal with the gravity vector in world space. Walls and ceiling score near zero and are rejected.

### Phase 5 — Boundary Generation

Inlier points are projected onto the floor plane, outliers removed by an interquartile filter, and the visible extent is estimated. Three distinct representations are maintained:

| Representation | Description |
|---|---|
| **Per-frame smart cube** | Tight fit to visible inliers this frame |
| **Preview rectangle** | Accumulated safe zone (grows during scanning, shown yellow) |
| **Locked rectangle** | Final frozen boundary (shown green), never from a single frame |

### Phase 6 — Multi-Frame Floor Accumulation

`FloorAccumulator` maintains a world-space occupancy grid of 5 cm cells. Every lock-eligible frame contributes its floor points. The accumulator is ready to lock only when:

- `min_frames` trusted frames have been accumulated
- `min_cells` grid cells are occupied
- `min_area_m2` total area is reached
- The boundary is **stable**: centre, yaw, floor-Y, and size all stay within tolerance for `stability_window` consecutive frames

Behavioural rule: **move = preview grows; stop = lock.** While the user sweeps, the preview keeps expanding. When the camera is still and the area is large and stable, the system locks automatically.

---

## Networking Protocol

### Wire Format (UDP, port 9001)

Each packet: `[4-byte LE int: JSON length][UTF-8 JSON metadata][JPEG bytes]`

Key metadata fields:

| Field | Type | Description |
|---|---|---|
| `frame_id` | int | Monotonic frame counter |
| `fx, fy, cx, cy` | float | Camera intrinsics (pixels) |
| `camera_to_world` | float[16] | Column-major 4×4 camera-to-world matrix |
| `rotation_degrees` | int | Frame rotation to apply before inference (0/90/180/270) |
| `timestamp` | float | Unity-side capture time |

### Result Format (TCP, port 9002)

JSON messages pushed by `TCPServer`. Two message types:

**Guardian boundary (`AI_GUARDIAN_DATA`):**
```json
{
  "type": "AI_GUARDIAN_DATA",
  "state": "locked",
  "frame_id": 516,
  "corners_world": [[x0,y0,z0], [x1,y1,z1], [x2,y2,z2], [x3,y3,z3]],
  "floor_y_world": -0.82,
  "width_m": 2.40,
  "depth_m": 2.40,
  "confidence": 0.97,
  "horizontal_score": 0.994
}
```

**Hand tracking (`AI_HAND_DATA`):**
```json
{
  "type": "AI_HAND_DATA",
  "frame_id": 517,
  "landmarks_2d": [[u,v], ...],
  "landmarks_3d_cam": [[x,y,z], ...],
  "ray_origin_cam": [x,y,z],
  "ray_direction_cam": [x,y,z],
  "handedness": "Right",
  "hand_confidence": 0.98
}
```

### Unity Lock ACK (TCP, port 9991)

A lightweight reverse channel. When Unity accepts a lock it sends a JSON ACK back to the Python server so that `export_session_artifacts()` can record the `lock_frame_id` that Unity actually used (rather than only the frame Python emitted).

---

## Lock Logic & Safety Gates

A lock is permitted only when **three independent gates** agree simultaneously:

### Gate 1 — Per-frame quality (`lock_allowed`)

| Parameter | Default | Meaning |
|---|---|---|
| `LOCK_MIN_HORIZONTAL_SCORE` | 0.90 | Plane normal must align tightly with gravity |
| `LOCK_MIN_INLIER_RATIO` | 0.75 | ≥75% of floor pixels must be RANSAC inliers |
| `LOCK_MAX_RMSE` | 0.03 m | Plane fit residual must be very tight |
| `LOCK_MIN_CONFIDENCE` | 0.80 | Combined segmentation confidence |
| `LOCK_MIN_FLOOR_RATIO` | 0.05 | Minimum fraction of image that is floor |

### Gate 2 — Accumulator readiness

`ready_to_lock()` checks the stability window; `can_lock_now()` is used for manual user-triggered lock requests.

### Gate 3 — Coordinate sanity (`_coord_sanity_ok`)

- The four boundary corner heights must lie within `UNITY_LOCK_MAX_Y_SPREAD_M` (0.18 m) of each other — a tilted "floor" is rejected.
- The floor-Y must be a plausible distance below the camera.
- The plane normal in world space must point upward.

Unity repeats an independent coordinate-sanity check (corner-height spread ≤ 0.20 m, side-length plausibility) before rendering, providing defence-in-depth.

When any gate fails, the system keeps scanning — **safety over forced locking**.

---

## Hand Tracking

`HandTracker` wraps the MediaPipe Tasks `HandLandmarker`:

- Returns 21 landmarks (wrist + 4 fingers × 4 joints + fingertips) with handedness and confidence
- Runs on its own FPS budget (`HAND_TRACKING_TARGET_FPS`) independently of the floor pipeline
- An index-finger ray is computed from the fingertip and adjacent finger joints, normalised, and converted to the Unity camera convention
- The server emits `AI_HAND_DATA` packets on every tracked frame; Unity renders a holographic 21-joint glove and casts the ray for button interaction

Hand tracking continues running in post-lock idle mode (when depth and segmentation are switched off) because interaction is the primary use case after the boundary is set.

---

## Post-Lock Optimization

Once the boundary is locked, the expensive pipeline stages are selectively disabled under `POST_LOCK_ENABLED`:

| Stage | Pre-lock | Post-lock |
|---|---|---|
| Depth Anything V2 | ✅ every frame | ❌ stopped (`POST_LOCK_STOP_DEPTH`) |
| SegFormer segmentation | ✅ every frame | ❌ stopped (`POST_LOCK_STOP_SEGMENTATION`) |
| RANSAC plane fitting | ✅ every frame | ❌ stopped (`POST_LOCK_STOP_PLANE_FIT`) |
| MediaPipe hand tracking | ✅ at `HAND_TRACKING_TARGET_FPS` | ✅ continues at full rate |
| Diagnostic pipeline | ❌ — | Optional at `POST_LOCK_DIAGNOSTIC_FPS` |

This cuts the per-frame cost from ~350 ms (depth + segmentation + RANSAC) to ~25 ms (hand only), allowing hand tracking to saturate its target FPS without competing with the heavy models.

---

## Session Metrics & Artifacts

On shutdown, `export_session_artifacts()` writes three files to `artifacts/`:

**`session_metrics.json`** — full dictionary of all performance counters  
**`session_metrics.csv`** — flat key/value table for spreadsheet analysis  
**`session_summary.txt`** — human-readable report:

```
AI Guardian — Phase 9.1 Session Summary
==================================================
Duration: 242.4s
Total Frames: 567
Lock Frame: 516
Lock Source: unity_accepted
Errors: 0
Warnings: 0

--- WARM PHASE (depth+seg ON) ---
Duration: 181.2s
Frames: 516
Pipeline FPS: 0.9
Avg Depth: 210.3ms
Avg Seg: 132.7ms
Avg Plane: 8.1ms
Avg Hand: 24.8ms

--- IDLE PHASE (depth+seg OFF, hand ON) ---
Duration: 61.2s
Frames: 51
Idle FPS: 0.8
Hand FPS: 9.3
Avg Hand: 24.5ms
```

---

## Configuration Reference

All parameters live in a single `config.py`. Key groups:

```python
# ── Networking ─────────────────────────────────────────────────
UDP_HOST, UDP_PORT          = "0.0.0.0", 9001
TCP_HOST, TCP_PORT          = "0.0.0.0", 9002

# ── Models ─────────────────────────────────────────────────────
DEPTH_MODEL_NAME            = "depth_anything_v2_small"
SEG_MODEL_NAME              = "nvidia/segformer-b0-finetuned-ade-512-512"
DEPTH_USE_METRIC_INDOOR     = True

# ── Pipeline ───────────────────────────────────────────────────
PIPELINE_TARGET_FPS         = 1.0       # Heavy pipeline; ~1 FPS is realistic
BOUNDARY_MODE               = "ai_floor"

# ── RANSAC ─────────────────────────────────────────────────────
RANSAC_ITERATIONS           = 200
RANSAC_INLIER_THRESHOLD_M   = 0.05
PLANE_MIN_INLIERS           = 500
PLANE_MIN_INLIER_RATIO      = 0.45
PLANE_MAX_RMSE_M            = 0.08

# ── Scan gate (loose, for preview) ─────────────────────────────
PLANE_MIN_HORIZONTAL_SCORE  = 0.75
FLOOR_MIN_PIXEL_RATIO       = 0.005

# ── Lock gate (strict) ─────────────────────────────────────────
LOCK_MIN_HORIZONTAL_SCORE   = 0.90
LOCK_MIN_FLOOR_RATIO        = 0.05
LOCK_MIN_INLIER_RATIO       = 0.75
LOCK_MAX_RMSE               = 0.03
LOCK_MIN_CONFIDENCE         = 0.80

# ── Accumulator ────────────────────────────────────────────────
FLOOR_ACCUM_ENABLED         = True
FLOOR_ACCUM_CELL_M          = 0.05     # 5 cm grid cells
FLOOR_ACCUM_MIN_AREA_M2     = 2.0     # Must accumulate ≥ 2 m² before lock
FLOOR_ACCUM_STABILITY_WINDOW= 5       # Stable for 5 consecutive frames

# ── Hand tracking ──────────────────────────────────────────────
HAND_TRACKING_ENABLED       = True
HAND_TRACKING_TARGET_FPS    = 10.0
HAND_MIN_DETECTION_CONF     = 0.5
HAND_MIN_TRACKING_CONF      = 0.5

# ── Post-lock optimisation ─────────────────────────────────────
POST_LOCK_ENABLED           = True
POST_LOCK_STOP_DEPTH        = True
POST_LOCK_STOP_SEGMENTATION = True
POST_LOCK_KEEP_HAND         = True
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- CUDA GPU strongly recommended for depth + segmentation inference
- Android device with ARCore support
- Unity 2022.3+ with AR Foundation

### Python Server Setup

```bash
# Clone the repository
git clone https://github.com/mohamed1232005/GP-VR-Guardian-System.git
cd GP-VR-Guardian-System
git checkout AI-Guardian-pipeline-2--final-version

# Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the server
python server.py
```

The server will print the active configuration and wait for UDP frames on port 9001. Models are downloaded automatically from HuggingFace on first run.

### Unity Client

1. Open the Unity project (separate repository / submodule).
2. Set the server IP in `ARCameraFrameProvider` (or the relevant settings panel) to the PC running `server.py`.
3. Build and deploy to your Android device.
4. Launch the app, select a mode, and slowly sweep the phone across the floor.

### Running in Dummy Mode (no models)

```python
# config.py
BOUNDARY_MODE   = "dummy"
DEPTH_ENABLED   = False
```

This starts the server without loading any AI models — useful for testing the networking layer.

---

## Performance Results

Measured on Samsung SM-A346E (Galaxy A34, arm64-v8a), Python server on a host PC over Wi-Fi.

| Metric | Value |
|---|---|
| Session duration | 242.4 s |
| Total UDP frames received | 567 |
| Lock frame | 516 |
| Lock source | Unity-accepted |
| Errors | 0 |
| Warnings | 0 |
| **Warm phase pipeline FPS** | **~0.9 FPS** (depth+seg+RANSAC, ~350 ms/frame) |
| Avg depth inference | ~210 ms |
| Avg segmentation inference | ~133 ms |
| Avg RANSAC + geometry | ~8 ms |
| **Post-lock hand tracking** | **~9 FPS** (~25 ms/frame) |
| **Unity render rate** | **27.5–30 FPS** (always smooth) |
| UDP input rate | ~9 FPS |

> Four frame rates coexist in this system: UDP input (~9 FPS), heavy AI pipeline (~1 FPS), hand tracking (~9 FPS), and Unity render (~30 FPS). Only the heavy AI is slow; the user experience is always smooth.

---

## Limitations & Future Work

### Current Limitations

- Monocular depth is uncertain on glossy or textureless floors and at long range.
- A minimum floor patch must be visible in the lower image; very cluttered or narrow spaces can starve the accumulator.
- The pipeline is sensitive to camera angle and the selected rotation pair — the main cause of "no preview."
- The boundary is a single oriented rectangle; irregular room shapes are only approximated.
- Evaluation is a single instrumented session; broader benchmarking across rooms and devices is still needed.
- The heavy AI runs at ~1 FPS; on-device acceleration would be required to raise this.

### Future Work

- Polygonal (non-rectangular) boundaries for irregular rooms.
- On-device model acceleration (ONNX or TensorRT) to lift the scanning-phase frame rate.
- Automatic validation and enabling of landscape optical-axis correction.
- Pose filtering and hand-depth refinement to further stabilise interaction.
- A formal multi-room, multi-device quantitative study and a user study for the downstream rehabilitation use case.

---

## References

1. L. Yang et al., "Depth Anything V2," NeurIPS 2024.
2. E. Xie et al., "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers," NeurIPS 2021.
3. B. Zhou et al., "Scene Parsing through ADE20K Dataset," CVPR 2017.
4. F. Zhang et al., "MediaPipe Hands: On-device Real-time Hand Tracking," arXiv:2006.10214, 2020.
5. M. A. Fischler & R. C. Bolles, "Random Sample Consensus," Communications of the ACM, vol. 24, no. 6, 1981.
6. A. Paszke et al., "PyTorch," NeurIPS 2019.
7. T. Wolf et al., "Transformers: State-of-the-Art NLP," EMNLP 2020.
8. G. Bradski, "The OpenCV Library," Dr. Dobb's Journal, 2000.
9. C. R. Harris et al., "Array Programming with NumPy," Nature, vol. 585, 2020.
10. Unity Technologies, "AR Foundation." https://docs.unity3d.com/Packages/com.unity.xr.arfoundation@latest
11. Google LLC, "ARCore." https://developers.google.com/ar

---

<p align="center">
  Developed at <strong>Zewail City of Science and Technology</strong>, School of CSAI<br/>
  Supervisor: Dr. Mayada Hadhad · June 2026
</p>
