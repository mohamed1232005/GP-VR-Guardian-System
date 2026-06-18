"""
toshfa_pose_session.py — Phase 10
=================================

Stand-alone Toshfa rehab subsystem. Uses the LAPTOP webcam (not the phone AR
camera) to confirm the patient's lower-back rotation pose, then emits low-rate
TOSHFA_* JSON events to Unity over UDP. Unity reacts by cross-fading the wall
image and playing the next voice clip.

Run:
    python -m ai_guardian_server.toshfa.toshfa_pose_session
    python -m ai_guardian_server.toshfa.toshfa_pose_session --unity-host 127.0.0.1 --unity-port 5566

What this script does NOT touch:
    • Floor detection, guardian boundary, depth model, anything in
      ai_guardian_server/models/ai_pipeline_worker.py.
    • Hand tracking — that runs from the phone AR camera and is separate.

Failure modes that must NEVER crash the gym:
    • Webcam unavailable          → log "camera_not_ready" and exit cleanly.
    • MediaPipe import fails      → log clearly and exit cleanly.
    • Network send fails          → log and keep running.

Acceptance logs (also sent to Unity):
    [TOSHFA_START]
    [TOSHFA_CAMERA_READY]
    [TOSHFA_STEP_ENTER]    step=A
    [TOSHFA_STEP_CONFIRMED]
    [TOSHFA_STEP_ENTER]    step=B
    [TOSHFA_STEP_CONFIRMED]
    [TOSHFA_SAFETY_WARNING] reason=over_rotation angle=…
    [TOSHFA_HEARTBEAT]      fps score step hold
    [TOSHFA_SESSION_DONE]
"""
from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import time
from typing import Optional, Tuple


# --------------------------------------------------------------------------
# Tunables (spec values — patient comfort)
# --------------------------------------------------------------------------

ENTER_THRESHOLD     = 0.75
EXIT_THRESHOLD      = 0.60
EMA_ALPHA           = 0.20
HOLD_SECONDS        = 2.0
MIN_VISIBILITY      = 0.40
SAFETY_ANGLE_DEG    = 65.0           # over-rotation > 65 → safety warning
SAFETY_HOLD_SECONDS = 0.4            # debounce — warn only if held >0.4s
HEARTBEAT_SECONDS   = 5.0
PROGRESS_SEND_HZ    = 4.0            # max HOLD_PROGRESS sends/sec

# Phase 10.1 — per the wall images:
#   Style B = STEP 1: PREPARATION & SETUP (sit upright, ankle crossed → neutral
#             seated, shoulder/hip yaw ≈ 0°). Shown FIRST.
#   Style A = STEP 2: THE TWIST & STRETCH (gently twist upper body to the right,
#             yaw ≈ 25-35°). Shown SECOND after B is confirmed.
# Reference rotation angles per step (degrees of shoulder-vs-hip yaw, signed):
STEP_TARGETS = {
    "B": (0.0,  12.0),    # Step 1: neutral seated, small sigma
    "A": (30.0, 15.0),    # Step 2: gentle right rotation, larger sigma
}


# --------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------

class UnitySender:
    """Tiny UDP JSON sender. Failures are non-fatal — they log and continue."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except OSError as e:
            print(f"[TOSHFA_NET_ERR] cannot create UDP socket: {e}")
            self.sock = None

    def send(self, payload: dict) -> None:
        if self.sock is None:
            return
        try:
            data = json.dumps(payload).encode("utf-8")
            self.sock.sendto(data, (self.host, self.port))
        except OSError as e:
            # Don't crash the session over a transient send error.
            print(f"[TOSHFA_NET_ERR] send failed: {e}")

    def close(self) -> None:
        try:
            if self.sock is not None:
                self.sock.close()
        except OSError:
            pass


# --------------------------------------------------------------------------
# Pose feature: shoulder-vs-hip yaw (degrees, signed)
# --------------------------------------------------------------------------

# MediaPipe landmark indices we care about
LM_LEFT_SHOULDER  = 11
LM_RIGHT_SHOULDER = 12
LM_LEFT_HIP       = 23
LM_RIGHT_HIP      = 24


def torso_normalized_rotation(lm) -> Optional[Tuple[float, float]]:
    """Return (rotation_deg, min_visibility) or None if landmarks invalid.

    Rotation is the signed yaw between the shoulder vector and the hip vector
    in the 2D image plane (positive = patient rotates right toward camera).
    Torso-normalized: only the *direction* of each vector matters, so different
    body sizes give the same angle.
    """
    if lm is None:
        return None
    try:
        ls = lm[LM_LEFT_SHOULDER]
        rs = lm[LM_RIGHT_SHOULDER]
        lh = lm[LM_LEFT_HIP]
        rh = lm[LM_RIGHT_HIP]
    except (KeyError, IndexError):
        return None

    vis = min(ls.visibility, rs.visibility, lh.visibility, rh.visibility)
    if vis < MIN_VISIBILITY:
        return None

    sh_dx = rs.x - ls.x
    sh_dy = rs.y - ls.y
    hp_dx = rh.x - lh.x
    hp_dy = rh.y - lh.y

    if abs(sh_dx) + abs(sh_dy) < 1e-4 or abs(hp_dx) + abs(hp_dy) < 1e-4:
        return None

    a_sh = math.atan2(sh_dy, sh_dx)
    a_hp = math.atan2(hp_dy, hp_dx)
    deg = math.degrees(a_sh - a_hp)
    if deg > 180.0:
        deg -= 360.0
    elif deg < -180.0:
        deg += 360.0
    return float(deg), float(vis)


def gaussian_score(rotation_deg: float, step: str) -> float:
    """Gaussian similarity between current rotation and the step's target."""
    if step not in STEP_TARGETS:
        return 0.0
    mu, sigma = STEP_TARGETS[step]
    z = (rotation_deg - mu) / max(sigma, 1e-3)
    return float(math.exp(-0.5 * z * z))


# --------------------------------------------------------------------------
# Session state machine — hysteresis hold
# --------------------------------------------------------------------------

class StepState:
    """Per-step hysteresis hold tracker.

    State machine:
        IDLE → (score >= ENTER) → HOLDING (record start_time)
        HOLDING → (score < EXIT) → IDLE (drop hold)
        HOLDING → (now - start_time >= HOLD_SECONDS) → CONFIRMED
    """
    def __init__(self):
        self.score_ema = 0.0
        self.holding = False
        self.hold_start = 0.0
        self.confirmed = False

    def update(self, raw_score: float, now: float) -> Tuple[bool, float, bool]:
        """Returns (in_hold, hold_progress_0_to_1, confirmed_now)."""
        # EMA so a single bad frame doesn't drop the hold.
        self.score_ema = (1.0 - EMA_ALPHA) * self.score_ema + EMA_ALPHA * raw_score

        confirmed_now = False
        if self.confirmed:
            return self.holding, 1.0, False

        if not self.holding:
            if self.score_ema >= ENTER_THRESHOLD:
                self.holding = True
                self.hold_start = now
        else:
            if self.score_ema < EXIT_THRESHOLD:
                self.holding = False
                self.hold_start = 0.0

        progress = 0.0
        if self.holding:
            progress = min(1.0, (now - self.hold_start) / HOLD_SECONDS)
            if progress >= 1.0:
                self.confirmed = True
                confirmed_now = True
        return self.holding, progress, confirmed_now


# --------------------------------------------------------------------------
# Pose adapter — works with BOTH the legacy mp.solutions.pose API (mediapipe
# <0.10.14) and the modern Tasks API (mediapipe >=0.10.14, which dropped
# `mp.solutions`). The legacy path is tried first because it doesn't need a
# downloaded model file. If that fails, we fall back to PoseLandmarker; if
# the .task model file is missing, we print clear download instructions.
# --------------------------------------------------------------------------

class _PoseLandmark:
    """Uniform landmark record (matches MediaPipe NormalizedLandmark API)."""
    __slots__ = ("x", "y", "z", "visibility")
    def __init__(self, x, y, z, visibility):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


class PoseAdapter:
    """Pick whichever MediaPipe API is available.

    Public API:
        adapter.detect(rgb) -> list[_PoseLandmark] | None
        adapter.close()
        adapter.api_name -> "legacy_solutions" | "tasks_landmarker"
    """

    POSE_TASK_FILENAMES = ("pose_landmarker_lite.task",
                           "pose_landmarker_full.task",
                           "pose_landmarker.task")

    def __init__(self):
        self._mode = None          # "legacy" or "tasks"
        self._legacy_pose = None
        self._tasks_detector = None
        self._mp = None            # module ref for Image construction
        self.api_name = "uninit"

    # -- factory ----------------------------------------------------------
    @classmethod
    def try_create(cls):
        import os
        import mediapipe as mp
        adapter = cls()
        adapter._mp = mp

        # 1) Legacy mp.solutions.pose.Pose() — works on mediapipe < 0.10.14.
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
            try:
                adapter._legacy_pose = mp.solutions.pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                adapter._mode = "legacy"
                adapter.api_name = "legacy_solutions"
                print("[TOSHFA_MP_API] legacy=mp.solutions.pose.Pose")
                return adapter
            except Exception as e:
                print(f"[TOSHFA_MP_API] legacy init failed: {e}, trying Tasks API…")

        # 2) Tasks API — modern mediapipe. Needs a downloaded .task model file.
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = None
            for fn in cls.POSE_TASK_FILENAMES:
                candidate = os.path.join(base_dir, fn)
                if os.path.exists(candidate):
                    model_path = candidate
                    break

            if model_path is None:
                print(f"[TOSHFA_MP_ERR] PoseLandmarker model NOT FOUND under {base_dir}")
                print(f"[TOSHFA_MP_ERR] download one of these:")
                print(f"  Lite  (5.5 MB): https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")
                print(f"  Full  (9 MB):   https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task")
                print(f"[TOSHFA_MP_ERR] place it at: {os.path.join(base_dir, cls.POSE_TASK_FILENAMES[0])}")
                return None

            base_options = mp_python.BaseOptions(model_asset_path=model_path)
            options = mp_vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_segmentation_masks=False,
            )
            adapter._tasks_detector = mp_vision.PoseLandmarker.create_from_options(options)
            adapter._mode = "tasks"
            adapter.api_name = "tasks_landmarker"
            print(f"[TOSHFA_MP_API] tasks=PoseLandmarker model={os.path.basename(model_path)}")
            return adapter
        except Exception as e:
            print(f"[TOSHFA_MP_ERR] Tasks API init failed: {type(e).__name__}: {e}")
            return None

    # -- per-frame --------------------------------------------------------
    def detect(self, rgb_frame):
        """Run detection on an HxWx3 RGB uint8 array. Returns list of 33
        _PoseLandmark records, or None if no pose detected."""
        if self._mode == "legacy":
            results = self._legacy_pose.process(rgb_frame)
            if not results or not results.pose_landmarks:
                return None
            return [
                _PoseLandmark(lm.x, lm.y, lm.z, lm.visibility)
                for lm in results.pose_landmarks.landmark
            ]
        elif self._mode == "tasks":
            mp_image = self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB,
                data=rgb_frame.copy(),
            )
            result = self._tasks_detector.detect(mp_image)
            if not result or not result.pose_landmarks or len(result.pose_landmarks) == 0:
                return None
            # Tasks API: result.pose_landmarks is List[List[NormalizedLandmark]].
            lms = result.pose_landmarks[0]
            return [
                _PoseLandmark(lm.x, lm.y, lm.z, lm.visibility)
                for lm in lms
            ]
        return None

    # -- cleanup ----------------------------------------------------------
    def close(self):
        try:
            if self._legacy_pose is not None: self._legacy_pose.close()
        except Exception:
            pass
        try:
            if self._tasks_detector is not None: self._tasks_detector.close()
        except Exception:
            pass


# --------------------------------------------------------------------------
# Main session
# --------------------------------------------------------------------------

def run_session(unity_host: str, unity_port: int, camera_index: int,
                headless: bool) -> int:
    sender = UnitySender(unity_host, unity_port)
    print(f"[TOSHFA_START] unity={unity_host}:{unity_port}")
    sender.send({"type": "TOSHFA_START"})

    # Lazy imports so an absent dependency is reported clearly, not as a stack
    # trace from the harness.
    try:
        import cv2
    except Exception as e:
        print(f"[TOSHFA_DEP_ERR] cv2 missing: {e} — pip install opencv-python")
        sender.send({"type": "TOSHFA_SESSION_DONE"})
        sender.close()
        return 2
    try:
        import mediapipe as mp
    except Exception as e:
        print(f"[TOSHFA_DEP_ERR] mediapipe missing: {e} — pip install mediapipe")
        sender.send({"type": "TOSHFA_SESSION_DONE"})
        sender.close()
        return 2

    # ── Webcam ──────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[TOSHFA_CAMERA_ERR] cv2.VideoCapture({camera_index}) failed — " +
              "gym continues without Toshfa")
        sender.send({"type": "TOSHFA_SAFETY_WARNING",
                     "reason": "camera_not_ready", "angle": 0.0})
        sender.send({"type": "TOSHFA_SESSION_DONE"})
        sender.close()
        return 3
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS,          30)
    print("[TOSHFA_CAMERA_READY]")
    sender.send({"type": "TOSHFA_CAMERA_READY"})

    # ── MediaPipe Pose ──────────────────────────────────────────────────
    # PoseAdapter tries the legacy mp.solutions.pose first (no model file
    # needed) and falls back to the modern Tasks API PoseLandmarker
    # (needs pose_landmarker_lite.task next to this script). If neither
    # works, log clearly and exit — never crash the gym.
    pose = PoseAdapter.try_create()
    if pose is None:
        print("[TOSHFA_MP_ERR] could not start MediaPipe Pose with either API — " +
              "see download URL above. Gym continues without Toshfa.")
        cap.release()
        sender.send({"type": "TOSHFA_SESSION_DONE"})
        sender.close()
        return 4

    # ── Session state ────────────────────────────────────────────────────
    # Phase 10.1: B first (Preparation), then A (Twist & Stretch).
    sequence = ["B", "A"]
    seq_idx = 0
    current_step = sequence[seq_idx]
    state = StepState()

    sender.send({"type": "TOSHFA_STEP_ENTER", "step": current_step})
    print(f"[TOSHFA_STEP_ENTER] step={current_step}")

    last_progress_sent = 0.0
    last_heartbeat = time.time()
    safety_active_since: Optional[float] = None
    frame_count = 0
    start = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                # Don't crash — just keep trying for a few seconds.
                time.sleep(0.05)
                continue
            frame = cv2.flip(frame, 1)
            frame_count += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # PoseAdapter abstracts the legacy vs Tasks API split. It returns a
            # list of 33 _PoseLandmark records (or None). The downstream
            # torso_normalized_rotation expects indexable landmarks with .x/.y/.visibility,
            # which _PoseLandmark satisfies.
            landmarks = pose.detect(rgb)

            score = 0.0
            rotation_deg = 0.0
            if landmarks is not None:
                rot = torso_normalized_rotation(landmarks)
                if rot is not None:
                    rotation_deg, _vis = rot
                    score = gaussian_score(rotation_deg, current_step)

            # Safety: over-rotation. Debounced so a single flicker doesn't
            # trigger the warning.
            now = time.time()
            abs_rot = abs(rotation_deg)
            if abs_rot > SAFETY_ANGLE_DEG:
                if safety_active_since is None:
                    safety_active_since = now
                elif now - safety_active_since >= SAFETY_HOLD_SECONDS:
                    sender.send({"type": "TOSHFA_SAFETY_WARNING",
                                 "reason": "over_rotation", "angle": abs_rot})
                    print(f"[TOSHFA_SAFETY_WARNING] reason=over_rotation angle={abs_rot:.1f}")
                    # Reset so we don't spam — re-warn after another 0.4s of violation.
                    safety_active_since = now + HEARTBEAT_SECONDS
            else:
                safety_active_since = None

            # Hysteresis update — only when not in safety violation.
            confirmed_now = False
            holding = False
            progress = 0.0
            if abs_rot <= SAFETY_ANGLE_DEG:
                holding, progress, confirmed_now = state.update(score, now)

            # Send HOLD_PROGRESS at <= PROGRESS_SEND_HZ to keep the wire calm.
            if now - last_progress_sent >= 1.0 / PROGRESS_SEND_HZ:
                last_progress_sent = now
                sender.send({
                    "type": "TOSHFA_HOLD_PROGRESS",
                    "step": current_step,
                    "progress": round(progress, 3),
                    "score": round(state.score_ema, 3),
                })

            if confirmed_now:
                sender.send({"type": "TOSHFA_STEP_CONFIRMED", "step": current_step})
                print(f"[TOSHFA_STEP_CONFIRMED] step={current_step}")
                # Advance to next step or end the session.
                seq_idx += 1
                if seq_idx >= len(sequence):
                    sender.send({"type": "TOSHFA_SESSION_DONE"})
                    print("[TOSHFA_SESSION_DONE]")
                    break
                current_step = sequence[seq_idx]
                state = StepState()
                sender.send({"type": "TOSHFA_STEP_ENTER", "step": current_step})
                print(f"[TOSHFA_STEP_ENTER] step={current_step}")

            # Heartbeat
            if now - last_heartbeat >= HEARTBEAT_SECONDS:
                elapsed = now - start
                fps = frame_count / elapsed if elapsed > 0 else 0.0
                print(f"[TOSHFA_HEARTBEAT] fps={fps:.1f} score={state.score_ema:.2f} " +
                      f"step={current_step} hold={progress:.2f}")
                last_heartbeat = now

            if not headless:
                # Lightweight diagnostic window — operator only.
                txt = (f"step={current_step} score={state.score_ema:.2f} " +
                       f"hold={progress:.2f} rot={rotation_deg:+.0f}")
                cv2.putText(frame, txt, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)
                cv2.imshow("Toshfa Session", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        try:
            pose.close()
        except Exception:
            pass
        cap.release()
        if not headless:
            cv2.destroyAllWindows()
        sender.close()
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Toshfa Phase-10 pose session")
    p.add_argument("--unity-host", default="127.0.0.1")
    p.add_argument("--unity-port", type=int, default=5566)
    p.add_argument("--camera",     type=int, default=0)
    p.add_argument("--headless",   action="store_true",
                   help="Suppress the OpenCV diagnostic window.")
    args = p.parse_args()
    try:
        return run_session(args.unity_host, args.unity_port, args.camera, args.headless)
    except Exception as e:
        # Last-chance guard: never let a Toshfa exception bring down the host.
        print(f"[TOSHFA_FATAL] {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
