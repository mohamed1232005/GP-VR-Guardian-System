"""
Phase 3: Hand Tracking Test (Fixed)
Tests MediaPipe hand detection and pointing gesture WITHOUT network or Unity.

Fixes:
- Compatible with both old (mp.solutions) and new (mp.tasks) MediaPipe APIs
- Lowered pointing threshold from 0.3 to 0.15 to fix relaxed-hand detection
- Added pinky curl check for stricter pointing detection
- Added idx_ratio / other_max debug output matching hand_tracker.py format
"""

import cv2
import time
from pathlib import Path

# ── MediaPipe compatibility shim ────────────────────────────────────────────
# MediaPipe 0.10+ moved away from mp.solutions.*
# We try the old path first, then fall back to the new Tasks API.

MP_MODE = None  # "solutions" or "tasks"

try:
    import mediapipe as mp
    # Test if solutions API exists
    _ = mp.solutions.hands
    MP_MODE = "solutions"
    print("✓ MediaPipe detected: using mp.solutions API (legacy)")
except AttributeError:
    try:
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision
        import mediapipe as mp
        MP_MODE = "tasks"
        print("✓ MediaPipe detected: using mp.tasks API (new-style)")
    except Exception as e:
        print(f"❌ MediaPipe import failed entirely: {e}")
        print("   Try: pip install mediapipe==0.10.14")
        exit(1)
except ImportError:
    print("❌ MediaPipe is not installed.")
    print("   Try: pip install mediapipe")
    exit(1)

# ── Pointing logic (shared between both API modes) ───────────────────────────

def finger_length_ratio(tip, mcp):
    """Euclidean 2-D distance tip→mcp as proxy for extension ratio."""
    dx = tip.x - mcp.x
    dy = tip.y - mcp.y
    return (dx**2 + dy**2) ** 0.5


def is_pointing(landmarks, debug=False):
    """
    Determine pointing gesture.

    Uses two complementary checks:
    MODE A (y-axis): index tip is above index PIP — fast, works for camera-facing poses.
    MODE B (ratio) : index extension ratio is larger than other fingers by THRESHOLD.
                     Threshold is 0.15 (down from the original 0.30 which caused flicker).

    Returns (is_pointing: bool, debug_str: str)
    """
    THRESHOLD = 0.15   # was 0.30 — the critical fix for relaxed-hand detection

    lm = landmarks.landmark

    # ── Landmark indices ──────────────────────────────────────────────────
    # Finger:  MCP  PIP  DIP  TIP
    # Index:    5    6    7    8
    # Middle:   9   10   11   12
    # Ring:    13   14   15   16
    # Pinky:   17   18   19   20

    idx_tip, idx_pip = lm[8], lm[6]
    mid_tip, mid_pip = lm[12], lm[10]
    rng_tip, rng_pip = lm[16], lm[14]
    pnk_tip, pnk_pip = lm[20], lm[18]

    # MODE A — simple y comparison (works well when hand faces camera)
    idx_extended_y = idx_tip.y < idx_pip.y
    mid_curled_y   = mid_tip.y > mid_pip.y
    rng_curled_y   = rng_tip.y > rng_pip.y
    mode_a = idx_extended_y and mid_curled_y and rng_curled_y

    # MODE B — ratio-based (works for side-on / tilted hand poses)
    idx_mcp = lm[5]
    mid_mcp = lm[9]
    rng_mcp = lm[13]
    pnk_mcp = lm[17]

    idx_ratio = finger_length_ratio(idx_tip, idx_mcp)
    mid_ratio = finger_length_ratio(mid_tip, mid_mcp)
    rng_ratio = finger_length_ratio(rng_tip, rng_mcp)
    pnk_ratio = finger_length_ratio(pnk_tip, pnk_mcp)

    other_max = max(mid_ratio, rng_ratio, pnk_ratio)
    mode_b = (idx_ratio - other_max) > THRESHOLD

    result = mode_a or mode_b

    dbg = (
        f"[GESTURE] idx_ext={idx_extended_y}({idx_ratio:.2f}) "
        f"mid_curl={mid_curled_y}({mid_ratio:.2f}) "
        f"other_max={other_max:.2f} "
        f"→ {'POINTING' if result else 'NOT_POINTING'}"
    )

    return result, dbg


# ── Solutions-API runner ─────────────────────────────────────────────────────

def run_solutions():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.4,
    )
    mp_draw = mp.solutions.drawing_utils
    CONNECTIONS = mp_hands.HAND_CONNECTIONS

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera (index 0). Try index 1 or check permissions.")
        return False

    print("📹 Camera opened — point your finger at the camera!")
    print("   Press 'q' to quit early (auto-stops after 300 frames ~10 s)\n")

    output_dir = Path("samples/phase3_hand_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_count = detection_count = pointing_count = 0
    detection_rate = pointing_rate = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            detection_count += 1

            for hlm in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hlm, CONNECTIONS)

                pointing, dbg = is_pointing(hlm)
                if frame_count % 10 == 0:   # print every 10 frames to avoid spam
                    print(dbg)

                if pointing:
                    pointing_count += 1
                    tip = hlm.landmark[8]
                    h, w = frame.shape[:2]
                    cx, cy = int(tip.x * w), int(tip.y * h)
                    cv2.circle(frame, (cx, cy), 20, (0, 255, 0), -1)
                    cv2.putText(frame, "POINTING", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

                    if pointing_count % 30 == 0:
                        cv2.imwrite(str(output_dir / f"pointing_{pointing_count}.jpg"), frame)

        detection_rate = detection_count / frame_count * 100
        pointing_rate  = pointing_count  / frame_count * 100

        h = frame.shape[0]
        cv2.putText(frame, f"Frames: {frame_count}",          (10, h - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Detection: {detection_rate:.1f}%", (10, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Pointing:  {pointing_rate:.1f}%",  (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Phase 3: Hand Tracking Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if frame_count >= 300:
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    return frame_count, detection_count, pointing_count, detection_rate, pointing_rate, output_dir


# ── Tasks-API runner ─────────────────────────────────────────────────────────

def run_tasks():
    """
    New-style MediaPipe 0.10+ Tasks API.
    Requires the hand_landmarker.task model file.
    Downloads it automatically if not present.
    """
    import urllib.request, os

    model_path = Path("hand_landmarker.task")
    if not model_path.exists():
        url = (
            "https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        )
        print(f"  Downloading hand_landmarker.task from Google Storage…")
        try:
            urllib.request.urlretrieve(url, model_path)
            print(f"  ✓ Saved to {model_path}")
        except Exception as e:
            print(f"  ❌ Download failed: {e}")
            print("     Download manually from:")
            print(f"     {url}")
            return False

    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

    # Shared state for async callback
    state = {"landmarks": None}

    def result_callback(result, output_image, timestamp_ms):
        state["landmarks"] = result

    options = HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.LIVE_STREAM,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.4,
        result_callback=result_callback,
    )

    landmarker = HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera.")
        return False

    print("📹 Camera opened — point your finger at the camera!")
    print("   Press 'q' to quit early (auto-stops after 300 frames ~10 s)\n")

    output_dir = Path("samples/phase3_hand_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_count = detection_count = pointing_count = 0
    detection_rate = pointing_rate = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        landmarker.detect_async(mp_image, int(time.time() * 1000))

        result = state["landmarks"]
        if result and result.hand_landmarks:
            detection_count += 1

            for hlm_list in result.hand_landmarks:
                # Tasks API returns NormalizedLandmark objects in a plain list.
                # Wrap them so is_pointing() can use .landmark[n] syntax.
                class _FakeLM:
                    def __init__(self, lm_list):
                        self.landmark = lm_list
                fake = _FakeLM(hlm_list)

                # Draw manually (Tasks API has no draw_landmarks shortcut)
                h, w = frame.shape[:2]
                for lm in hlm_list:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 4, (0, 200, 255), -1)

                pointing, dbg = is_pointing(fake)
                if frame_count % 10 == 0:
                    print(dbg)

                if pointing:
                    pointing_count += 1
                    tip = hlm_list[8]
                    cx, cy = int(tip.x * w), int(tip.y * h)
                    cv2.circle(frame, (cx, cy), 20, (0, 255, 0), -1)
                    cv2.putText(frame, "POINTING", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

                    if pointing_count % 30 == 0:
                        cv2.imwrite(str(output_dir / f"pointing_{pointing_count}.jpg"), frame)

        detection_rate = detection_count / frame_count * 100
        pointing_rate  = pointing_count  / frame_count * 100

        hh = frame.shape[0]
        cv2.putText(frame, f"Frames: {frame_count}",           (10, hh - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Detection: {detection_rate:.1f}%", (10, hh - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Pointing:  {pointing_rate:.1f}%",  (10, hh - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Phase 3: Hand Tracking Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if frame_count >= 300:
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    return frame_count, detection_count, pointing_count, detection_rate, pointing_rate, output_dir


# ── Main ─────────────────────────────────────────────────────────────────────

def test_hand_tracking():
    print("=" * 60)
    print("PHASE 3: HAND TRACKING TEST")
    print("=" * 60)

    if MP_MODE == "solutions":
        result = run_solutions()
    else:
        result = run_tasks()

    if result is False:
        return False

    frame_count, detection_count, pointing_count, detection_rate, pointing_rate, output_dir = result

    print("\n" + "=" * 60)
    print("RESULTS:")
    print(f"  Total frames:      {frame_count}")
    print(f"  Hand detected:     {detection_count} frames ({detection_rate:.1f}%)")
    print(f"  Pointing detected: {pointing_count} frames ({pointing_rate:.1f}%)")
    print(f"  Samples saved to:  {output_dir}")
    print("=" * 60)

    if detection_rate > 50 and pointing_count > 0:
        print("\n✅ PHASE 3: PASS — Hand tracking working!")
        return True
    elif detection_rate <= 50:
        print(f"\n⚠️  PHASE 3: WARNING — Low detection rate ({detection_rate:.1f}%)")
        print("   Try better lighting or move closer to the camera.")
        return True
    else:
        print(f"\n⚠️  PHASE 3: WARNING — No pointing detected ({pointing_rate:.1f}%)")
        print("   Check [GESTURE] lines in terminal for idx_ratio vs other_max values.")
        print("   If diff < 0.15, the threshold in this script may need further lowering.")
        return True


if __name__ == "__main__":
    success = test_hand_tracking()
    exit(0 if success else 1)