import cv2
import numpy as np
import time
from pathlib import Path


def test_camera_capture_improved():
    """Enhanced Phase 1 with proper validation"""
    
    print("=" * 60)
    print("PHASE 1: CAMERA CAPTURE TEST (ENHANCED)")
    print("=" * 60)
    
    output_dir = Path("samples/phase1_camera_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera not accessible")
        return False
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print(f"✓ Camera: {int(cap.get(3))}×{int(cap.get(4))} @ {cap.get(5)} FPS (claimed)")
    
    # Warm-up: discard first 5 frames (camera initialization)
    print("\n Warming up camera...")
    for _ in range(5):
        cap.read()
    
    # Capture test
    print("\n Capturing 30 frames with validation...")
    
    frame_times = []
    encode_times = []
    jpeg_sizes = []
    saved_files = []
    dropped_frames = 0
    corrupt_jpegs = 0
    dark_frames = 0
    
    target_fps = 30
    frame_period = 1.0 / target_fps
    
    for i in range(30):
        loop_start = time.time()
        
        # 1. CAPTURE
        t_cap_start = time.time()
        ret, frame = cap.read()
        t_cap = (time.time() - t_cap_start) * 1000
        
        if not ret or frame is None or frame.size == 0:
            dropped_frames += 1
            print(f"  Frame {i+1}: DROPPED")
            continue
        
        # 2. VALIDATE FRAME CONTENT
        mean_brightness = np.mean(frame)
        if mean_brightness < 10:
            dark_frames += 1
            print(f"  ⚠️  Frame {i+1}: Too dark ({mean_brightness:.1f})")
        
        # 3. ENCODE
        t_enc_start = time.time()
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
        result, encoded = cv2.imencode('.jpg', frame, encode_param)
        t_enc = (time.time() - t_enc_start) * 1000
        
        if not result:
            print(f"   Frame {i+1}: Encoding failed")
            continue
        
        # 4. VALIDATE JPEG (decode to verify)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is None:
            corrupt_jpegs += 1
            print(f"  ❌ Frame {i+1}: JPEG corrupt")
            continue
        
        # 5. SAVE & VERIFY
        output_path = output_dir / f"frame_{i+1:03d}.jpg"
        with open(output_path, 'wb') as f:
            f.write(encoded.tobytes())
        
        # Verify file was written correctly
        if not output_path.exists() or output_path.stat().st_size != len(encoded):
            print(f"  ❌ Frame {i+1}: File write error")
            continue
        
        saved_files.append(output_path)
        
        # 6. RECORD STATS
        total_time = t_cap + t_enc
        frame_times.append(total_time)
        encode_times.append(t_enc)
        jpeg_sizes.append(len(encoded))
        
        # 7. MAINTAIN FRAME RATE
        elapsed = time.time() - loop_start
        sleep_time = max(0, frame_period - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)
        
        actual_period = time.time() - loop_start
        actual_fps = 1.0 / actual_period if actual_period > 0 else 0
        
        if i % 10 == 0:
            print(f"  Frame {i+1:2d}: cap={t_cap:4.1f}ms enc={t_enc:4.1f}ms "
                  f"size={len(encoded)/1024:5.1f}KB fps={actual_fps:4.1f}")
    
    cap.release()
    
    # RESULTS
    print("\n" + "=" * 60)
    print("RESULTS:")
    print(f"  Frames attempted:  30")
    print(f"  Frames dropped:    {dropped_frames}")
    print(f"  Corrupt JPEGs:     {corrupt_jpegs}")
    print(f"  Dark frames:       {dark_frames}")
    print(f"  Files saved:       {len(saved_files)}")
    
    if len(frame_times) > 0:
        print(f"\n  Capture time:  {np.mean(frame_times):.1f}ms  (±{np.std(frame_times):.1f}ms)")
        print(f"                 min={np.min(frame_times):.1f}ms  max={np.max(frame_times):.1f}ms")
        print(f"                 p95={np.percentile(frame_times, 95):.1f}ms")
        
        print(f"\n  Encode time:   {np.mean(encode_times):.1f}ms  (±{np.std(encode_times):.1f}ms)")
        
        print(f"\n  JPEG size:     {np.mean(jpeg_sizes)/1024:.1f}KB  (±{np.std(jpeg_sizes)/1024:.1f}KB)")
        print(f"                 min={np.min(jpeg_sizes)/1024:.1f}KB  max={np.max(jpeg_sizes)/1024:.1f}KB")
        
        print(f"\n  Timing budget: {np.mean(frame_times):.1f}ms of 33.3ms available ({np.mean(frame_times)/33.3*100:.0f}%)")
    
    print("=" * 60)
    
    # SUCCESS CRITERIA (strict)
    success = True
    reasons = []
    
    if dropped_frames > 0:
        success = False
        reasons.append(f"Dropped {dropped_frames} frames")
    
    if corrupt_jpegs > 0:
        success = False
        reasons.append(f"{corrupt_jpegs} corrupt JPEGs")
    
    if len(frame_times) == 0:
        success = False
        reasons.append("No successful frames")
    elif np.mean(frame_times) > 25:  # Leave 8ms for network + hand tracking
        success = False
        reasons.append(f"Too slow ({np.mean(frame_times):.1f}ms > 25ms budget)")
    elif np.percentile(frame_times, 95) > 30:
        success = False
        reasons.append(f"Inconsistent timing (p95={np.percentile(frame_times, 95):.1f}ms)")
    
    if dark_frames > 5:
        print(f"\n⚠️  WARNING: {dark_frames} dark frames (camera may need more warm-up)")
    
    if success:
        print("\n PHASE 1: PASS - Camera capture validated!")
        return True
    else:
        print(f"\n❌ PHASE 1: FAIL - {', '.join(reasons)}")
        return False

if __name__ == "__main__":
    success = test_camera_capture_improved()
    exit(0 if success else 1)