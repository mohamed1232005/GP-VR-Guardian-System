# ===========================================================================
# networking/udp_receiver.py — UDP Frame Receiver (Phase 4)
# Listens for Unity frame packets on UDP, decodes JPEG, logs metadata.
# Phase 4: Pushes decoded frames to depth worker (non-blocking).
# Runs in its own thread.
# ===========================================================================

import socket
import threading
import numpy as np
import cv2

from networking.protocol import parse_frame_packet
from debug.fps_meter import FPSMeter


class UDPReceiver:
    """Receives and decodes Unity AR camera frames over UDP."""

    def __init__(self, host: str, port: int, max_packet_size: int = 65535):
        self.host = host
        self.port = port
        self.max_packet_size = max_packet_size

        self._socket = None
        self._thread = None
        self._running = False
        self._fps_meter = FPSMeter("udp_frames")

        # Latest decoded frame (thread-safe access)
        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_metadata = None
        self._frame_count = 0

        # Phase 4: Optional depth worker reference
        self._depth_worker = None

    def set_depth_worker(self, depth_worker):
        """
        Attach a depth worker to receive frames for depth estimation.
        Called after construction, before or after start().
        """
        self._depth_worker = depth_worker
        print("[P4_UDP] depth worker attached to UDP receiver")

    def start(self):
        """Start the UDP receiver thread."""
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((self.host, self.port))
        self._socket.settimeout(1.0)  # 1s timeout for clean shutdown
        self._running = True

        self._thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._thread.start()

        print(f"[P2_UDP] listening on {self.host}:{self.port}")

    def stop(self):
        """Stop the UDP receiver."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        if self._socket:
            self._socket.close()
            self._socket = None
        print(f"[P2_UDP] stopped. Total frames={self._frame_count}")

    def get_latest(self) -> tuple:
        """
        Get the latest decoded frame and metadata.
        Returns (frame: np.ndarray or None, metadata: dict or None)
        """
        with self._lock:
            return self._latest_frame, self._latest_metadata

    def _receive_loop(self):
        """Main receive loop running in background thread."""
        while self._running:
            try:
                data, addr = self._socket.recvfrom(self.max_packet_size)
            except socket.timeout:
                continue
            except OSError:
                if self._running:
                    print("[P2_UDP] socket error")
                break

            # Parse packet
            metadata, jpeg_bytes = parse_frame_packet(data)
            if metadata is None or jpeg_bytes is None or len(jpeg_bytes) == 0:
                continue

            # Decode JPEG with OpenCV
            jpeg_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(jpeg_array, cv2.IMREAD_COLOR)

            if frame is None:
                print("[P2_UDP] JPEG decode failed")
                continue

            # Phase 2D [H_FRAME_RECV]: prove frames arrive with the rotation metadata
            # in horizontal mode (throttled ~every 15 frames). Diagnostic only.
            if self._frame_count % 15 == 1:
                print(f"[H_FRAME_RECV] frameId={metadata.get('frame_id','?')} "
                      f"bytes={len(jpeg_bytes)} "
                      f"metaRotation={metadata.get('input_rotation_degrees','?')} "
                      f"image={frame.shape[1]}x{frame.shape[0]}")

            # Store latest frame
            with self._lock:
                self._latest_frame = frame
                self._latest_metadata = metadata
                self._frame_count += 1

            # Phase 4: Push to depth worker (non-blocking)
            if self._depth_worker is not None:
                self._depth_worker.push_frame(frame, metadata)

            # Update FPS
            self._fps_meter.tick()

            # Log periodically
            if self._frame_count % 30 == 1:
                frame_id = metadata.get("frame_id", "?")
                fx = metadata.get("fx", "?")
                fy = metadata.get("fy", "?")
                pose = metadata.get("camera_to_world", [])
                pose_len = len(pose) if isinstance(pose, list) else 0
                print(
                    f"[P2_UDP] frame={frame_id} shape={frame.shape} "
                    f"jpeg={len(jpeg_bytes)} fx={fx} fy={fy} poseLen={pose_len}"
                )
