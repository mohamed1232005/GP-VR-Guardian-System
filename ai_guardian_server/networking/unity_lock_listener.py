"""
unity_lock_listener.py — Phase 10.2

Tiny UDP listener that receives Unity→Python lock-accepted notifications and
exposes the latest one via `get_last_lock()`. Used by server.py / metrics so
the session log can record `lock_frame_id` (which only Unity knows for sure).

Protocol: one JSON object per packet, e.g.
    {"type":"UNITY_LOCK_ACCEPTED","frame_id":237,"width":1.92,"depth":1.85,
     "floorY":-0.18,"unity_time":12.43}

Design notes:
    - One daemon thread, recv timeout 0.5s — clean shutdown via stop().
    - Failures NEVER crash the host — every exception is caught + logged.
    - No reply / ack; sender is fire-and-forget UDP.
    - Port default 9991, separate from frame UDP and Toshfa UDP 5566.
"""
from __future__ import annotations

import json
import socket
import threading
import time
from typing import Optional


class UnityLockListener:
    def __init__(self, port: int = 9991, host: str = "0.0.0.0"):
        self.port = port
        self.host = host
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._last_lock: Optional[dict] = None
        # Phase 10.6: when Unity sends HAND_PIPELINE_OFF (after the Toshfa
        # button is clicked) we stop running MediaPipe Hand on the phone-camera
        # frames. The Toshfa Pose session on the laptop is the only CV source
        # the patient needs from that point.
        self._hand_off = False
        # Phase 3: Unity tells us the lobby is built so we can drop to post-lock
        # (stop floor models → free GPU for the hand) even if the lock ACK packet
        # was lost.
        self._lobby_ready = False
        # Phase 2L (req 4): one-shot MANUAL lock request. Set when Unity sends
        # {"type":"UNITY_REQUEST_LOCK"} (the "Lock Now" button); consumed by the
        # pipeline on the next frame so it locks the current accumulated preview
        # (subject to the safety checks). Edge-triggered: consume_manual_lock()
        # returns True exactly once per request.
        self._manual_lock_req = False
        self._lock = threading.Lock()

    @property
    def hand_pipeline_off(self) -> bool:
        with self._lock:
            return self._hand_off

    @property
    def lobby_ready(self) -> bool:
        with self._lock:
            return self._lobby_ready

    @property
    def manual_lock_pending(self) -> bool:
        """Non-consuming peek: True while a manual-lock request is latched. The
        request persists (so 'Lock Now' means 'lock as soon as it is safe') until
        consume_manual_lock() clears it on a successful lock."""
        with self._lock:
            return self._manual_lock_req

    def consume_manual_lock(self) -> bool:
        """Clear the latched manual-lock request. Returns whether one was pending."""
        with self._lock:
            if self._manual_lock_req:
                self._manual_lock_req = False
                return True
            return False

    def start(self) -> bool:
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._sock.bind((self.host, self.port))
            self._sock.settimeout(0.5)
        except OSError as e:
            print(f"[U_LOCK_LISTEN_ERR] cannot bind {self.host}:{self.port}: {e}")
            self._sock = None
            return False
        self._running = True
        self._thread = threading.Thread(
            target=self._recv_loop, name="UnityLockListener", daemon=True)
        self._thread.start()
        print(f"[U_LOCK_LISTEN] listening on {self.host}:{self.port}")
        return True

    def stop(self) -> None:
        self._running = False
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def get_last_lock(self) -> Optional[dict]:
        with self._lock:
            return dict(self._last_lock) if self._last_lock is not None else None

    # ----- internals -----
    def _recv_loop(self) -> None:
        while self._running and self._sock is not None:
            try:
                data, _addr = self._sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break
            except Exception as e:
                print(f"[U_LOCK_LISTEN_RX_ERR] {type(e).__name__}: {e}")
                time.sleep(0.1)
                continue
            try:
                text = data.decode("utf-8")
                obj = json.loads(text)
                t = obj.get("type")
                if t == "UNITY_LOCK_ACCEPTED":
                    with self._lock:
                        self._last_lock = obj
                        self._last_lock["python_time"] = time.time()
                    print(f"[U_LOCK_RECV] {text}")
                    print(f"[UNITY_LOCK_ACK] frame={obj.get('frame_id')} "
                          f"size=({obj.get('width')}x{obj.get('depth')}) "
                          f"floorY={obj.get('floorY')}")
                elif t == "HAND_PIPELINE_OFF":
                    with self._lock:
                        self._hand_off = True
                    print(f"[HAND_PIPELINE_OFF] received from Unity — stopping MediaPipe Hand")
                elif t in ("UNITY_LOBBY_READY", "LOBBY_READY"):
                    with self._lock:
                        self._lobby_ready = True
                    print(f"[UNITY_LOBBY_READY] received from Unity — lobby built, dropping to post-lock")
                elif t in ("UNITY_REQUEST_LOCK", "REQUEST_LOCK", "MANUAL_LOCK"):
                    with self._lock:
                        self._manual_lock_req = True
                    print(f"[MANUAL_LOCK_REQUEST] received from Unity — will lock current preview if sane")
                # Unknown types are silently routed — forward-compatible.
            except Exception as e:
                print(f"[U_LOCK_PARSE_ERR] {type(e).__name__}: {e}")
