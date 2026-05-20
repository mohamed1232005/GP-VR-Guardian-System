# ===========================================================================
# networking/tcp_server.py — TCP Server for AI responses (Phase 7C)
# Sends AI_GUARDIAN_DATA, AI_HAND_DATA, and AI_GUARDIAN_STATUS to Unity.
#
# Phase 7C: Uses a per-type message queue so AI_GUARDIAN_DATA and
# AI_HAND_DATA don't overwrite each other. Both are drained and sent
# to Unity every tick.
# ===========================================================================

import socket
import threading
import json
import time
import collections

import numpy as np

from models.dummy_guardian import DummyGuardian


# ---------------------------------------------------------------------------
# JSON serialisation safety
# ---------------------------------------------------------------------------
def make_json_safe(obj):
    """
    Recursively convert numpy scalars/arrays and Python tuples to native
    Python types so json.dumps() never raises
    "Object of type int64/float32/bool_/ndarray is not JSON serializable".

    Handles:
        np.integer  → int
        np.floating → float
        np.bool_    → bool
        np.ndarray  → list  (nested arrays become nested lists)
        tuple       → list
        dict        → dict  (keys and values both sanitised)
        list        → list  (elements sanitised)
    All other types are returned unchanged (json.dumps handles them).
    """
    if isinstance(obj, dict):
        return {make_json_safe(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [make_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


class TCPServer:
    """TCP server that sends AI results to Unity.

    Phase 9.4: Post-lock freeze — after sending a locked guardian packet,
    all subsequent guardian geometry is frozen. Only hand data passes through.
    This prevents stale boundary packets from causing drift.
    """

    def __init__(
        self,
        host: str,
        port: int,
        send_interval: float = 1.0,
        boundary_mode: str = "ai_floor",
        allow_dummy_guardian: bool = False,
    ):
        self.host = host
        self.port = port
        self.send_interval = send_interval
        self._boundary_mode = boundary_mode
        self._allow_dummy_guardian = allow_dummy_guardian

        self._server_socket = None
        self._thread = None
        self._running = False
        self._counter = 0
        self._dummy_guardian = DummyGuardian()

        # Phase 7C: Per-type message slots (thread-safe)
        self._result_lock = threading.Lock()
        self._latest_guardian = None   # AI_GUARDIAN_DATA or AI_GUARDIAN_STATUS
        self._latest_hand    = None   # AI_HAND_DATA
        self._guardian_ready = False
        self._hand_ready     = False

        # Phase 9.4: Post-lock freeze state
        self._post_lock_frozen = False
        self._frozen_lock_packet = None  # The one final locked packet
        self._post_lock_ignored_count = 0

        # Legacy compat
        self._latest_ai_result = None
        self._result_ready = False

        # Track connected client for pushing
        self._client_socket = None
        self._client_lock = threading.Lock()

    def push_ai_result(self, result: dict):
        """
        Push a new AI pipeline result to be sent to Unity.
        Called by AIPipelineWorker. Thread-safe, non-blocking.

        Phase 9.4: After lock, freeze guardian geometry — only hand data passes.
        """
        msg_type = result.get("type", "")
        with self._result_lock:
            if msg_type == "AI_HAND_DATA":
                # Hand data ALWAYS passes through, even after lock
                self._latest_hand = result
                self._hand_ready = True
            else:
                # Phase 9.4: Check if this is the lock packet
                boundary_state = result.get("boundary_state", "")
                if boundary_state == "locked" and not self._post_lock_frozen:
                    # First locked packet — freeze it and mark post-lock
                    self._post_lock_frozen = True
                    self._frozen_lock_packet = result
                    self._latest_guardian = result
                    self._guardian_ready = True
                    frame_id = result.get("frame_id", "?")
                    w = result.get("debug", {}).get("boundary_width", 0)
                    d = result.get("debug", {}).get("boundary_depth", 0)
                    center_x = result.get("debug", {}).get("center_x", 0)
                    center_z = result.get("debug", {}).get("center_z", 0)
                    print(f"[P9_LOCK_PACKET] frame={frame_id} "
                          f"center=({center_x},{center_z}) w={w} d={d}")
                    print(f"[P9_LOCK_FREEZE] guardian geometry frozen — "
                          f"only hand data will be sent from now on")
                elif self._post_lock_frozen:
                    # After lock — IGNORE all guardian geometry
                    self._post_lock_ignored_count += 1
                    frame_id = result.get("frame_id", "?")
                    if self._post_lock_ignored_count % 20 == 1:
                        print(f"[P9_POST_LOCK_IGNORE_GEOM] frame={frame_id} "
                              f"total_ignored={self._post_lock_ignored_count}")
                    return  # Do NOT update guardian slot
                else:
                    # Pre-lock — normal flow
                    self._latest_guardian = result
                    self._guardian_ready = True

            # Legacy compat
            self._latest_ai_result = result
            self._result_ready = True

    def start(self):
        """Start the TCP server thread."""
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((self.host, self.port))
        self._server_socket.listen(1)
        self._server_socket.settimeout(1.0)
        self._running = True

        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()

        print(f"[P2_TCP] listening on {self.host}:{self.port}")

    def stop(self):
        """Stop the TCP server."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        if self._server_socket:
            self._server_socket.close()
            self._server_socket = None
        print("[P2_TCP] stopped")

    def _accept_loop(self):
        """Accept connections and handle clients."""
        while self._running:
            try:
                client_socket, addr = self._server_socket.accept()
                print(f"[P2_TCP] Unity connected from {addr}")
                self._handle_client(client_socket)
            except socket.timeout:
                continue
            except OSError:
                if self._running:
                    print("[P2_TCP] accept error")
                break

    def _handle_client(self, client_socket: socket.socket):
        """Send AI results to Unity based on boundary mode."""
        try:
            client_socket.settimeout(1.0)

            with self._client_lock:
                self._client_socket = client_socket

            while self._running:
                self._counter += 1
                sent_count = 0

                if self._boundary_mode == "ai_floor":
                    # Phase 7C: Send BOTH guardian and hand data if available
                    sent_count = self._send_all_ai_messages(client_socket)
                elif self._boundary_mode == "dummy":
                    # Phase 3 dummy mode (debug only)
                    if self._allow_dummy_guardian:
                        message = self._get_dummy_message()
                    else:
                        message = {
                            "type": "AI_GUARDIAN_STATUS",
                            "floor_valid": False,
                            "message": "Dummy guardian disabled. "
                                       "Set ALLOW_DUMMY_GUARDIAN=True",
                        }
                    self._send_message(client_socket, message)
                    sent_count = 1
                else:
                    message = {
                        "type": "AI_GUARDIAN_STATUS",
                        "floor_valid": False,
                        "message": f"Unknown boundary mode: "
                                   f"{self._boundary_mode}",
                    }
                    self._send_message(client_socket, message)
                    sent_count = 1

                time.sleep(self.send_interval)

        except Exception as e:
            print(f"[P2_TCP] client handler error: {e}")
        finally:
            with self._client_lock:
                self._client_socket = None
            try:
                client_socket.close()
            except Exception:
                pass
            print("[P2_TCP] client connection closed")

    def _send_all_ai_messages(self, client_socket) -> int:
        """Phase 7C: Drain both guardian and hand slots, send all available.

        Returns number of messages sent this tick.
        """
        guardian_msg = None
        hand_msg = None

        with self._result_lock:
            if self._guardian_ready and self._latest_guardian is not None:
                guardian_msg = self._latest_guardian
                self._guardian_ready = False
            if self._hand_ready and self._latest_hand is not None:
                hand_msg = self._latest_hand
                self._hand_ready = False

        sent = 0

        # Send guardian data first
        if guardian_msg is not None:
            self._send_message(client_socket, guardian_msg)
            sent += 1

        # Send hand data second
        if hand_msg is not None:
            self._send_message(client_socket, hand_msg)
            sent += 1

        if sent == 0:
            # No result yet — send scanning status
            self._send_message(client_socket, {
                "type": "AI_GUARDIAN_STATUS",
                "floor_valid": False,
                "message": "Python connected. Depth model active. "
                           "Guardian boundary mode active.",
            })
            sent = 1

        return sent

    def _get_dummy_message(self) -> dict:
        """Get alternating dummy messages (Phase 3 style)."""
        if self._counter % 2 == 1:
            return self._dummy_guardian.generate()
        else:
            return {
                "type": "AI_DUMMY_DATA",
                "counter": self._counter,
                "floor_valid": False,
                "message": "Python connected. Depth model active. "
                           "Guardian boundary mode active.",
            }

    def _send_message(self, client_socket, message: dict):
        """Send a JSON message to Unity over TCP."""
        # Sanitise all numpy / tuple types so json.dumps never crashes with
        # "Object of type int64/float32/ndarray is not JSON serializable"
        safe_message = make_json_safe(message)
        json_line = json.dumps(safe_message) + "\n"
        try:
            client_socket.sendall(json_line.encode("utf-8"))
        except (BrokenPipeError, ConnectionResetError, OSError):
            print("[P2_TCP] Unity disconnected")
            raise

        # Log periodically
        msg_type = message.get("type", "?")
        if self._counter % 10 == 1 or self._counter <= 3:
            if msg_type == "AI_GUARDIAN_DATA":
                source = message.get("source", "?")
                fid = message.get("frame_id", "?")
                conf = message.get("confidence", "?")
                print(f"[P2_TCP_SEND] {msg_type} source={source} "
                      f"frame={fid} confidence={conf}")
            elif msg_type == "AI_HAND_DATA":
                fid = message.get("frame_id", "?")
                hand = message.get("handedness", "?")
                valid_3d = message.get("valid_3d", "?")
                print(f"[P2_TCP_SEND] {msg_type} frame={fid} "
                      f"hand={hand} valid_3d={valid_3d}")
            else:
                floor_valid = message.get("floor_valid", "?")
                msg_text = message.get("message", "")[:60]
                print(f"[P2_TCP_SEND] {msg_type} "
                      f"floor_valid={floor_valid} msg={msg_text}")