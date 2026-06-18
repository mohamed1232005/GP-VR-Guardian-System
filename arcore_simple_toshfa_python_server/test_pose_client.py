"""Standalone TCP test client for the Guardian Python service.

Connects to the TCP control port like Unity would, which makes the server set an
active client id and start streaming results. It decodes the framed JSON messages
(>IH header) and prints HAND_DATA / BODY_POSE / WARNING so you can validate the
wire protocol end-to-end without Unity.

Usage:
    python test_pose_client.py [host] [port]
    (defaults: 127.0.0.1 9000)

Notes:
- BODY_POSE only flows if POSE_ENABLED is true, the webcam opens, and
  pose_landmarker.task is present beside server.py.
- Press Ctrl+C to stop.
"""

import json
import socket
import struct
import sys

from config import TCP_CONTROL_PORT

# Must match transport.py byte-for-byte.
_TCP_HEADER_FMT = ">IH"
_TCP_HEADER_SIZE = struct.calcsize(_TCP_HEADER_FMT)

_TYPE_NAMES = {
    0x11: "HAND_DATA",
    0x14: "WARNING",
    0x15: "BODY_POSE",
    0x03: "STATE_CHANGE",
}


def _recv_exact(sock: socket.socket, count: int) -> bytes:
    buf = bytearray()
    while len(buf) < count:
        chunk = sock.recv(count - len(buf))
        if not chunk:
            raise ConnectionError("server closed the connection")
        buf.extend(chunk)
    return bytes(buf)


def _summarize_body_pose(payload: dict) -> str:
    tracked = payload.get("tracked")
    lms = payload.get("landmarks") or []
    ts = payload.get("frame_timestamp_ms")
    if not tracked or not lms:
        return f"tracked={tracked} landmarks=0 ts={ts}"

    rows = len(lms)
    width = len(lms[0]) if rows else 0
    sample = lms[0] if rows else []
    sample_str = ", ".join(f"{v:.3f}" for v in sample)
    return (
        f"tracked={tracked} landmarks={rows}x{width} ts={ts} "
        f"lm[0]=[{sample_str}]"
    )


def main() -> None:
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else TCP_CONTROL_PORT

    print(f"[client] connecting to {host}:{port} ...", flush=True)
    with socket.create_connection((host, port), timeout=5) as sock:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print("[client] connected. waiting for messages (Ctrl+C to stop)...", flush=True)

        counts = {}
        while True:
            header = _recv_exact(sock, _TCP_HEADER_SIZE)
            length, type_id = struct.unpack(_TCP_HEADER_FMT, header)
            body = _recv_exact(sock, length)

            try:
                payload = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError as exc:
                print(f"[client] bad JSON for type=0x{type_id:02X}: {exc}", flush=True)
                continue

            name = _TYPE_NAMES.get(type_id, f"UNKNOWN_0x{type_id:02X}")
            counts[name] = counts.get(name, 0) + 1

            if type_id == 0x15:  # BODY_POSE
                print(f"[BODY_POSE #{counts[name]}] {_summarize_body_pose(payload)}", flush=True)
            elif type_id == 0x11:  # HAND_DATA
                if counts[name] % 30 == 1:
                    n = len(payload.get("landmarks_smoothed") or [])
                    print(f"[HAND_DATA #{counts[name]}] gesture={payload.get('gesture')} landmarks={n}", flush=True)
            else:
                print(f"[{name} #{counts[name]}] {payload}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[client] stopped.", flush=True)
    except Exception as exc:  # noqa: BLE001 - test utility, surface any failure
        print(f"[client] error: {exc!r}", flush=True)
