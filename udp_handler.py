"""
UDP Frame Receiver.
Receives camera frames from Unity and writes only the latest into frame_queue.
"""

import asyncio
import queue
import struct
import time


_HEADER_FMT = ">III16f"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)


class UDPFrameReceiver(asyncio.DatagramProtocol):
    def __init__(self, frame_queue):
        self.frame_queue = frame_queue
        self.active_client_id = None
        self._packet_count = 0
        self._last_log = time.monotonic()
        self._last_seq = 0
        self._last_jpeg_bytes = 0

    def datagram_received(self, data: bytes, addr) -> None:
        if len(data) < _HEADER_SIZE:
            print(f"[UDP] discarded malformed packet from {addr}, bytes={len(data)}", flush=True)
            return

        try:
            fields = struct.unpack_from(_HEADER_FMT, data, 0)
        except struct.error as exc:
            print(f"[UDP] unpack failed from {addr}: {exc}", flush=True)
            return

        seq = fields[0]
        ts = fields[1]
        jlen = fields[2]
        pose = list(fields[3:])
        jpeg = data[_HEADER_SIZE:]

        if len(jpeg) != jlen:
            print(
                f"[UDP] length mismatch from {addr}: declared={jlen}, actual={len(jpeg)}, seq={seq}",
                flush=True,
            )
            return

        client_id = self.active_client_id
        if client_id is None:
            return

        packet = {
            "seq": seq,
            "timestamp_ms": ts,
            "pose": pose,
            "jpeg": jpeg,
            "client_id": client_id,
        }

        try:
            self.frame_queue.put_nowait(packet)
        except queue.Full:
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.frame_queue.put_nowait(packet)
            except queue.Full:
                return

        self._packet_count += 1
        self._last_seq = seq
        self._last_jpeg_bytes = len(jpeg)

        now = time.monotonic()
        if now - self._last_log >= 2.0:
            print(
                f"[UDP] receiving OK packets={self._packet_count} "
                f"last_seq={self._last_seq} jpeg_bytes={self._last_jpeg_bytes} "
                f"client_id={client_id}",
                flush=True,
            )
            self._packet_count = 0
            self._last_log = now

    def error_received(self, exc: Exception) -> None:
        print(f"[UDP] error: {exc}", flush=True)

    def connection_lost(self, exc) -> None:
        print(f"[UDP] connection lost: {exc}", flush=True)