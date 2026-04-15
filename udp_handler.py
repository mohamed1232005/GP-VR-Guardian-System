"""
UDP Frame Receiver — asyncio DatagramProtocol.
Receives camera frames from Unity and writes the latest into frame_queue.
"""

import asyncio
import multiprocessing as mp
import queue
import struct


_HEADER_FMT = ">III16f"   # seq(4) + ts(4) + jlen(4) + 16 floats pose(64) = 76 bytes
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)  # 76


class UDPFrameReceiver(asyncio.DatagramProtocol):
    def __init__(self, frame_queue: mp.Queue):
        self.frame_queue = frame_queue
        self.active_client_id = None   # set by TCPHandler when client connects

    def datagram_received(self, data: bytes, addr) -> None:
        if len(data) < _HEADER_SIZE:
            print(f"[UDP] discarded malformed packet from {addr}, bytes={len(data)}")
            return

        try:
            fields = struct.unpack_from(_HEADER_FMT, data, 0)
        except struct.error as exc:
            print(f"[UDP] unpack failed from {addr}: {exc}")
            return

        seq = fields[0]
        ts = fields[1]
        jlen = fields[2]
        pose = list(fields[3:])
        jpeg = data[_HEADER_SIZE:]

        if len(jpeg) != jlen:
            print(
                f"[UDP] length mismatch from {addr}: "
                f"declared={jlen}, actual={len(jpeg)}, seq={seq}"
            )

        packet = {
            "seq": seq,
            "timestamp_ms": ts,
            "pose": pose,
            "jpeg": jpeg,
            "client_id": self.active_client_id,
        }

        print(
            f"[UDP] packet from {addr} "
            f"seq={seq} ts={ts} jpeg_bytes={len(jpeg)} "
            f"client_id={self.active_client_id}"
        )

        try:
            self.frame_queue.put_nowait(packet)
        except queue.Full:
            try:
                _ = self.frame_queue.get_nowait()
            except queue.Empty:
                pass

            try:
                self.frame_queue.put_nowait(packet)
            except queue.Full:
                print("[UDP] queue still full after overwrite attempt; dropping packet")

    def error_received(self, exc: Exception) -> None:
        print(f"[UDP] error: {exc}")

    def connection_lost(self, exc) -> None:
        print(f"[UDP] connection lost: {exc}")