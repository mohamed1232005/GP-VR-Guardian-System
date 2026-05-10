"""Networking transport for the active Mode 2/System A Python service.

UDP receives Unity camera frames. TCP receives small Unity control messages
and sends HAND_DATA/WARNING responses back to Unity.
"""

import asyncio
import json
import queue
import struct
import time
from typing import Any, Optional

# UDP frame packet format: uint32 seq, uint32 timestamp_ms,
# uint32 jpeg_length, float32 pose[16], then JPEG bytes.
_UDP_HEADER_FMT = ">III16f"
_UDP_HEADER_SIZE = struct.calcsize(_UDP_HEADER_FMT)

# TCP framed JSON format: uint32 payload_length + uint16 type_id.
_TCP_HEADER_FMT = ">IH"
_TCP_HEADER_SIZE = struct.calcsize(_TCP_HEADER_FMT)

# Type IDs: Unity -> Python.
MSG_FLOOR_CONFIRM = 0x02  # Legacy/manual-boundary flow; ignored in active mode.
MSG_STATE_CHANGE = 0x03   # Unity state log only.
MSG_RESET = 0x04

# Type IDs: Python -> Unity.
MSG_HAND_DATA = 0x11
MSG_WARNING = 0x14

_TYPE_TO_ID = {
    "HAND_DATA": MSG_HAND_DATA,
    "WARNING": MSG_WARNING,
    "STATE_CHANGE": MSG_STATE_CHANGE,
}


class UDPFrameReceiver(asyncio.DatagramProtocol):
    """Receive Unity camera frames and keep only the latest one."""

    def __init__(self, frame_queue: Any):
        self.frame_queue = frame_queue
        self.active_client_id: str | None = None
        self._packet_count = 0
        self._last_log = time.monotonic()
        self._last_seq = 0
        self._last_jpeg_bytes = 0

    def datagram_received(self, data: bytes, addr) -> None:
        if len(data) < _UDP_HEADER_SIZE:
            print(f"[UDP] discarded malformed packet from {addr}, bytes={len(data)}", flush=True)
            return

        try:
            fields = struct.unpack_from(_UDP_HEADER_FMT, data, 0)
        except struct.error as exc:
            print(f"[UDP] unpack failed from {addr}: {exc}", flush=True)
            return

        seq = fields[0]
        timestamp_ms = fields[1]
        jpeg_len = fields[2]
        pose = list(fields[3:])
        jpeg = data[_UDP_HEADER_SIZE:]

        if len(jpeg) != jpeg_len:
            print(
                f"[UDP] length mismatch from {addr}: declared={jpeg_len}, "
                f"actual={len(jpeg)}, seq={seq}",
                flush=True,
            )
            return

        client_id = self.active_client_id
        if client_id is None:
            return

        self._put_latest(
            {
                "seq": seq,
                "timestamp_ms": timestamp_ms,
                "pose": pose,
                "jpeg": jpeg,
                "client_id": client_id,
            }
        )
        self._log_status(seq, len(jpeg), client_id)

    def _put_latest(self, packet: dict) -> None:
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

    def _log_status(self, seq: int, jpeg_bytes: int, client_id: str) -> None:
        self._packet_count += 1
        self._last_seq = seq
        self._last_jpeg_bytes = jpeg_bytes

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


class TCPHandler:
    """Handle Unity control messages and dispatch CV results to Unity."""

    def __init__(self, result_queue: Any, frame_queue: Any, udp_receiver: UDPFrameReceiver):
        self.result_queue = result_queue
        self.frame_queue = frame_queue
        self.udp_receiver = udp_receiver
        self._writer: Optional[asyncio.StreamWriter] = None
        self._client_id: Optional[str] = None

    def _put_control_nowait(self, item: dict) -> None:
        """Insert a high-priority control item into the latest-frame queue."""
        try:
            self.frame_queue.put_nowait(item)
        except queue.Full:
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
            self.frame_queue.put_nowait(item)

    async def handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        addr = writer.get_extra_info("peername")
        self._client_id = f"{addr[0]}:{addr[1]}"
        self._writer = writer
        self.udp_receiver.active_client_id = self._client_id

        print(f"[TCP] client connected: {self._client_id}", flush=True)

        try:
            while True:
                type_id, payload = await _read_message(reader)
                await self._dispatch_incoming(type_id, payload)
        except (asyncio.IncompleteReadError, ConnectionResetError):
            print(f"[TCP] client disconnected: {self._client_id}", flush=True)
        finally:
            self._writer = None
            self.udp_receiver.active_client_id = None
            writer.close()
            await writer.wait_closed()

    async def _dispatch_incoming(self, type_id: int, payload: dict) -> None:
        if type_id == MSG_RESET:
            self._put_control_nowait({"client_id": self._client_id, "_reset_detector": True})
            await self._send(MSG_STATE_CHANGE, {"type": "STATE_CHANGE", "state": "INIT"})
            print("[TCP] RESET received", flush=True)
            return

        if type_id == MSG_STATE_CHANGE:
            print(f"[TCP] Unity state: {payload.get('state')}", flush=True)
            return

        if type_id == MSG_FLOOR_CONFIRM:
            print("[TCP] FLOOR_CONFIRM ignored in active Mode 2/System A", flush=True)
            return

        print(f"[TCP] ignored unknown incoming type=0x{type_id:02X}", flush=True)

    async def result_dispatcher(self) -> None:
        """Drain CV results without blocking the asyncio event loop."""
        loop = asyncio.get_running_loop()
        while True:
            item = await loop.run_in_executor(None, self.result_queue.get)
            if self._writer and not self._writer.is_closing():
                response = item.get("response", {})
                type_id = _TYPE_TO_ID.get(response.get("type"), MSG_WARNING)
                await self._send(type_id, response)

    async def _send(self, type_id: int, payload: dict) -> None:
        if self._writer and not self._writer.is_closing():
            self._writer.write(_pack_message(type_id, payload))
            await self._writer.drain()


def _pack_message(type_id: int, payload: dict) -> bytes:
    body = json.dumps(payload).encode("utf-8")
    header = struct.pack(_TCP_HEADER_FMT, len(body), type_id)
    return header + body


async def _read_message(reader: asyncio.StreamReader) -> tuple[int, dict]:
    header = await reader.readexactly(_TCP_HEADER_SIZE)
    length, type_id = struct.unpack(_TCP_HEADER_FMT, header)
    body = await reader.readexactly(length)
    return type_id, json.loads(body.decode("utf-8"))
