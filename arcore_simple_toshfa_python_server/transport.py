"""Networking transport for the active Mode 2/System A Python service.

UDP receives Unity camera frames. TCP receives small Unity control messages
and sends HAND_DATA/WARNING responses back to Unity.
"""

import asyncio
import json
import logging
import queue
import struct
import time
from typing import Any, Optional

from config import dlog, HEARTBEAT_INTERVAL_SECONDS, MAX_JPEG_BYTES, MAX_TCP_MESSAGE_BYTES

_LOGGER = logging.getLogger("guardian.transport")

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
MSG_BODY_POSE = 0x15  # Raw MediaPipe Pose landmarks from the laptop webcam.
MSG_HEARTBEAT = 0x16  # Liveness ping; payload is always "{}". Unity treats it as silent.

_TYPE_TO_ID = {
    "HAND_DATA": MSG_HAND_DATA,
    "BODY_POSE": MSG_BODY_POSE,
    "WARNING": MSG_WARNING,
    "STATE_CHANGE": MSG_STATE_CHANGE,
}

# How often (seconds) the UDP receiver logs aggregated oversized-frame drops.
_OVERSIZE_LOG_INTERVAL_SECONDS = 5.0


class TCPMessageTooLargeError(Exception):
    """A TCP header declared a payload larger than MAX_TCP_MESSAGE_BYTES."""


class UDPFrameReceiver(asyncio.DatagramProtocol):
    """Receive Unity camera frames and keep only the latest one."""

    def __init__(self, frame_queue: Any):
        self.frame_queue = frame_queue
        self.active_client_id: str | None = None
        self._packet_count = 0
        self._last_log = time.monotonic()
        self._last_seq = 0
        self._last_jpeg_bytes = 0
        self._oversize_drop_count = 0
        self._last_oversize_log = 0.0

    def datagram_received(self, data: bytes, addr) -> None:
        if len(data) < _UDP_HEADER_SIZE:
            dlog(f"[UDP] discarded malformed packet from {addr}, bytes={len(data)}")
            return

        try:
            fields = struct.unpack_from(_UDP_HEADER_FMT, data, 0)
        except struct.error as exc:
            dlog(f"[UDP] unpack failed from {addr}: {exc}")
            return

        seq = fields[0]
        timestamp_ms = fields[1]
        jpeg_len = fields[2]
        pose = list(fields[3:])
        jpeg = data[_UDP_HEADER_SIZE:]

        if jpeg_len > MAX_JPEG_BYTES:
            self._count_oversize_drop(jpeg_len)
            return

        if len(jpeg) != jpeg_len:
            dlog(
                f"[UDP] length mismatch from {addr}: declared={jpeg_len}, "
                f"actual={len(jpeg)}, seq={seq}"
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

    def _count_oversize_drop(self, jpeg_len: int) -> None:
        """Count dropped oversized datagrams; log aggregated, not per-packet."""
        self._oversize_drop_count += 1
        now = time.monotonic()
        if now - self._last_oversize_log >= _OVERSIZE_LOG_INTERVAL_SECONDS:
            _LOGGER.warning(
                "[UDP] dropped %d oversized frame(s); last declared jpeg_len=%d "
                "> MAX_JPEG_BYTES=%d",
                self._oversize_drop_count,
                jpeg_len,
                MAX_JPEG_BYTES,
            )
            self._oversize_drop_count = 0
            self._last_oversize_log = now

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
            dlog(
                f"[UDP] receiving OK packets={self._packet_count} "
                f"last_seq={self._last_seq} jpeg_bytes={self._last_jpeg_bytes} "
                f"client_id={client_id}"
            )
            self._packet_count = 0
            self._last_log = now

    def error_received(self, exc: Exception) -> None:
        _LOGGER.error("[UDP] error: %s", exc)

    def connection_lost(self, exc) -> None:
        _LOGGER.info("[UDP] connection lost: %s", exc)


class TCPHandler:
    """Handle Unity control messages and dispatch CV results to Unity."""

    def __init__(self, result_queue: Any, frame_queue: Any, udp_receiver: UDPFrameReceiver):
        self.result_queue = result_queue
        self.frame_queue = frame_queue
        self.udp_receiver = udp_receiver
        self._writer: Optional[asyncio.StreamWriter] = None
        self._client_id: Optional[str] = None
        # Serializes all outbound writes (results, control replies, heartbeats)
        # so no message can ever interleave mid-frame on the wire.
        self._send_lock = asyncio.Lock()

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
        client_id = f"{addr[0]}:{addr[1]}"

        # Closing the previous writer unblocks any stale handler still parked in
        # _read_message (dead-but-unclosed socket) so its cleanup runs promptly.
        previous_writer = self._writer
        if previous_writer is not None and not previous_writer.is_closing():
            _LOGGER.info("[TCP] new client %s replacing previous connection", client_id)
            previous_writer.close()

        self._client_id = client_id
        self._writer = writer
        self.udp_receiver.active_client_id = client_id

        _LOGGER.info("[TCP] client connected: %s", client_id)

        heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        try:
            while True:
                type_id, payload = await _read_message(reader)
                await self._dispatch_incoming(type_id, payload)
        except (asyncio.IncompleteReadError, ConnectionResetError):
            _LOGGER.info("[TCP] client disconnected: %s", client_id)
        except TCPMessageTooLargeError as exc:
            _LOGGER.error(
                "[TCP] oversized message from %s: %s -- closing connection",
                client_id,
                exc,
            )
        except (UnicodeDecodeError, ValueError) as exc:
            # Invalid UTF-8 / invalid JSON from a foreign or hostile client; Unity's
            # minimum body is '{}' so this never fires for the real app.
            _LOGGER.error(
                "[TCP] malformed payload from %s: %s -- closing connection",
                client_id,
                exc,
            )
        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
            # Only clear the shared slots if they still belong to THIS connection —
            # a stale handler must never tear down state a newer client now owns.
            if self._writer is writer:
                self._writer = None
            if self.udp_receiver.active_client_id == client_id:
                self.udp_receiver.active_client_id = None
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionResetError, OSError) as exc:
                # Windows proactor raises WinError 64 / ConnectionReset when the peer
                # (the phone) drops the link first. The socket is already gone, so this
                # is benign; swallow it so it does not surface as an unretrieved task
                # exception and alarm during normal phone connect/disconnect.
                dlog(f"[TCP] wait_closed ignored on disconnect: {exc!r}")

    async def _dispatch_incoming(self, type_id: int, payload: dict) -> None:
        if type_id == MSG_RESET:
            self._put_control_nowait({"client_id": self._client_id, "_reset_detector": True})
            await self._send(MSG_STATE_CHANGE, {"type": "STATE_CHANGE", "state": "INIT"})
            _LOGGER.info("[TCP] RESET received")
            return

        if type_id == MSG_STATE_CHANGE:
            dlog(f"[TCP] Unity state: {payload.get('state')}")
            return

        if type_id == MSG_FLOOR_CONFIRM:
            dlog("[TCP] FLOOR_CONFIRM ignored in active Mode 2/System A")
            return

        dlog(f"[TCP] ignored unknown incoming type=0x{type_id:02X}")

    async def result_dispatcher(self) -> None:
        """Drain CV results without blocking the asyncio event loop."""
        loop = asyncio.get_running_loop()
        while True:
            item = await loop.run_in_executor(None, self.result_queue.get)
            if self._writer and not self._writer.is_closing():
                response = item.get("response", {})
                type_id = _TYPE_TO_ID.get(response.get("type"), MSG_WARNING)
                await self._send(type_id, response)

    async def _heartbeat_loop(self) -> None:
        """Send MSG_HEARTBEAT to the connected client at a fixed interval.

        Started per connection by handle_client and cancelled on disconnect.
        Sends go through _send (same writer + lock as the result dispatcher)
        so a heartbeat can never interleave mid-message with HAND_DATA.
        """
        try:
            while True:
                await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)
                await self._send(MSG_HEARTBEAT, {})
        except asyncio.CancelledError:
            raise
        except (ConnectionResetError, OSError) as exc:
            dlog(f"[TCP] heartbeat stopped: {exc!r}")

    async def _send(self, type_id: int, payload: dict) -> None:
        async with self._send_lock:
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
    if length > MAX_TCP_MESSAGE_BYTES:
        # Never trust the declared length: reading it blindly would let a
        # malformed/malicious header allocate up to 4 GiB.
        raise TCPMessageTooLargeError(
            f"declared payload {length} bytes > MAX_TCP_MESSAGE_BYTES="
            f"{MAX_TCP_MESSAGE_BYTES} (type=0x{type_id:02X})"
        )
    body = await reader.readexactly(length)
    return type_id, json.loads(body.decode("utf-8"))
