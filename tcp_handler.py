"""
TCP Control Handler — asyncio.
Handles FLOOR_CONFIRM, STATE_CHANGE, RESET from Unity.
Dispatches HAND_DATA, POINT_ADDED, GUARDIAN_READY, WARNING back to Unity.
Wire format: [4-byte payload_len][2-byte type_id][N-byte UTF-8 JSON]
"""

import asyncio
import json
import struct
import multiprocessing as mp
from typing import Optional
import queue

# Type IDs (Unity -> Python)
MSG_FLOOR_CONFIRM = 0x02
MSG_STATE_CHANGE  = 0x03
MSG_RESET         = 0x04

# Type IDs (Python -> Unity)
MSG_HAND_DATA      = 0x11
MSG_POINT_ADDED    = 0x12
MSG_GUARDIAN_READY = 0x13
MSG_WARNING        = 0x14

_TYPE_TO_ID = {
    "HAND_DATA"     : MSG_HAND_DATA,
    "POINT_ADDED"   : MSG_POINT_ADDED,
    "GUARDIAN_READY": MSG_GUARDIAN_READY,
    "WARNING"       : MSG_WARNING,
}

_HEADER_FMT  = ">IH"   # uint32 len + uint16 type_id
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)  # 6 bytes


def _pack_message(type_id: int, payload: dict) -> bytes:
    body = json.dumps(payload).encode("utf-8")
    hdr  = struct.pack(_HEADER_FMT, len(body), type_id)
    return hdr + body


async def _read_message(reader: asyncio.StreamReader) -> tuple[int, dict] | None:
    hdr_bytes = await reader.readexactly(_HEADER_SIZE)
    length, type_id = struct.unpack(_HEADER_FMT, hdr_bytes)
    body = await reader.readexactly(length)
    return type_id, json.loads(body.decode("utf-8"))


class TCPHandler:
    def __init__(
        self,
        result_queue : mp.Queue,
        frame_queue  : mp.Queue,
        udp_receiver,           # UDPFrameReceiver instance (to set client_id)
    ):
        self.result_queue = result_queue
        self.frame_queue  = frame_queue
        self.udp_receiver = udp_receiver
        # active writer for dispatching results back to Unity
        self._writer: Optional[asyncio.StreamWriter] = None
        self._client_id: Optional[str] = None

    # ------------------------------------------------------------------
    

    def _put_control_nowait(self, item: dict) -> None:
        """
        Put an important TCP control message into frame_queue.

        frame_queue has maxsize=1 and is also used by UDP frames.
        If the queue is full, remove the old item first. This is safe because
        a TCP control message such as FLOOR_CONFIRM or RESET is more important
        than an old camera frame.
        """
        
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
        self._writer    = writer
        self.udp_receiver.active_client_id = self._client_id

        print(f"[TCP] client connected: {self._client_id}")

        try:
            while True:
                type_id, payload = await _read_message(reader)
                await self._dispatch_incoming(type_id, payload)
        except (asyncio.IncompleteReadError, ConnectionResetError):
            print(f"[TCP] client disconnected: {self._client_id}")
        finally:
            self._writer = None
            self.udp_receiver.active_client_id = None
            writer.close()
            await writer.wait_closed()

    # ------------------------------------------------------------------
    async def _dispatch_incoming(self, type_id: int, payload: dict) -> None:
        if type_id == MSG_FLOOR_CONFIRM:
            # Push floor data into CV worker via frame_queue sentinel.
            # This must not be silently dropped if a UDP frame is already queued.
            self._put_control_nowait({
                "client_id"   : self._client_id,
                "_floor_data" : payload,
            })

            # ACK with STATE_CHANGE
            ack = {"type": "STATE_CHANGE", "state": "PLACING_POINTS"}
            await self._send(MSG_STATE_CHANGE, ack)
            print(f"[TCP] FLOOR_CONFIRM received — y={payload.get('floor_y_world')}")

        elif type_id == MSG_RESET:
            # Push reset override into CV worker.
            # This must not be silently dropped if a UDP frame is already queued.
            self._put_control_nowait({
                "client_id"      : self._client_id,
                "_state_override": "INIT",
            })

            ack = {"type": "STATE_CHANGE", "state": "INIT"}
            await self._send(MSG_STATE_CHANGE, ack)
            print("[TCP] RESET received")

        elif type_id == MSG_STATE_CHANGE:
            # Unity informing us of its state — log only
            print(f"[TCP] Unity state: {payload.get('state')}")

    # ------------------------------------------------------------------
    async def result_dispatcher(self) -> None:
        """Coroutine: drain result_queue and send responses back to Unity."""
        loop = asyncio.get_event_loop()
        while True:
            # result_queue is a multiprocessing.Queue — poll without blocking
            # the event loop by offloading to a thread executor.
            item = await loop.run_in_executor(None, self.result_queue.get)
            if self._writer and not self._writer.is_closing():
                resp    = item["response"]
                type_id = _TYPE_TO_ID.get(resp.get("type"), MSG_WARNING)
                await self._send(type_id, resp)

    # ------------------------------------------------------------------
    async def _send(self, type_id: int, payload: dict) -> None:
        if self._writer and not self._writer.is_closing():
            self._writer.write(_pack_message(type_id, payload))
            await self._writer.drain()
            
    