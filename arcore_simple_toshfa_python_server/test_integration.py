"""End-to-end socket integration test for the TCP result path.

Spins up the real asyncio TCPHandler on a loopback port, connects a client like
Unity would, pushes a synthetic BODY_POSE (and HAND_DATA) through the shared
result_queue, and asserts the bytes that come back over the socket decode exactly
as Unity's BodyPoseReceiver expects. No webcam or model file required.

Run from the GP-VR-Guardian-System folder:
    .venv\\Scripts\\python.exe -m unittest test_integration -v
"""

import asyncio
import json
import queue
import struct
import unittest

import transport


async def _exchange(responses):
    """Start the server, connect a client, push `responses`, return decoded (type_id, dict) list."""
    result_q = queue.Queue()
    frame_q = queue.Queue(maxsize=1)
    udp = transport.UDPFrameReceiver(frame_q)
    handler = transport.TCPHandler(result_q, frame_q, udp)

    server = await asyncio.start_server(handler.handle_client, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    dispatcher = asyncio.create_task(handler.result_dispatcher())

    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    # Let handle_client run and set its _writer before we enqueue.
    await asyncio.sleep(0.1)

    for resp in responses:
        result_q.put({"client_id": "test", "response": resp})

    decoded = []
    try:
        for _ in responses:
            header = await asyncio.wait_for(reader.readexactly(6), timeout=2.0)
            length, type_id = struct.unpack(">IH", header)
            body = await asyncio.wait_for(reader.readexactly(length), timeout=2.0)
            decoded.append((type_id, json.loads(body.decode("utf-8"))))
    finally:
        # The real dispatcher blocks a thread on result_queue.get() via run_in_executor.
        # Cancelling the coroutine does NOT unblock that thread, and asyncio.run() waits
        # for the default executor at shutdown -> hang. Cancel, then push a sentinel so
        # the orphaned get() returns and its thread can exit cleanly.
        dispatcher.cancel()
        try:
            await dispatcher
        except (asyncio.CancelledError, Exception):
            pass
        result_q.put({"client_id": "_stop", "response": {"type": "WARNING"}})
        await asyncio.sleep(0.05)

        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
        server.close()
        await server.wait_closed()

    return decoded


class TestTcpIntegration(unittest.TestCase):
    def test_body_pose_travels_over_socket(self):
        body_pose = {
            "type": "BODY_POSE",
            "landmarks": [[0.1, 0.2, -0.3, 0.95] for _ in range(33)],
            "frame_timestamp_ms": 42,
            "tracked": True,
        }
        decoded = asyncio.run(_exchange([body_pose]))

        self.assertEqual(len(decoded), 1)
        type_id, payload = decoded[0]
        self.assertEqual(type_id, 0x15)
        self.assertTrue(payload["tracked"])
        self.assertEqual(payload["frame_timestamp_ms"], 42)
        self.assertEqual(len(payload["landmarks"]), 33)
        self.assertTrue(all(len(r) == 4 for r in payload["landmarks"]))

    def test_hand_and_body_coexist_with_correct_type_ids(self):
        """Hand and body responses must route to their distinct type ids over the same channel."""
        hand = {
            "type": "HAND_DATA",
            "gesture": "Pinch",
            "gesture_confirmed": True,
            "landmarks_smoothed": [[0.0, 0.0, 0.0] for _ in range(21)],
        }
        body = {
            "type": "BODY_POSE",
            "landmarks": [[0.0, 0.0, 0.0, 0.5] for _ in range(33)],
            "frame_timestamp_ms": 7,
            "tracked": True,
        }
        decoded = asyncio.run(_exchange([hand, body]))

        ids = [d[0] for d in decoded]
        self.assertIn(0x11, ids)  # HAND_DATA
        self.assertIn(0x15, ids)  # BODY_POSE


if __name__ == "__main__":
    unittest.main(verbosity=2)
