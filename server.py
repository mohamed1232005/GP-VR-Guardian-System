"""
Guardian Server — production entry point.
Runs UDP, TCP, and CV worker with visible crash reporting.

Final hand-skeleton fix:
- Uses a supervised thread instead of a silent multiprocessing child.
- Ensures CV logs appear in the same terminal.
- Keeps the latest-frame queue behaviour.
"""

import asyncio
import queue
import threading
import traceback

from config import TCP_CONTROL_PORT, UDP_FRAME_PORT
from cv_worker import run_cv_worker
from tcp_handler import TCPHandler
from udp_handler import UDPFrameReceiver


async def main() -> None:
    frame_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue()

    def guarded_cv_worker() -> None:
        try:
            run_cv_worker(frame_queue, result_queue)
        except BaseException:
            print("[CV] FATAL: CV worker crashed:", flush=True)
            traceback.print_exc()

    cv_thread = threading.Thread(
        target=guarded_cv_worker,
        name="GuardianCVWorker",
        daemon=True,
    )
    cv_thread.start()
    print("[*] CV worker thread started", flush=True)

    udp_receiver = UDPFrameReceiver(frame_queue)
    tcp_handler = TCPHandler(result_queue, frame_queue, udp_receiver)

    loop = asyncio.get_running_loop()

    await loop.create_datagram_endpoint(
        lambda: udp_receiver,
        local_addr=("0.0.0.0", UDP_FRAME_PORT),
    )
    print(f"[*] UDP listening on :{UDP_FRAME_PORT}", flush=True)

    tcp_server = await asyncio.start_server(
        tcp_handler.handle_client,
        "0.0.0.0",
        TCP_CONTROL_PORT,
    )
    print(f"[*] TCP listening on :{TCP_CONTROL_PORT}", flush=True)

    asyncio.create_task(tcp_handler.result_dispatcher())

    async with tcp_server:
        await tcp_server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())