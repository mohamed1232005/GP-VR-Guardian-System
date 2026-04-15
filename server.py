"""
Guardian Server — entry point.
Starts the CV worker process and the asyncio event loop.
"""

import asyncio
import multiprocessing as mp

from config      import TCP_CONTROL_PORT, UDP_FRAME_PORT
from cv_worker   import run_cv_worker
from tcp_handler import TCPHandler
from udp_handler import UDPFrameReceiver


async def main() -> None:
    frame_queue  = mp.Queue(maxsize=1)
    result_queue = mp.Queue()

    # Spawn CV worker in a separate OS process (own GIL)
    cv_proc = mp.Process(
        target=run_cv_worker,
        args=(frame_queue, result_queue),
        daemon=True,
    )
    cv_proc.start()
    print(f"[*] CV worker PID: {cv_proc.pid}")

    udp_receiver = UDPFrameReceiver(frame_queue)
    tcp_handler  = TCPHandler(result_queue, frame_queue, udp_receiver)

    loop = asyncio.get_event_loop()

    # Bind UDP (fire-and-forget frames)
    await loop.create_datagram_endpoint(
        lambda: udp_receiver,
        local_addr=("0.0.0.0", UDP_FRAME_PORT),
    )
    print(f"[*] UDP listening on :{UDP_FRAME_PORT}")

    # Bind TCP (control messages)
    tcp_server = await asyncio.start_server(
        tcp_handler.handle_client,
        "0.0.0.0",
        TCP_CONTROL_PORT,
    )
    print(f"[*] TCP listening on :{TCP_CONTROL_PORT}")

    # Dispatch CV results back to Unity concurrently
    asyncio.ensure_future(tcp_handler.result_dispatcher())

    async with tcp_server:
        await tcp_server.serve_forever()


if __name__ == "__main__":
    mp.set_start_method("spawn")   # required on macOS / Windows; safe on Linux
    asyncio.run(main())