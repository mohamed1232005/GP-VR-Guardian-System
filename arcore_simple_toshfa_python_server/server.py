"""Guardian Python hand-tracking service entry point."""

import asyncio
import logging
import logging.handlers
import os
import queue
import signal
import threading
import time

from config import (
    DEBUG_LOG,
    FRAME_QUEUE_MAXSIZE,
    LOG_FILE_BACKUP_COUNT,
    LOG_FILE_MAX_BYTES,
    LOG_FILE_NAME,
    POSE_CAMERA_INDEX,
    POSE_ENABLED,
    POSE_FRAME_QUEUE_MAXSIZE,
    POSE_MODEL_FILENAME,
    POSE_PREVIEW,
    TCP_CONTROL_PORT,
    UDP_FRAME_PORT,
    WORKER_RESTART_DELAY_SECONDS,
)
from hand_tracking import run_cv_worker
from pose_tracking import (
    preview_close,
    preview_open,
    preview_tick,
    run_pose_capture,
    run_pose_worker,
)
from transport import TCPHandler, UDPFrameReceiver

_LOGGER = logging.getLogger("guardian.server")


def setup_logging() -> None:
    """Attach console + rotating file handlers to the shared guardian logger.

    Everything (including dlog/debug traffic from transport and the CV
    workers) routes through the "guardian" logger hierarchy, so one setup
    covers all modules. Safe to call once at startup; idempotent.
    """
    root = logging.getLogger("guardian")
    if root.handlers:
        return

    root.setLevel(logging.DEBUG if DEBUG_LOG else logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    # delay=True defers opening the file to the first emit, and the try/except keeps a locked
    # log file (OneDrive sync holds transient exclusive handles on this folder) from killing
    # the whole server before any socket opens — console logging continues either way.
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), LOG_FILE_NAME)
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=LOG_FILE_MAX_BYTES,
            backupCount=LOG_FILE_BACKUP_COUNT,
            encoding="utf-8",
            delay=True,
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError as exc:
        root.warning("[LOG] file logging disabled (%s): %s", log_path, exc)


def _run_supervised(worker_name: str, worker_fn, *args) -> None:
    """Run a worker function forever, rebuilding it after any crash.

    Each worker function constructs its own detector/recognizer internally, so
    re-invoking it after a crash re-creates those objects fresh. Previously an
    unhandled exception killed the daemon thread silently and hand tracking
    just stopped; now the full traceback is logged and the worker restarts
    after WORKER_RESTART_DELAY_SECONDS.
    """
    while True:
        try:
            worker_fn(*args)
            _LOGGER.error(
                "[%s] worker returned unexpectedly; restarting in %.1fs",
                worker_name,
                WORKER_RESTART_DELAY_SECONDS,
            )
        except BaseException:
            _LOGGER.exception(
                "[%s] worker crashed; restarting in %.1fs",
                worker_name,
                WORKER_RESTART_DELAY_SECONDS,
            )
        time.sleep(WORKER_RESTART_DELAY_SECONDS)


async def _preview_loop() -> None:
    """Pump the OpenCV preview window on the main (asyncio) thread — reliable on Windows."""
    if not preview_open():
        return
    try:
        while True:
            if not preview_tick():
                break
            await asyncio.sleep(0.03)
    except asyncio.CancelledError:
        pass
    finally:
        preview_close()


async def main() -> None:
    frame_queue = queue.Queue(maxsize=FRAME_QUEUE_MAXSIZE)
    result_queue = queue.Queue()
    pose_preview_active = False

    # Build the UDP receiver first so the (independent) pose pipeline can read
    # the active Unity client id and share the same TCP writer.
    udp_receiver = UDPFrameReceiver(frame_queue)
    tcp_handler = TCPHandler(result_queue, frame_queue, udp_receiver)

    cv_thread = threading.Thread(
        target=_run_supervised,
        args=("CV", run_cv_worker, frame_queue, result_queue),
        name="GuardianCVWorker",
        daemon=True,
    )
    cv_thread.start()
    _LOGGER.info("[*] CV worker thread started (supervised)")

    if POSE_ENABLED:
        # Pre-check the model file so a missing pose_landmarker.task cannot crash-loop
        # the supervised pose worker. If it is absent we skip the pose threads entirely;
        # hand tracking continues and Unity sees no BODY_POSE (-> "body tracking unavailable").
        base_dir = os.path.dirname(os.path.abspath(__file__))
        pose_model_path = os.path.join(base_dir, POSE_MODEL_FILENAME)

        if not os.path.exists(pose_model_path):
            _LOGGER.warning(
                "[*] Body-pose ENABLED but model '%s' is missing at %s -> body tracking "
                "UNAVAILABLE; hand tracking continues. Place pose_landmarker.task beside "
                "server.py, or set GUARDIAN_POSE_ENABLED=0 to silence this.",
                POSE_MODEL_FILENAME, pose_model_path,
            )
        else:
            pose_frame_queue = queue.Queue(maxsize=POSE_FRAME_QUEUE_MAXSIZE)
            client_id_provider = lambda: udp_receiver.active_client_id

            _LOGGER.info(
                "[*] Body-pose pipeline ENABLED: model=%s, camera_index=%d "
                "(override with GUARDIAN_POSE_ENABLED=0 / GUARDIAN_POSE_CAMERA_INDEX)",
                pose_model_path, POSE_CAMERA_INDEX,
            )

            threading.Thread(
                target=_run_supervised,
                args=("POSE-CAPTURE", run_pose_capture, pose_frame_queue, client_id_provider),
                name="GuardianPoseCapture",
                daemon=True,
            ).start()
            threading.Thread(
                target=_run_supervised,
                args=("POSE", run_pose_worker, pose_frame_queue, result_queue, client_id_provider),
                name="GuardianPoseWorker",
                daemon=True,
            ).start()
            _LOGGER.info("[*] Body-pose capture + worker threads started (supervised)")

            # The preview WINDOW is driven on the main thread by _preview_loop() below
            # (Windows OpenCV only paints reliably from the main thread). The worker just
            # stashes annotated frames.
            pose_preview_active = POSE_PREVIEW
    else:
        _LOGGER.info(
            "[*] Body-pose pipeline DISABLED via GUARDIAN_POSE_ENABLED=0. "
            "Hand tracking only; webcam is not opened."
        )

    loop = asyncio.get_running_loop()

    udp_transport, _ = await loop.create_datagram_endpoint(
        lambda: udp_receiver,
        local_addr=("0.0.0.0", UDP_FRAME_PORT),
    )
    _LOGGER.info("[*] UDP listening on :%d", UDP_FRAME_PORT)

    tcp_server = await asyncio.start_server(
        tcp_handler.handle_client,
        "0.0.0.0",
        TCP_CONTROL_PORT,
    )
    _LOGGER.info("[*] TCP listening on :%d", TCP_CONTROL_PORT)

    dispatcher_task = asyncio.create_task(tcp_handler.result_dispatcher())

    preview_task = asyncio.create_task(_preview_loop()) if pose_preview_active else None
    if preview_task is not None:
        _LOGGER.info("[*] Body-pose preview window active on main thread (GUARDIAN_POSE_PREVIEW=1)")

    # Graceful shutdown: SIGINT/SIGTERM set stop_event on POSIX. On Windows,
    # loop.add_signal_handler raises NotImplementedError, so we fall back to
    # the KeyboardInterrupt raised out of asyncio.run() (handled in __main__).
    stop_event = asyncio.Event()
    signals_registered = False
    try:
        loop.add_signal_handler(signal.SIGINT, stop_event.set)
        loop.add_signal_handler(signal.SIGTERM, stop_event.set)
        signals_registered = True
    except (NotImplementedError, RuntimeError):
        pass

    serve_task = asyncio.create_task(tcp_server.serve_forever())
    try:
        if signals_registered:
            await stop_event.wait()
            _LOGGER.info("[*] shutdown signal received")
        else:
            await serve_task
    finally:
        serve_task.cancel()
        try:
            await serve_task
        except (asyncio.CancelledError, Exception):
            pass

        dispatcher_task.cancel()
        try:
            await dispatcher_task
        except (asyncio.CancelledError, Exception):
            pass

        if preview_task is not None:
            preview_task.cancel()
            try:
                await preview_task
            except (asyncio.CancelledError, Exception):
                pass
        # The dispatcher blocks a default-executor thread on result_queue.get();
        # cancelling the coroutine does NOT unblock that thread, and asyncio.run
        # joins the default executor at shutdown. Push a sentinel so the orphaned
        # get() returns and the thread can exit cleanly.
        result_queue.put({"client_id": "_shutdown", "response": {"type": "WARNING"}})

        udp_transport.close()
        tcp_server.close()
        await tcp_server.wait_closed()
        _LOGGER.info("[*] servers closed; shutdown complete")


if __name__ == "__main__":
    setup_logging()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        _LOGGER.info("[*] interrupted (Ctrl+C); shutdown complete")
