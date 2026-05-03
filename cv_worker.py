"""
CV Worker — supervised in-process thread.
Reads latest camera frames from frame_queue and writes HAND_DATA / events to result_queue.
"""

import os
import queue
import traceback
from typing import Any


def _new_session() -> dict:
    return {
        "state": "INIT",
        "floor": None,
        "boundary_pts": [],
    }


def _put_result(result_queue: Any, client_id: str, response: dict) -> None:
    if not response:
        return
    result_queue.put({"client_id": client_id, "response": response})


def run_cv_worker(frame_queue: Any, result_queue: Any) -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)

    from hand_detector import HandDetector

    detector = HandDetector(model_path=os.path.join(base_dir, "gesture_recognizer.task"))
    session_store: dict[str, dict] = {}

    print("[CV] worker ready", flush=True)

    frame_counter = 0

    while True:
        item = frame_queue.get()

        client_id = item.get("client_id")
        if not client_id:
            continue

        session = session_store.setdefault(client_id, _new_session())

        try:
            if "_state_override" in item:
                new_state = item["_state_override"]
                print(f"[CV] state override for {client_id}: {new_state}", flush=True)

                if new_state == "INIT":
                    detector.reset()
                    session_store[client_id] = _new_session()
                else:
                    session["state"] = new_state
                continue

            if "_floor_data" in item:
                session["floor"] = item["_floor_data"]
                session["state"] = "PLACING_POINTS"
                detector.reset()
                print(f"[CV] floor data received for {client_id}; state=PLACING_POINTS", flush=True)
                continue

            jpeg = item.get("jpeg", b"")
            pose = item.get("pose")
            ts = item.get("timestamp_ms")

            if not jpeg or pose is None or ts is None:
                print("[CV] skipped malformed frame", flush=True)
                continue

            frame_counter += 1
            if frame_counter % 30 == 1:
                print(
                    f"[CV] frame #{frame_counter} client={client_id} "
                    f"state={session.get('state')} ts={ts} jpeg_bytes={len(jpeg)}",
                    flush=True,
                )

            response = detector.process_frame(
                jpeg_bytes=jpeg,
                pose_matrix=pose,
                timestamp_ms=int(ts),
                session=session,
            )

            if response:
                if frame_counter % 30 == 1 or response.get("type") != "HAND_DATA":
                    print(
                        f"[CV] produced response type={response.get('type')} "
                        f"gesture={response.get('gesture')}",
                        flush=True,
                    )
                _put_result(result_queue, client_id, response)

        except BaseException as exc:
            print("[CV] recovered from frame-processing error:", repr(exc), flush=True)
            traceback.print_exc()

            _put_result(
                result_queue,
                client_id,
                {
                    "type": "WARNING",
                    "message": "CV worker error. Check Python terminal for traceback.",
                },
            )