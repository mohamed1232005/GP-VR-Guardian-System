"""
CV Worker — runs in a separate OS process (own GIL).
Reads frames from frame_queue, writes response dicts to result_queue.
"""

import multiprocessing as mp


def run_cv_worker(
    frame_queue: mp.Queue,
    result_queue: mp.Queue,
) -> None:
    # Imports deferred intentionally — they run inside the child process.
    from hand_detector import HandDetector
    from session import new_session

    detector = HandDetector()
    session_store: dict = {}

    print("[CV] worker started")

    while True:
        item = frame_queue.get()   # blocks until a frame arrives

        client_id = item.get("client_id")
        if client_id is None:
            print("[CV] skipping packet with no client_id")
            continue

        session = session_store.setdefault(client_id, new_session())

        # Propagate session state updates pushed by the TCP handler
        if "_state_override" in item:
            new_state = item["_state_override"]
            print(f"[CV] state override for {client_id}: {new_state}")

            if new_state == "INIT":
                detector.reset()
                session_store[client_id] = new_session()
                session_store[client_id]["state"] = "INIT"
            else:
                session["state"] = new_state
            continue

        if "_floor_data" in item:
            session["floor"] = item["_floor_data"]
            print(f"[CV] floor data received for {client_id}")
            continue

        print(
            f"[CV] frame for {client_id} "
            f"state={session.get('state')} "
            f"ts={item.get('timestamp_ms')} "
            f"jpeg_bytes={len(item.get('jpeg', b''))}"
        )

        response = detector.process_frame(
            item["jpeg"],
            item["pose"],
            item["timestamp_ms"],
            session,
        )

        if response:
            print(f"[CV] produced response type={response.get('type')} for {client_id}")
            result_queue.put({"client_id": client_id, "response": response})