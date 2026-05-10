# Merged active Python service

This is the merged version of the simplified active Mode 2/System A Python backend.

## Files

- `server.py` — entry point only. Starts queues, CV worker thread, UDP, and TCP.
- `transport.py` — merged `udp_handler.py` + `tcp_handler.py`.
- `hand_tracking.py` — merged `cv_worker.py` + `hand_detector.py` + `smoother.py`.
- `config.py` — runtime constants.
- `gesture_recognizer.task` — keep this model file beside `server.py` when running.

## Removed from active runtime

- `geometry.py`
- `session.py`
- `udp_handler.py`
- `tcp_handler.py`
- `cv_worker.py`
- `hand_detector.py`
- `smoother.py`

Those responsibilities are now either removed as legacy behavior or merged into the active files above.

## Active behavior preserved

Python receives Unity JPEG frames over UDP, detects hand landmarks/gestures with MediaPipe,
applies smoothing and gesture voting, then sends TCP `HAND_DATA` back to Unity.
Unity still owns safe-space creation, boundary safety, rehab scene state, and interactions.
