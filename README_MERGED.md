# Merged active Python service

This is the merged version of the simplified active Mode 2/System A Python backend.

## Files

- `server.py` — entry point only. Starts queues, CV worker thread, UDP, and TCP.
- `transport.py` — merged `udp_handler.py` + `tcp_handler.py`.
- `hand_tracking.py` — merged `cv_worker.py` + `hand_detector.py` + `smoother.py`.
- `config.py` — runtime constants.
- `gesture_recognizer.task` — keep this model file beside `server.py` when running.
- `pose_tracking.py` — body-pose pipeline (laptop webcam → MediaPipe Pose). Independent of
  the hand pipeline; emits raw `BODY_POSE` (type `0x15`) landmarks to Unity.
- `pose_landmarker.task` — **required** MediaPipe Pose model, kept beside `server.py` (same
  rule as `gesture_recognizer.task`). Download from
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
  and save it as `pose_landmarker.task` (the Lite model is recommended for laptop CPU).
  Toggle the pipeline with `POSE_ENABLED` in `config.py`.

## Body-pose pipeline (laptop webcam)

A second, fully independent pipeline captures the laptop webcam (`cv2.VideoCapture`), runs
MediaPipe Pose, EMA-smooths the 33 landmarks, and emits raw landmarks as TCP `BODY_POSE`
(`0x15`). It has its own capture thread, queue, and worker — the hand/UDP pipeline is
untouched and cannot be starved. JSON body shape:
`{ "type": "BODY_POSE", "landmarks": [[x,y,z,visibility] ×33], "frame_timestamp_ms": int, "tracked": bool }`.
Python stays stateless CV; Unity (`BodyPoseReceiver`) computes form/rep/hold metrics.

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
