# ===========================================================================
# models/depth_worker.py — Phase 4: Async Depth Inference Worker
# Runs depth model inference in a background thread.
# Processes only the latest available frame (drops stale frames).
# Throttled to target_depth_fps to avoid overload.
# ===========================================================================

import threading
import time
import numpy as np

from models.depth_model import DepthModelBase, create_depth_model


class DepthWorker:
    """
    Background worker that runs depth estimation on the latest available frame.

    Design:
    - UDP receiver pushes the newest frame via push_frame().
    - Worker thread wakes up, grabs the latest frame, runs inference.
    - Only the newest frame is kept; stale frames are dropped.
    - Inference rate is limited by target_depth_fps.
    """

    def __init__(
        self,
        model_name: str = "depth_anything_v2_small",
        target_depth_fps: float = 1.0,
        allow_dummy: bool = False,
        use_metric_indoor: bool = True,
    ):
        self._model_name = model_name
        self._target_depth_fps = target_depth_fps
        self._allow_dummy = allow_dummy
        self._use_metric_indoor = use_metric_indoor
        self._min_interval = 1.0 / max(0.01, target_depth_fps)

        # Depth model (created on start)
        self._model: DepthModelBase = None

        # Frame input (latest only, thread-safe)
        self._input_lock = threading.Lock()
        self._pending_frame = None       # np.ndarray BGR
        self._pending_metadata = None    # dict
        self._pending_ready = False

        # Depth output (latest result, thread-safe)
        self._output_lock = threading.Lock()
        self._latest_depth = None        # np.ndarray float32 (H, W) meters
        self._latest_depth_metadata = None
        self._latest_depth_stats = None  # dict with min, max, mean
        self._depth_count = 0

        # Thread control
        self._thread = None
        self._running = False
        self._event = threading.Event()  # Signal when new frame arrives

        # Timing
        self._fps_count = 0
        self._fps_start = time.time()

    def start(self):
        """Initialize model and start the worker thread."""
        print(f"[P4_DEPTH] starting depth worker "
              f"(target_fps={self._target_depth_fps})")

        # Create model (may take time for real models)
        self._model = create_depth_model(
            model_name=self._model_name,
            allow_dummy=self._allow_dummy,
            use_metric_indoor=self._use_metric_indoor,
        )

        self._running = True
        self._fps_start = time.time()
        self._fps_count = 0

        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

        # Summary log
        print(f"[P4_DEPTH] depth worker started "
              f"(model={self._model.model_name}, "
              f"real={self._model.is_real}, "
              f"relative={self._model.is_relative})")

        if not self._model.is_real:
            print("[P4_DEPTH_WARN] running with DUMMY model — "
                  "Phase 4 pass requires a real AI backend")

    def stop(self):
        """Stop the worker thread cleanly."""
        self._running = False
        self._event.set()  # Wake up the thread so it can exit
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        print(f"[P4_DEPTH] depth worker stopped. "
              f"Total inferences={self._depth_count}")

    def push_frame(self, frame: np.ndarray, metadata: dict):
        """
        Push a new frame for depth processing.
        Called by UDP receiver. Non-blocking — just overwrites the pending frame.
        Stale frames are automatically dropped.
        """
        with self._input_lock:
            self._pending_frame = frame
            self._pending_metadata = metadata
            self._pending_ready = True
        self._event.set()  # Signal the worker

    def get_latest_depth(self) -> tuple:
        """
        Get the latest depth result.
        Returns (depth_map: np.ndarray or None, stats: dict or None)
        """
        with self._output_lock:
            return self._latest_depth, self._latest_depth_stats

    def _worker_loop(self):
        """Main worker loop: wait for frame -> run inference -> store result."""
        while self._running:
            # Wait for a frame signal or timeout
            self._event.wait(timeout=self._min_interval)
            self._event.clear()

            if not self._running:
                break

            # Grab the latest pending frame (drops all stale)
            frame = None
            metadata = None
            with self._input_lock:
                if self._pending_ready:
                    frame = self._pending_frame
                    metadata = self._pending_metadata
                    self._pending_ready = False

            if frame is None:
                continue

            # Run depth inference with timing
            frame_id = metadata.get("frame_id", "?") if metadata else "?"

            try:
                t_start = time.time()

                # Preprocess timing (model handles internally, but we time the full call)
                t_pre = time.time()
                # The predict method handles all pre/post processing
                t_infer_start = time.time()
                depth_map = self._model.predict(frame)
                t_infer_end = time.time()

                t_post_start = time.time()
                # Compute stats
                depth_min = float(np.min(depth_map))
                depth_max = float(np.max(depth_map))
                depth_mean = float(np.mean(depth_map))
                t_post_end = time.time()

                t_total = time.time() - t_start

                stats = {
                    "frame_id": frame_id,
                    "depth_shape": depth_map.shape,
                    "min": depth_min,
                    "max": depth_max,
                    "mean": depth_mean,
                    "is_real": self._model.is_real,
                    "is_relative": self._model.is_relative,
                }

                # Store result
                with self._output_lock:
                    self._latest_depth = depth_map
                    self._latest_depth_metadata = metadata
                    self._latest_depth_stats = stats
                    self._depth_count += 1

                # Compute AI FPS
                self._fps_count += 1
                elapsed_since_start = time.time() - self._fps_start
                ai_fps = (self._fps_count / elapsed_since_start
                          if elapsed_since_start > 0 else 0)

                # Timing breakdown (ms)
                preprocess_ms = (t_infer_start - t_pre) * 1000
                infer_ms = (t_infer_end - t_infer_start) * 1000
                post_ms = (t_post_end - t_post_start) * 1000

                print(
                    f"[P4_DEPTH] frame={frame_id} "
                    f"depth_shape={depth_map.shape} "
                    f"min={depth_min:.2f} max={depth_max:.2f} "
                    f"mean={depth_mean:.2f} ai_fps={ai_fps:.2f}"
                )
                print(
                    f"[P4_DEPTH_TIME] preprocess_ms={preprocess_ms:.1f} "
                    f"infer_ms={infer_ms:.1f} post_ms={post_ms:.1f}"
                )

                # Note for relative depth
                if self._model.is_relative and self._depth_count <= 3:
                    print("[P4_DEPTH_NOTE] relative_depth=True "
                          "metric_scale_pending=True")

            except Exception as e:
                print(f"[P4_DEPTH_ERR] inference failed for "
                      f"frame={frame_id}: {e}")

            # Throttle: sleep remaining time to respect target FPS
            elapsed = time.time() - t_start
            sleep_time = self._min_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
