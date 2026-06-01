# ===========================================================================
# models/depth_model.py — Phase 4: Depth Model Wrapper
# Provides a unified interface for metric depth estimation.
#
# Backend priority:
#   GPU available  → Depth Pro (apple/DepthPro-hf)
#   CPU only       → Depth Anything V2 Small (metric indoor preferred)
#   Metric fails   → Depth Anything V2 Small (relative)
#   All fail       → DummyDepthModel (only if DEPTH_ALLOW_DUMMY=True)
#
# Uses subprocess-based probing to safely detect Depth Pro crashes/segfaults.
# ===========================================================================

import time
import subprocess
import sys
import os
import numpy as np


class DepthModelBase:
    """Abstract base for depth estimation models."""

    def __init__(self):
        self.model_name = "base"
        self.is_real = False
        self.is_relative = False   # True if model outputs relative (not metric) depth

    def predict(self, rgb_frame: np.ndarray) -> np.ndarray:
        """
        Run depth estimation on an RGB frame.

        Args:
            rgb_frame: HxWx3 uint8 BGR (OpenCV format) or RGB numpy array.

        Returns:
            Depth map as HxW float32 numpy array, values in meters (metric)
            or arbitrary units (relative).
        """
        raise NotImplementedError


class DummyDepthModel(DepthModelBase):
    """
    Dummy fallback that generates a synthetic depth map.
    Produces a smooth gradient from 0.5m (top) to 5.0m (bottom)
    to simulate a simple floor-receding pattern.

    WARNING: This does NOT count as Phase 4 pass.
    """

    def __init__(self):
        super().__init__()
        self.model_name = "DummyDepth"
        self.is_real = False
        self.is_relative = False
        print("[P4_DEPTH_ERROR] all real depth backends failed")
        print("[P4_DEPTH_WARN] emergency dummy enabled; "
              "this does NOT count as Phase 4 pass")

    def predict(self, rgb_frame: np.ndarray) -> np.ndarray:
        h, w = rgb_frame.shape[:2]
        # Gradient: near at top (0.5m), far at bottom (5.0m)
        depth = np.linspace(0.5, 5.0, h, dtype=np.float32)
        depth = np.tile(depth[:, np.newaxis], (1, w))
        # Add slight noise for realism
        noise = np.random.uniform(-0.05, 0.05, (h, w)).astype(np.float32)
        depth = depth + noise
        return depth


# ===========================================================================
# Depth Anything V2 Small — lightweight CPU-friendly backend
# ===========================================================================

class DepthAnythingV2Model(DepthModelBase):
    """
    Depth Anything V2 Small via HuggingFace transformers.
    Supports both metric-indoor and relative variants.

    Metric Indoor: depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf
        → outputs absolute depth in meters, ideal for VR guardian.
    Relative:      depth-anything/Depth-Anything-V2-Small-hf
        → outputs relative depth (unitless), needs future calibration.
    """

    # HuggingFace model IDs
    MODEL_ID_METRIC_INDOOR = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
    MODEL_ID_RELATIVE = "depth-anything/Depth-Anything-V2-Small-hf"

    def __init__(self, device: str = "cpu", use_metric_indoor: bool = True):
        super().__init__()
        self._device = device
        self._model = None
        self._processor = None
        self._use_metric_indoor = use_metric_indoor

        self._load_model()

    def _load_model(self):
        """Load Depth Anything V2 Small model and processor."""
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        # Try metric indoor first, fall back to relative
        if self._use_metric_indoor:
            model_id = self.MODEL_ID_METRIC_INDOOR
            try:
                self._load_from_id(model_id, AutoImageProcessor,
                                   AutoModelForDepthEstimation)
                self.model_name = "DepthAnythingV2_MetricIndoor_Small"
                self.is_real = True
                self.is_relative = False
                print(f"[P4_DEPTH] backend=depth_anything_v2_small "
                      f"device={self._device} real=True loaded=True")
                print(f"[P4_DEPTH] variant=metric_indoor "
                      f"model_id={model_id}")
                return
            except Exception as e:
                print(f"[P4_DEPTH_WARN] metric indoor failed: {e}")
                print("[P4_DEPTH] falling back to relative variant...")

        # Relative fallback
        model_id = self.MODEL_ID_RELATIVE
        self._load_from_id(model_id, AutoImageProcessor,
                           AutoModelForDepthEstimation)
        self.model_name = "DepthAnythingV2_Relative_Small"
        self.is_real = True
        self.is_relative = True
        print(f"[P4_DEPTH] backend=depth_anything_v2_small "
              f"device={self._device} real=True loaded=True")
        print(f"[P4_DEPTH] variant=relative model_id={model_id}")
        print("[P4_DEPTH_NOTE] relative_depth=True "
              "metric_scale_pending=True")

    def _load_from_id(self, model_id, ProcessorClass, ModelClass):
        """Load a specific model by HuggingFace ID."""
        import torch

        print(f"[P4_DEPTH] loading model={model_id} device={self._device}")
        t0 = time.time()

        self._processor = ProcessorClass.from_pretrained(model_id)
        self._model = ModelClass.from_pretrained(model_id)
        self._model.to(self._device)
        self._model.eval()

        load_s = time.time() - t0
        print(f"[P4_DEPTH] model loaded in {load_s:.1f}s on {self._device}")

    def predict(self, rgb_frame: np.ndarray) -> np.ndarray:
        """
        Run Depth Anything V2 inference on an OpenCV BGR frame.

        Returns depth map at original resolution as float32 numpy array.
        For metric model: values in meters.
        For relative model: relative depth values (unitless).
        """
        import torch
        from PIL import Image
        import cv2

        h, w = rgb_frame.shape[:2]

        # OpenCV BGR -> RGB PIL Image
        rgb = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        # Preprocess
        inputs = self._processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = self._model(**inputs)

        predicted_depth = outputs.predicted_depth  # (1, H', W')

        # Interpolate to original size
        depth_tensor = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).squeeze()  # (H, W)

        # Convert to numpy float32
        depth_np = depth_tensor.cpu().numpy().astype(np.float32)

        return depth_np


# ===========================================================================
# Depth Pro — high-accuracy GPU backend
# ===========================================================================

class DepthProModel(DepthModelBase):
    """
    Depth Pro model via HuggingFace transformers.
    Uses apple/DepthPro-hf for metric monocular depth estimation.
    Best used with GPU. Segfaults on CPU-only machines.
    """

    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.model_name = "DepthPro"
        self.is_real = True
        self.is_relative = False
        self._device = device
        self._model = None
        self._processor = None

        self._load_model()

    def _load_model(self):
        """Load Depth Pro model and processor from HuggingFace."""
        import torch
        from transformers import DepthProForDepthEstimation, DepthProImageProcessor

        model_id = "apple/DepthPro-hf"
        print(f"[P4_DEPTH] loading model={model_id} device={self._device}")

        if self._device == "cpu":
            print("[P4_DEPTH_WARN] running on CPU — Depth Pro will be slow. "
                  "GPU strongly recommended.")

        t0 = time.time()
        self._processor = DepthProImageProcessor.from_pretrained(model_id)
        self._model = DepthProForDepthEstimation.from_pretrained(model_id)
        self._model.to(self._device)
        self._model.eval()

        load_s = time.time() - t0
        print(f"[P4_DEPTH] model loaded in {load_s:.1f}s on {self._device}")
        print(f"[P4_DEPTH] backend=depth_pro device={self._device} "
              f"real=True loaded=True")

    def predict(self, rgb_frame: np.ndarray) -> np.ndarray:
        """
        Run Depth Pro inference on an OpenCV BGR frame.

        Returns depth map in meters, same spatial size as input.
        """
        import torch
        from PIL import Image
        import cv2

        h, w = rgb_frame.shape[:2]

        # OpenCV BGR -> RGB PIL Image
        rgb = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        # Preprocess
        inputs = self._processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = self._model(**inputs)

        # Post-process to get depth map at original resolution
        post = self._processor.post_process_depth_estimation(
            outputs,
            target_sizes=[(h, w)]
        )

        depth_tensor = post[0]["predicted_depth"]  # (H, W) tensor

        # Convert to numpy float32
        depth_np = depth_tensor.cpu().numpy().astype(np.float32)

        return depth_np


# ---------------------------------------------------------------------------
# Subprocess-based safety probe (Depth Pro only)
# ---------------------------------------------------------------------------
# Depth Pro can segfault on CPU (exit code 0xC0000005 on Windows).
# A segfault kills the entire process — no Python exception handler can
# catch it. We run the probe in an isolated subprocess so a crash there
# does NOT take down our main server.

PROBE_TIMEOUT = 180  # seconds — generous for CPU-only first inference

_PROBE_SCRIPT = '''
import sys, os, time
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
try:
    import numpy as np, torch, cv2
    from PIL import Image
    from transformers import DepthProForDepthEstimation, DepthProImageProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "apple/DepthPro-hf"
    processor = DepthProImageProcessor.from_pretrained(model_id)
    model = DepthProForDepthEstimation.from_pretrained(model_id)
    model.to(device)
    model.eval()

    # Tiny inference test
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    pil = Image.fromarray(img)
    inputs = processor(images=pil, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    post = processor.post_process_depth_estimation(outputs, target_sizes=[(32, 32)])
    d = post[0]["predicted_depth"]
    print(f"PROBE_OK shape={tuple(d.shape)}")
    sys.exit(0)

except Exception as e:
    print(f"PROBE_FAIL {e}")
    sys.exit(1)
'''


def _probe_depth_pro() -> bool:
    """
    Run Depth Pro load+inference in an isolated subprocess.
    Returns True if the probe succeeds, False if it fails or crashes.
    """
    print(f"[P4_DEPTH] running safety probe in subprocess "
          f"(timeout={PROBE_TIMEOUT}s)...")

    try:
        result = subprocess.run(
            [sys.executable, "-c", _PROBE_SCRIPT],
            capture_output=True, text=True,
            timeout=PROBE_TIMEOUT,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        if result.returncode == 0 and "PROBE_OK" in stdout:
            print(f"[P4_DEPTH] probe succeeded: {stdout}")
            return True
        else:
            print(f"[P4_DEPTH_WARN] probe failed (exit={result.returncode})")
            if stdout:
                print(f"[P4_DEPTH_WARN] stdout: {stdout[-300:]}")
            if stderr:
                # Only print last 300 chars to avoid noise
                print(f"[P4_DEPTH_WARN] stderr: {stderr[-300:]}")
            return False

    except subprocess.TimeoutExpired:
        print(f"[P4_DEPTH_WARN] probe timed out after {PROBE_TIMEOUT}s")
        return False
    except Exception as e:
        print(f"[P4_DEPTH_WARN] probe error: {e}")
        return False


# ===========================================================================
# Factory: create_depth_model — cascading backend selection
# ===========================================================================

def create_depth_model(
    model_name: str = "depth_anything_v2_small",
    allow_dummy: bool = False,
    use_metric_indoor: bool = True,
    try_depth_pro: bool = False,
) -> DepthModelBase:
    """
    Factory function to create the appropriate depth model.

    Backend priority:
      1. If model_name="dummy"          → DummyDepthModel (explicit)
      2. If try_depth_pro=True and GPU  → Depth Pro (with subprocess probe)
      3. If model_name="depth_anything_v2_small" or auto-selected:
         a. Try Depth Anything V2 Metric Indoor Small
         b. Fall back to Depth Anything V2 Relative Small
      4. If all real backends fail:
         - If allow_dummy=True  → DummyDepthModel (emergency)
         - If allow_dummy=False → raise RuntimeError

    Args:
        model_name: "depth_anything_v2_small", "depth_pro", or "dummy"
        allow_dummy: Whether to allow dummy fallback when all real models fail
        use_metric_indoor: Whether to prefer metric indoor variant
        try_depth_pro: Whether to attempt Depth Pro probe (disabled by default)

    Returns:
        An instance of DepthModelBase.
    """

    # --- Explicit dummy request ---
    if model_name == "dummy":
        if not allow_dummy:
            raise RuntimeError(
                "[P4_DEPTH_ERROR] dummy model explicitly requested but "
                "DEPTH_ALLOW_DUMMY=False. Set DEPTH_ALLOW_DUMMY=True in "
                "config.py to allow dummy depth."
            )
        print("[P4_DEPTH] using dummy depth model (explicitly requested)")
        return DummyDepthModel()

    # --- Detect device ---
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        has_cuda = torch.cuda.is_available()
        print(f"[P4_DEPTH] torch={torch.__version__} device={device} "
              f"cuda={has_cuda}")
    except ImportError as e:
        print(f"[P4_DEPTH_WARN] torch import failed: {e}")
        return _fallback_or_raise(allow_dummy)

    # --- GPU path: try Depth Pro if enabled ---
    if try_depth_pro and (model_name == "depth_pro" or has_cuda):
        print("[P4_DEPTH] Depth Pro probe enabled, attempting...")
        depth_pro_model = _try_depth_pro(device)
        if depth_pro_model is not None:
            return depth_pro_model
        print("[P4_DEPTH] Depth Pro failed, trying "
              "Depth Anything V2 Small...")
    elif not try_depth_pro and model_name == "depth_pro":
        print("[P4_DEPTH_WARN] DEPTH_TRY_DEPTH_PRO=False, skipping "
              "Depth Pro probe. Using Depth Anything V2 instead.")

    # --- CPU path (or GPU fallback): Depth Anything V2 Small ---
    depth_anything_model = _try_depth_anything_v2(device, use_metric_indoor)
    if depth_anything_model is not None:
        return depth_anything_model

    # --- All real backends failed ---
    return _fallback_or_raise(allow_dummy)


def _try_depth_pro(device: str):
    """Attempt to load Depth Pro with subprocess safety probe."""
    try:
        from transformers import (DepthProForDepthEstimation,
                                  DepthProImageProcessor)
        print("[P4_DEPTH] transformers DepthPro classes found")
    except ImportError as e:
        print(f"[P4_DEPTH_WARN] DepthPro import error: {e}")
        return None

    # Subprocess safety probe
    if not _probe_depth_pro():
        print("[P4_DEPTH_WARN] Depth Pro probe failed.")
        return None

    # Load in-process (probe passed, so this should be safe)
    try:
        model = DepthProModel(device=device)
        print("[P4_DEPTH] Depth Pro loaded successfully in main process")
        return model
    except Exception as e:
        print(f"[P4_DEPTH_WARN] Depth Pro in-process load failed: {e}")
        return None


def _try_depth_anything_v2(device: str, use_metric_indoor: bool):
    """
    Attempt to load Depth Anything V2 Small.
    Tries metric indoor first (if requested), then relative.
    """
    try:
        from transformers import (AutoImageProcessor,
                                  AutoModelForDepthEstimation)
        print("[P4_DEPTH] transformers DepthAnything classes found")
    except ImportError as e:
        print(f"[P4_DEPTH_WARN] Depth Anything import error: {e}")
        return None

    try:
        model = DepthAnythingV2Model(
            device=device,
            use_metric_indoor=use_metric_indoor,
        )
        return model
    except Exception as e:
        print(f"[P4_DEPTH_WARN] Depth Anything V2 Small load failed: {e}")
        return None


def _fallback_or_raise(allow_dummy: bool):
    """Return DummyDepthModel if allowed, otherwise raise."""
    if allow_dummy:
        return DummyDepthModel()
    else:
        raise RuntimeError(
            "[P4_DEPTH_ERROR] all real depth backends failed and "
            "DEPTH_ALLOW_DUMMY=False. Cannot start depth worker.\n"
            "Solutions:\n"
            "  1. Install transformers>=4.45.0 and torch\n"
            "  2. Ensure network access for model download\n"
            "  3. Set DEPTH_ALLOW_DUMMY=True for emergency dummy mode"
        )