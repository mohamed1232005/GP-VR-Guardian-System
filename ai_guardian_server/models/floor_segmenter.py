# ===========================================================================
# models/floor_segmenter.py — Phase 5 (Validation Upgrade)
# Floor segmentation via SegFormer ADE20K.
#
# Key behaviours controlled by config:
#
#   USE_FALLBACK_FLOOR_MASK = False  (recommended for serious testing)
#       If SegFormer semantic floor ratio is too low the segmenter returns
#       stats["rejected"] = True immediately.  No synthetic lower-half mask
#       is created.  The pipeline sees this and emits AI_GUARDIAN_STATUS with
#       reason="semantic_floor_too_low".  This prevents fake geometry from
#       corrupting RANSAC during real validation.
#
#   USE_FALLBACK_FLOOR_MASK = True   (legacy / debug only)
#       Original behaviour: fall back to lower-N% of image as geometric
#       candidate when semantic ratio is too low.
#
#   TEST_ALL_ROTATIONS = True
#       On every debug-save frame, run SegFormer at 0°/90°/180°/270° and
#       save four overlays so you can visually pick the best orientation
#       for INPUT_ROTATION_DEGREES without changing the main pipeline.
# ===========================================================================

import time
import os
import numpy as np

from config import USE_FALLBACK_FLOOR_MASK, TEST_ALL_ROTATIONS, INPUT_ROTATION_DEGREES


class FloorSegmenter:
    """
    Semantic segmentation for floor detection using SegFormer ADE20K.

    When USE_FALLBACK_FLOOR_MASK=False (default for validation):
        Frames with insufficient semantic floor are rejected outright.
    When USE_FALLBACK_FLOOR_MASK=True (legacy):
        Falls back to lower-image geometric candidate mask.
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        hybrid_fallback_ratio: float = 0.01,
        geometric_lower_frac: float = 0.50,
        debug_dir: str = "./debug",
        debug_every_n: int = 10,
        save_debug: bool = True,
    ):
        self._model_id              = model_id
        self._device                = device
        self._hybrid_fallback_ratio = hybrid_fallback_ratio
        self._geometric_lower_frac  = geometric_lower_frac
        self._debug_dir             = debug_dir
        self._debug_every_n         = debug_every_n
        self._save_debug            = save_debug
        self._call_count            = 0

        self._model          = None
        self._processor      = None
        self._floor_class_id = None
        self._floor_label    = None

        if save_debug:
            os.makedirs(debug_dir, exist_ok=True)

        self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self):
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

        print(f"[P5_SEG] loading model={self._model_id} device={self._device}")
        t0 = time.time()

        self._processor = SegformerImageProcessor.from_pretrained(self._model_id)
        self._model     = SegformerForSemanticSegmentation.from_pretrained(self._model_id)
        self._model.to(self._device)
        self._model.eval()

        print(f"[P5_SEG] model loaded in {time.time()-t0:.1f}s on {self._device}")
        self._discover_floor_class()

    def _discover_floor_class(self):
        id2label = self._model.config.id2label

        for class_id, label in id2label.items():
            if label.lower() == "floor":
                self._floor_class_id = int(class_id)
                self._floor_label    = label
                print(f"[P5_SEG] floor_class_id={self._floor_class_id} label={self._floor_label}")
                return

        for class_id, label in id2label.items():
            if "floor" in label.lower():
                self._floor_class_id = int(class_id)
                self._floor_label    = label
                print(f"[P5_SEG] floor_class_id={self._floor_class_id} label={self._floor_label} (partial)")
                return

        if 3 in id2label:
            self._floor_class_id = 3
            self._floor_label    = id2label[3]
            print(f"[P5_SEG] floor_class_id={self._floor_class_id} label={self._floor_label} (fallback id=3)")
            return

        raise RuntimeError(
            "[P5_SEG_ERROR] could not find floor class. "
            f"Labels: {list(id2label.values())[:20]}..."
        )

    # ------------------------------------------------------------------
    # Raw inference (no fallback, no debug save)
    # ------------------------------------------------------------------

    def _run_inference(self, bgr_frame: np.ndarray):
        """
        Run SegFormer inference on a BGR frame.

        Returns:
            (floor_mask: HxW bool, floor_prob: HxW float32, confidence: float)
        """
        import torch
        from PIL import Image
        import cv2

        h, w = bgr_frame.shape[:2]
        rgb       = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        inputs = self._processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        logits    = outputs.logits  # (1, C, H', W')
        upsampled = torch.nn.functional.interpolate(
            logits, size=(h, w), mode="bilinear", align_corners=False,
        )

        seg_map    = upsampled.argmax(dim=1).squeeze(0)
        probs      = torch.nn.functional.softmax(upsampled, dim=1)
        floor_prob = probs[0, self._floor_class_id, :, :]

        floor_mask_tensor = (seg_map == self._floor_class_id)
        floor_mask        = floor_mask_tensor.cpu().numpy().astype(bool)
        floor_prob_np     = floor_prob.cpu().numpy()

        if int(floor_mask.sum()) > 0:
            confidence = float(floor_prob_np[floor_mask].mean())
        else:
            confidence = 0.0

        return floor_mask, floor_prob_np, confidence

    # ------------------------------------------------------------------
    # Main inference
    # ------------------------------------------------------------------

    def segment(self, bgr_frame: np.ndarray, frame_id="?") -> tuple:
        """
        Run floor segmentation on an OpenCV BGR frame.

        Returns:
            (floor_mask, stats)
            floor_mask: HxW bool.
            stats keys:
                floor_pixels, ratio, confidence, used_fallback,
                rejected (bool), reject_reason (str if rejected)
        """
        self._call_count += 1
        h, w = bgr_frame.shape[:2]

        floor_mask, floor_prob_np, confidence = self._run_inference(bgr_frame)

        floor_pixels  = int(floor_mask.sum())
        total_pixels  = h * w
        ratio         = floor_pixels / total_pixels if total_pixels > 0 else 0.0
        used_fallback = False

        is_debug_frame = (self._call_count % self._debug_every_n == 1)

        # ------------------------------------------------------------------
        # Handle semantic floor too low
        # ------------------------------------------------------------------
        if ratio < self._hybrid_fallback_ratio:
            if not USE_FALLBACK_FLOOR_MASK:
                # SERIOUS MODE: reject cleanly — do not create fake mask
                print(
                    f"[P5_SEG_REJECT] id={frame_id} semantic_floor_too_low "
                    f"ratio={ratio:.4f} < {self._hybrid_fallback_ratio} "
                    f"(USE_FALLBACK_FLOOR_MASK=False)"
                )
                if self._save_debug and is_debug_frame:
                    self._save_debug_images(bgr_frame, floor_mask, floor_prob_np,
                                            False, frame_id, floor_pixels, ratio, confidence)
                    if TEST_ALL_ROTATIONS:
                        self._save_rotation_test_overlays(bgr_frame, frame_id)

                return floor_mask, {
                    "floor_pixels":  floor_pixels,
                    "ratio":         ratio,
                    "confidence":    confidence,
                    "used_fallback": False,
                    "rejected":      True,
                    "reject_reason": "semantic_floor_too_low",
                }

            # FALLBACK MODE (legacy): use lower half as geometric candidate
            roi_start = int(h * (1.0 - self._geometric_lower_frac))
            geo_mask  = np.zeros((h, w), dtype=bool)
            geo_mask[roi_start:, :] = True
            floor_mask    = geo_mask
            floor_pixels  = int(geo_mask.sum())
            ratio         = floor_pixels / total_pixels
            confidence    = 0.0
            used_fallback = True
            print(
                f"[P5_SEG_FALLBACK] id={frame_id} semantic too low, using lower "
                f"{self._geometric_lower_frac*100:.0f}% geometric candidate "
                f"(pixels={floor_pixels})"
            )

        # ------------------------------------------------------------------
        # Debug images (normal path)
        # ------------------------------------------------------------------
        if self._save_debug and is_debug_frame:
            self._save_debug_images(bgr_frame, floor_mask, floor_prob_np,
                                    used_fallback, frame_id, floor_pixels, ratio, confidence)
            if TEST_ALL_ROTATIONS:
                self._save_rotation_test_overlays(bgr_frame, frame_id)

        return floor_mask, {
            "floor_pixels":  floor_pixels,
            "ratio":         ratio,
            "confidence":    confidence,
            "used_fallback": used_fallback,
            "rejected":      False,
        }

    # ------------------------------------------------------------------
    # Rotation test overlays
    # ------------------------------------------------------------------

    def _save_rotation_test_overlays(self, bgr_frame: np.ndarray, frame_id):
        """
        Run SegFormer at 0/90/180/270° and save an overlay for each rotation.
        This lets you visually compare which orientation gives the best floor
        mask without changing INPUT_ROTATION_DEGREES in the main pipeline.
        """
        try:
            import cv2
            from geometry.rotation_utils import rotate_image

            ts = int(time.time())
            for rot in (0, 90, 180, 270):
                rotated = rotate_image(bgr_frame, rot)
                mask, prob_np, conf = self._run_inference(rotated)

                pix    = int(mask.sum())
                h_r, w_r = rotated.shape[:2]
                ratio  = pix / (h_r * w_r) if (h_r * w_r) > 0 else 0.0

                overlay = rotated.copy()
                if pix > 0:
                    overlay[mask] = (
                        overlay[mask].astype(np.int32) * [0.4, 1.0, 0.4]
                    ).clip(0, 255).astype(np.uint8)

                cv2.putText(overlay, f"ROT={rot} id={frame_id}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(overlay, f"pix={pix} ratio={ratio:.3f} conf={conf:.3f}",
                            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                fname = os.path.join(
                    self._debug_dir,
                    f"frame_{frame_id}_{ts}_rot{rot}_overlay.jpg"
                )
                cv2.imwrite(fname, overlay)

            print(f"[P5_SEG_ROTATION_TEST] id={frame_id} saved 4 rotation overlays")

        except Exception as e:
            print(f"[P5_SEG_ROTATION_TEST_ERR] {e}")

    # ------------------------------------------------------------------
    # Debug image saving
    # ------------------------------------------------------------------

    def _save_debug_images(self, bgr_frame, floor_mask, floor_prob_map,
                           used_fallback, frame_id="?",
                           pixels=0, ratio=0.0, confidence=0.0):
        """Save RGB, mask, overlay (+ heatmap for semantic path) for inspection."""
        try:
            import cv2
            ts     = int(time.time())
            prefix = os.path.join(self._debug_dir, f"frame_{frame_id}_{ts}")

            # 1. Raw BGR
            cv2.imwrite(f"{prefix}_rgb.jpg", bgr_frame)

            # 2. Binary floor mask
            cv2.imwrite(f"{prefix}_mask.png",
                        (floor_mask.astype(np.uint8)) * 255)

            # 3. Colour overlay
            overlay = bgr_frame.copy()
            overlay[floor_mask] = (
                overlay[floor_mask].astype(np.int32) * [0.4, 1.0, 0.4]
            ).clip(0, 255).astype(np.uint8)

            label     = "FALLBACK" if used_fallback else "SEMANTIC"
            rot_label = f"rot={INPUT_ROTATION_DEGREES}"
            cv2.putText(overlay, f"{label} id={frame_id} {rot_label}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(overlay, f"pixels={pixels} ratio={ratio:.3f}",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(overlay, f"conf={confidence:.3f}",
                        (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            cv2.imwrite(f"{prefix}_overlay.jpg", overlay)

            # 4. Probability heatmap (semantic path only)
            if not used_fallback and floor_prob_map is not None:
                heat = (floor_prob_map * 255).clip(0, 255).astype(np.uint8)
                cv2.imwrite(f"{prefix}_prob.png",
                            cv2.applyColorMap(heat, cv2.COLORMAP_JET))

            print(f"[P5_SEG_DEBUG] id={frame_id} saved prefix={prefix}")

        except Exception as e:
            print(f"[P5_SEG_DEBUG_ERR] could not save debug images: {e}")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def floor_class_id(self):
        return self._floor_class_id

    @property
    def floor_label(self):
        return self._floor_label