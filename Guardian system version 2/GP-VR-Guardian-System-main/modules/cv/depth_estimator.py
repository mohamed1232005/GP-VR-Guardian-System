"""
Depth Estimator Module
Uses MiDaS for depth estimation
"""

import torch
import numpy as np


class DepthEstimator:
    def __init__(self, logger):
        """Initialize MiDaS depth estimator"""
        self.logger = logger
        self.is_ready = False
        
        try:
            self.logger.info("[DEPTH] Loading MiDaS model...")
            
            # Load model
            self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
            
            # Setup device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.logger.info(f"[DEPTH] Device: {self.device}")
            
            self.model.to(self.device)
            self.model.eval()
            
            # Load transform
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            self.transform = transforms.small_transform
            
            # Warm up
            dummy = np.zeros((240, 320, 3), dtype=np.uint8)
            with torch.no_grad():
                dummy_input = self.transform(dummy).to(self.device)
                _ = self.model(dummy_input)
            
            self.is_ready = True
            self.logger.info("[DEPTH] ✓ MiDaS ready!")
        
        except Exception as e:
            self.logger.error(f"[DEPTH] Failed to initialize: {e}")
            self.is_ready = False
    
    def estimate(self, img_rgb):
        """Estimate depth from RGB image"""
        if not self.is_ready:
            return None
        
        try:
            with torch.no_grad():
                # Preprocess
                input_batch = self.transform(img_rgb).to(self.device)
                
                # Predict
                prediction = self.model(input_batch)
                
                # Resize to original size
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img_rgb.shape[:2],
                    mode="bilinear",
                    align_corners=False
                ).squeeze()
                
                # Convert to numpy
                depth = prediction.cpu().numpy().astype(np.float32)
                
                return depth
        
        except Exception as e:
            self.logger.error(f"[DEPTH] Estimation failed: {e}")
            return None