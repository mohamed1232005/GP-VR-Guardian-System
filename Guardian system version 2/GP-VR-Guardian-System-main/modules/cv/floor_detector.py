"""
Floor Detector Module
Detects ground plane using RANSAC on point cloud
"""

import numpy as np
import open3d as o3d


class FloorDetector:
    def __init__(self, logger):
        """Initialize floor detector"""
        self.logger = logger
        self.is_ready = True
        self.logger.info("[FLOOR] ✓ Floor detector ready!")
    
    def depth_to_pointcloud(self, depth, img_rgb):
        """Convert depth map to 3D point cloud"""
        try:
            # Clean depth
            d = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Normalize
            lo, hi = np.percentile(d, 5), np.percentile(d, 95)
            if hi - lo < 1e-6:
                return None, None
            
            d_norm = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
            depth_u16 = (d_norm * 5000.0).astype(np.uint16)
            
            # Create RGBD
            h, w = img_rgb.shape[:2]
            color_o3d = o3d.geometry.Image(img_rgb)
            depth_o3d = o3d.geometry.Image(depth_u16)
            
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d, depth_o3d,
                depth_scale=1000.0,
                depth_trunc=10.0,
                convert_rgb_to_intensity=False
            )
            
            # Camera intrinsics
            fx = fy = 400.0
            cx, cy = w / 2.0, h / 2.0
            intr = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
            
            # Generate point cloud
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)
            
            return pcd, d_norm
        
        except Exception as e:
            self.logger.error(f"[FLOOR] Point cloud generation failed: {e}")
            return None, None
    
    def detect_plane(self, pcd):
        """Detect ground plane using RANSAC"""
        if pcd is None or len(pcd.points) < 100:
            return None, 0
        
        try:
            # Downsample if needed
            if len(pcd.points) > 2000:
                pcd = pcd.voxel_down_sample(voxel_size=0.04)
            elif len(pcd.points) > 800:
                pcd = pcd.voxel_down_sample(voxel_size=0.02)
            
            if len(pcd.points) < 100:
                return None, 0
            
            # RANSAC segmentation
            plane_model, inliers = pcd.segment_plane(
                distance_threshold=0.04,
                ransac_n=3,
                num_iterations=150
            )
            
            return plane_model, len(inliers)
        
        except Exception as e:
            self.logger.error(f"[FLOOR] RANSAC failed: {e}")
            return None, 0
    
    def detect(self, depth_map, img_rgb):
        """Main detection function"""
        if depth_map is None:
            return {
                "detected": False,
                "plane": [0.0, 0.0, 0.0, 0.0],
                "confidence": 0
            }
        
        # Generate point cloud
        pcd, depth_norm = self.depth_to_pointcloud(depth_map, img_rgb)
        
        # Detect plane
        plane, inliers = self.detect_plane(pcd)
        
        # Build result
        if plane is not None and inliers > 0:
            confidence = min(100, int((inliers / 30.0) * 100))
            return {
                "detected": True,
                "plane": [float(x) for x in plane],
                "confidence": confidence
            }
        else:
            return {
                "detected": False,
                "plane": [0.0, 0.0, 0.0, 0.0],
                "confidence": 0
            }