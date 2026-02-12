"""
Visualizer Module
Display processing results on laptop
"""

import cv2
import numpy as np


class Visualizer:
    def __init__(self, logger):
        """Initialize visualizer"""
        self.logger = logger
        self.window_created = False
    
    def create_windows(self):
        """Create OpenCV windows"""
        if not self.window_created:
            cv2.namedWindow("VR Guardian - Mobile Feed", cv2.WINDOW_NORMAL)
            cv2.namedWindow("VR Guardian - Depth Map", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("VR Guardian - Mobile Feed", 640, 480)
            cv2.resizeWindow("VR Guardian - Depth Map", 640, 480)
            self.window_created = True
            self.logger.info("[DISPLAY] ✓ Windows created")
    
    def draw_status(self, img, floor_data, fps):
        """Draw status overlay on image"""
        h, w = img.shape[:2]
        
        # Create overlay copy
        display = img.copy()
        
        # Floor status box
        if floor_data['detected'] and floor_data['confidence'] > 40:
            color = (0, 255, 0)  # Green
            status = "FLOOR DETECTED!"
            cv2.rectangle(display, (10, 10), (w-10, 90), color, 3)
            cv2.putText(display, status, (25, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.putText(display, f"Confidence: {floor_data['confidence']}%", 
                       (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            color = (0, 0, 255)  # Red
            status = "Searching for floor..."
            cv2.rectangle(display, (10, 10), (w-10, 60), color, 2)
            cv2.putText(display, status, (25, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # FPS
        fps_color = (0, 255, 0) if fps >= 5 else (0, 165, 255) if fps >= 3 else (0, 0, 255)
        cv2.putText(display, f"FPS: {fps:.1f}", (15, h-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)
        
        return display
    
    def draw_depth(self, depth_map, floor_data):
        """Create depth visualization"""
        if depth_map is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Normalize depth
        d = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)
        lo, hi = np.percentile(d, 5), np.percentile(d, 95)
        if hi - lo > 1e-6:
            d_norm = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
        else:
            d_norm = d
        
        # Convert to color
        depth_gray = (d_norm * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_gray, cv2.COLORMAP_MAGMA)
        
        # Highlight floor area if detected
        if floor_data['detected'] and floor_data['confidence'] > 40:
            h, w = depth_colored.shape[:2]
            overlay = depth_colored.copy()
            # Draw green overlay on bottom portion (likely floor)
            cv2.rectangle(overlay, (0, int(h*0.6)), (w, h), (0, 255, 0), -1)
            depth_colored = cv2.addWeighted(depth_colored, 0.7, overlay, 0.3, 0)
            
            # Add text
            cv2.putText(depth_colored, "FLOOR AREA", (15, h-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return depth_colored
    
    def show(self, img, processed_data, fps):
        """Display all visualizations"""
        self.create_windows()
        
        # Resize for display
        display_img = cv2.resize(img, (640, 480))
        
        # Draw status
        display_with_status = self.draw_status(display_img, processed_data['floor'], fps)
        
        # Draw depth
        depth_visual = self.draw_depth(processed_data['depth_map'], processed_data['floor'])
        depth_visual = cv2.resize(depth_visual, (640, 480))
        
        # Show windows
        cv2.imshow("VR Guardian - Mobile Feed", display_with_status)
        cv2.imshow("VR Guardian - Depth Map", depth_visual)