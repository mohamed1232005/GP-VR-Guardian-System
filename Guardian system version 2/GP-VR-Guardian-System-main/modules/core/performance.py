"""
Performance Monitor Module
Tracks FPS and processing times
"""

import time
from collections import deque


class PerformanceMonitor:
    def __init__(self, window_size=30):
        """Initialize performance monitor"""
        self.frame_times = deque(maxlen=window_size)
        self.process_times = deque(maxlen=window_size)
        self.last_time = time.time()
    
    def tick(self):
        """Mark new frame"""
        now = time.time()
        dt = now - self.last_time
        self.frame_times.append(dt)
        self.last_time = now
    
    def add_process_time(self, t):
        """Add processing time"""
        self.process_times.append(t)
    
    def get_fps(self):
        """Get current FPS"""
        if not self.frame_times:
            return 0.0
        avg_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0
    
    def get_avg_process_ms(self):
        """Get average processing time in milliseconds"""
        if not self.process_times:
            return 0.0
        return (sum(self.process_times) / len(self.process_times)) * 1000