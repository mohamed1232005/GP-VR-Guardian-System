"""
Logger Utility - Windows Compatible
Handles logging without unicode issues
"""

import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logger(name='VRGuardian'):
    """Setup logger with file and console handlers - Windows compatible"""
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Create logs directory
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # File handler (UTF-8 for unicode support in files)
    log_filename = log_dir / f'vr_guardian_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler (ASCII only for Windows compatibility)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Set encoding for console on Windows
    if sys.platform == 'win32':
        try:
            # Try to set UTF-8 encoding for Windows console
            sys.stdout.reconfigure(encoding='utf-8')
        except:
            # If fails, console handler will use ASCII
            pass
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', 
                                 datefmt='%H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger