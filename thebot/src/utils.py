"""
Utility Module
Purpose: Provides shared utility classes like logging and performance monitoring.
"""
import logging
import time
import torch
import win32api  # For getting current cursor position
# from logger_util import Logger

# Setup logging

def setup_logging(level=logging.INFO):
    """Sets up the console logger."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        logging.getLogger().setLevel(level)
    return logging.getLogger(__name__)

logger = setup_logging()

class Logger:
    """A simple, standardized logging wrapper."""
    _instance = None

    def __new__(cls, name, debug=False):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            log_level = logging.DEBUG if debug else logging.INFO
            setup_logging(log_level)
        return cls._instance

    def __init__(self, name, debug=False):
        self.logger = logging.getLogger(name)

    def info(self, msg):
        """Log an info message."""
        self.logger.info(msg)

    def debug(self, msg):
        """Log a debug message."""
        self.logger.debug(msg)

    def warning(self, msg):
        """Log a warning message."""
        self.logger.warning(msg)

    def error(self, msg, exc_info=False):
        """Log an error message."""
        self.logger.error(msg, exc_info=exc_info)

class CudaTimer:
    """Measures execution time for CUDA operations."""
    def __init__(self):
        if torch.cuda.is_available():
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.cuda_available = True
        else:
            self.cuda_available = False
            logger.warning("CUDA not available for CudaTimer.")
        self.start_time = None

    def start(self):
        """Starts the timer."""
        if self.cuda_available:
            self.start_event.record(torch.cuda.current_stream())
        self.start_time = time.perf_counter()

    def stop(self):
        """Stops the timer and returns elapsed time in milliseconds."""
        if self.cuda_available:
            self.end_event.record(torch.cuda.current_stream())
            torch.cuda.synchronize()
            cuda_time_ms = self.start_event.elapsed_time(self.end_event)
        else:
            cuda_time_ms = 0
        if self.start_time is None:
            raise RuntimeError("CudaTimer.stop() called before start().")
        cpu_time_ms = (time.perf_counter() - self.start_time) * 1000
        return cuda_time_ms, cpu_time_ms

def screen_to_target_coords(bbox, aim_height_offset):
    """Calculates target coordinates (center + offset) from a bounding box."""
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    target_y = center_y + aim_height_offset
    return center_x, target_y

def get_current_mouse_position():
    """Gets the current mouse cursor position (absolute screen coordinates)."""
    return win32api.GetCursorPos()

class PerformanceMonitor:
    """Measures and logs real-time performance metrics."""
    def __init__(self):
        self.start_time = None
        self.frame_count = 0
        self.total_time = 0
        self.last_log_time = time.time()

    def start_frame(self):
        """Marks the beginning of a processing frame."""
        self.start_time = time.perf_counter()

    def end_frame(self):
        """Marks the end of a processing frame and updates metrics."""
        if self.start_time:
            end_time = time.perf_counter()
            self.total_time += end_time - self.start_time
            self.frame_count += 1

    def log_performance(self, log_interval_frames=100):
        """Logs performance if enough frames have passed."""
        if self.frame_count > 0 and self.frame_count % log_interval_frames == 0:
            avg_latency_ms = (self.total_time / self.frame_count) * 1000
            fps = self.frame_count / self.total_time
            logger = Logger(__name__)
            logger.debug(f"Performance: Avg Latency: {avg_latency_ms:.2f} ms | FPS: {fps:.2f}")
            self.frame_count = 0
            self.total_time = 0

def log_display_info():
    from capture import get_actual_display_resolution, get_dpi_scale
    w, h = get_actual_display_resolution()
    dpi_scale = get_dpi_scale()
    logging.info(f"[DISPLAY] Actual display resolution: {w}x{h}")
    logging.info(f"[DISPLAY] Windows DPI scaling factor: {dpi_scale:.2f}x ({dpi_scale*100:.0f}%)")