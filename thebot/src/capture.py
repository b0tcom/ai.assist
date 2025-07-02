from logger_util import Logger
import bettercam
import numpy as np
from config_manager import ConfigManager  # NEW: Use ConfigManager for config access


logger = Logger(name=__name__)

# Use bettercam for all screen capture and resolution functions

def get_screen_size():
    """Get the Windows desktop resolution (what user sees - may be scaled)."""
    try:
        # Use fallback: bettercam does not provide get_screen_size
        w, h = 1920, 1080
        return w, h
    except Exception as e:
        logger.error(f"[bettercam] Failed to get screen size: {e}")
        return 1920, 1080

def get_actual_display_resolution():
    """Get the actual display resolution (GPU/signal, what capture libs use)."""
    try:
        # Fallback to get_screen_size if bettercam does not provide this function
        w, h = get_screen_size()
        logger.info(f"[bettercam] Actual display resolution (fallback): {w}x{h}")
        return w, h
    except Exception as e:
        logger.error(f"[bettercam] Failed to get actual display resolution: {e}")
        return 1920, 1080

def get_dpi_scale():
    """Calculate DPI scaling factor for x and y using bettercam."""
    try:
        get_dpi = getattr(bettercam, "get_dpi_scale", None)
        if callable(get_dpi):
            scale = get_dpi()
            if isinstance(scale, (tuple, list)) and len(scale) == 2:
                scale_x, scale_y = scale
            else:
                scale_x = scale_y = float(scale) if isinstance(scale, (int, float)) else 1.0
            logger.info(f"[bettercam] DPI Scale detected: x={scale_x:.3f}, y={scale_y:.3f}")
            return scale_x, scale_y
        logger.warning("[bettercam] get_dpi_scale not found, defaulting to 1.0")
        return 1.0, 1.0
    except Exception as e:
        logger.error(f"[bettercam] Failed to get DPI scale: {e}")
        return 1.0, 1.0

class ScreenCapture:
    """Simple region holder for screen capture."""
    def __init__(self, region=None, config_path="configs/config.ini"):
        # Use ConfigManager to get region if not provided
        if region is None:
            self.config_manager = ConfigManager(config_path)
            self.region = self.config_manager.get_screen_region()
        else:
            self.region = region

    def get_region(self):
        return self.region

    def __enter__(self):
        # Context manager entry: nothing special needed
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Context manager exit: no persistent resources, but could add future cleanup here
        return False

def get_frame(region=None, config_path="configs/config.ini"):
    """Capture a frame from the screen using bettercam for the given region.
    If region is None, use ConfigManager to get the region from config.
    """
    try:
        if region is None:
            config_manager = ConfigManager(config_path)
            region = config_manager.get_screen_region()
        # region: dict with keys left, top, width, height
        get_frame_func = getattr(bettercam, "get_frame", None)
        if callable(get_frame_func):
            frame = get_frame_func(region)
            return frame  # Should be a numpy array (BGR)
        else:
            logger.error("[bettercam] No 'get_frame' function found in bettercam module.")
            return None
    except Exception as e:
        logger.error(f"[bettercam] Failed to capture frame: {e}")
        return None
