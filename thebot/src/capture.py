from logger_util import Logger
import bettercam
import numpy as np
from config_manager import ConfigManager  # NEW: Use ConfigManager for config access


logger = Logger(name=__name__)

<<<<<<< HEAD
# Use bettercam for all screen capture and resolution functions
=======
# Load CONFIG before using it
CONFIG = getattr(config, "load_config", lambda: config.Config)() if hasattr(
    config, "load_config") else getattr(config, "CONFIG", None)

>>>>>>> b93bf65d8d3af3b268c813665afac1be30d6e3ec

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
<<<<<<< HEAD
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
=======
    """Handles high-performance screen capturing using bettercam."""

    def __init__(self, region=None):
        # Use real screen size for all region math
        self.screen_w, self.screen_h = get_screen_size()
        default_size = 640
        center_x, center_y = self.screen_w // 2, self.screen_h // 2
        default_region = {
            "left": max(center_x - default_size // 2, 0),
            "top": max(center_y - default_size // 2, 0),
            "width": min(default_size, self.screen_w),
            "height": min(default_size, self.screen_h)
        }
        # Use region from config if not provided
        if region:
            if isinstance(region, (tuple, list)):
                self.region = {
                    "left": int(region[0]),
                    "top": int(region[1]),
                    "width": int(region[2]),
                    "height": int(region[3])
                }
            elif isinstance(region, dict):
                self.region = {
                    "left": int(region.get("left", default_region["left"])),
                    "top": int(region.get("top", default_region["top"])),
                    "width": int(region.get("width", default_size)),
                    "height": int(region.get("height", default_size))
                }
            else:
                self.region = default_region
        else:
            self.region = default_region
        logger.info("Screen capture initialized for region: {}".format(
            self.region))
        self.cam = bettercam.create()

    def _sanitize_region(self):
        # Clamp to bounds and ensure int using real screen size
        left = int(max(0, min(int(self.region["left"]), self.screen_w - 1)))
        top = int(max(0, min(int(self.region["top"]), self.screen_h - 1)))
        width = int(min(int(self.region["width"]), self.screen_w - left))
        height = int(min(int(self.region["height"]), self.screen_h - top))
        region = {"left": left, "top": top, "width": width, "height": height}
        logger.info(f"Sanitized region: {region}")
        logger.info(
            f"Region types: {[type(left), type(top), type(width), type(height)]}"
        )
        assert all(
            isinstance(x, int) and x >= 0
            for x in [left, top, width, height
                      ]), f"Region values must be int and >=0: {region}"
        return region

    def capture(self):
        """Captures a frame from the specified region using bettercam."""
        region = self._sanitize_region()
        try:
            logger.info(f"About to capture region: {region}")
            try:
                frame = self.cam.grab(region=(region["left"], region["top"],
                                              region["width"],
                                              region["height"]))
            except Exception as e:
                logger.error(
                    f"Grab failed with region=({region['left']},{region['top']},{region['width']},{region['height']}): {e}"
                )
                # Try width-1/height-1 if error persists
                if region["width"] > 1 and region["height"] > 1:
                    logger.info(
                        f"Retrying with width-1/height-1: ({region['left']},{region['top']},{region['width']-1},{region['height']-1})"
                    )
                    try:
                        frame = self.cam.grab(region=(region["left"],
                                                      region["top"],
                                                      region["width"] - 1,
                                                      region["height"] - 1))
                    except Exception as e2:
                        logger.error(f"Second grab attempt failed: {e2}")
                        frame = None
                else:
                    frame = None
            if frame is not None:
                # Convert BGRA to BGR for OpenCV (if needed)
                if frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                frame = np.ascontiguousarray(frame)
                return frame
            else:
                logger.error("Screen capture returned None frame.")
                return None
        except Exception as e:
            logger.error(f"Screen capture error: {e}")
            return None
>>>>>>> b93bf65d8d3af3b268c813665afac1be30d6e3ec
