import bettercam
import numpy as np
import cv2
from utils import Logger
import config
try:
    import win32api
except ImportError:
    win32api = None
import mss

logger = Logger(name=__name__)

# Load CONFIG before using it
CONFIG = getattr(config, "load_config", lambda: config.Config)() if hasattr(config, "load_config") else getattr(config, "CONFIG", None)

def get_screen_size():
    if win32api:
        return win32api.GetSystemMetrics(0), win32api.GetSystemMetrics(1)
    else:
        with mss.mss() as sct:
            mon = sct.monitors[1]
            return mon['width'], mon['height']

class ScreenCapture:
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
        logger.info("Screen capture initialized for region: {}".format(self.region))
        self.cam = bettercam.create()

    def _sanitize_region(self):
        # Clamp to bounds and ensure int using real screen size
        left = int(max(0, min(int(self.region["left"]), self.screen_w-1)))
        top = int(max(0, min(int(self.region["top"]), self.screen_h-1)))
        width = int(min(int(self.region["width"]), self.screen_w - left))
        height = int(min(int(self.region["height"]), self.screen_h - top))
        region = {"left": left, "top": top, "width": width, "height": height}
        logger.info(f"Sanitized region: {region}")
        logger.info(f"Region types: {[type(left), type(top), type(width), type(height)]}")
        assert all(isinstance(x, int) and x >= 0 for x in [left, top, width, height]), f"Region values must be int and >=0: {region}"
        return region

    def capture(self):
        """Captures a frame from the specified region using bettercam."""
        region = self._sanitize_region()
        try:
            logger.info(f"About to capture region: {region}")
            try:
                frame = self.cam.grab(
                    region=(region["left"], region["top"], region["width"], region["height"]))
            except Exception as e:
                logger.error(f"Grab failed with region=({region['left']},{region['top']},{region['width']},{region['height']}): {e}")
                # Try width-1/height-1 if error persists
                if region["width"] > 1 and region["height"] > 1:
                    logger.info(f"Retrying with width-1/height-1: ({region['left']},{region['top']},{region['width']-1},{region['height']-1})")
                    try:
                        frame = self.cam.grab(region=(region["left"], region["top"], region["width"]-1, region["height"]-1))
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