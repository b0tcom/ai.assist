import bettercam
import numpy as np
import cv2
from utils import Logger
import config

logger = Logger(name=__name__)

# Load CONFIG before using it
CONFIG = getattr(config, "load_config", lambda: config.Config)() if hasattr(config, "load_config") else getattr(config, "CONFIG", None)

class ScreenCapture:
    """Handles high-performance screen capturing using bettercam."""
    def __init__(self, region=None):
        # Default region: 640x640 centered on 1600x1024
        screen_w, screen_h = 1600, 1024
        default_size = 640
        center_x, center_y = screen_w // 2, screen_h // 2
        default_region = {
            "left": max(center_x - default_size // 2, 0),
            "top": max(center_y - default_size // 2, 0),
            "width": default_size,
            "height": default_size
        }
        # Use region from config if not provided
        if region:
            if isinstance(region, (tuple, list)):
                self.region = {
                    "left": region[0],
                    "top": region[1],
                    "width": min(region[2], default_size),
                    "height": min(region[3], default_size)
                }
            elif isinstance(region, dict):
                self.region = {
                    "left": region.get("left", default_region["left"]),
                    "top": region.get("top", default_region["top"]),
                    "width": min(region.get("width", default_size), default_size),
                    "height": min(region.get("height", default_size), default_size)
                }
            else:
                self.region = default_region
        else:
            self.region = default_region
        logger.info("Screen capture initialized for region: {}".format(self.region))
        self.cam = bettercam.create()

    def capture(self):
        """Captures a frame from the specified region using bettercam."""
        try:
            frame = self.cam.grab(
                region=(
                    self.region["left"],
                    self.region["top"],
                    self.region["width"],
                    self.region["height"]
                )
            )
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

# Example usage (for testing) - Removed to keep files focused on their role
# If you want to use live screen capture, import ScreenCapture in main.py and call capture().
# Example:
# from capture import ScreenCapture
# capture = ScreenCapture(region=self.config.get("screen_region"))
# frame = capture.capture()