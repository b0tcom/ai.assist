import os
import json
import logging
import threading
import time
import numpy as np
import mss
import queue
import cv2
import configparser
import pygame

# DPI/screen detection
import ctypes
try:
    import win32api
    import win32con
except ImportError:
    win32api = None
    win32con = None

from input_handler import InputController
from capture import ScreenCapture, get_actual_display_resolution, get_frame
from detect import TargetPredictor, YOLOv8Detector
from pygame_overlay import PygameOverlay
from toggle import ToggleManager
from gui import load_config, save_config, update_field
from config_manager import ConfigManager  # NEW: Use ConfigManager for config access

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

def get_desktop_resolution():
    if win32api:
        return win32api.GetSystemMetrics(0), win32api.GetSystemMetrics(1)
    else:
        with mss.mss() as sct:
            mon = sct.monitors[1]
            return mon['width'], mon['height']

# --- Use only actual display resolution for region math ---
def get_centered_region(fov=None, offset_x=0, offset_y=0, use_desktop_coords=False):
    if use_desktop_coords:
        w, h = get_desktop_resolution()
        logger.info(f"[REGION] Using DESKTOP coordinates: {w}x{h}")
    else:
        w, h = get_actual_display_resolution()
        logger.info(f"[REGION] Using ACTUAL display coordinates: {w}x{h}")
    fov = fov or 280
    left = (w - fov) // 2 + offset_x
    top = (h - fov) // 2 + offset_y
    left = max(0, min(left, w - fov))
    top = max(0, min(top, h - fov))
    region = {'left': int(left), 'top': int(top), 'width': int(fov), 'height': int(fov)}
    logger.info(f"[AUTO] Centered region for {w}x{h} FOV={fov}: {region}")
    return region

print("=== Script is running ===")

class CVTargetingSystem:
    """Main class for the AI Aim Assist System."""
    def __init__(self):
        self.config_manager = ConfigManager("configs/config.ini")  # Use ConfigManager
        self.config = self.config_manager.as_dict()  # For legacy code compatibility
        self.model = None
        self.capture = None
        self.detector = None
        self.toggle_mgr = None
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = None
        self.capture_thread = None
        self.detect_thread = None
        self.monitor = None
        self.running = False
        self.predictor = None
        self.input_ctrl = None
        self.gui = None
        self.capture_window_name = "Live Detection"
        w, h = get_actual_display_resolution()
        self.display_w, self.display_h = w, h
        logger.info(f"Detected actual display resolution: {self.display_w}x{self.display_h}")
        self.overlay = None

    def load_config(self):
        # Use ConfigManager for all config/region logic
        self.config_manager.reload()
        # Always get region as a dict of ints
        region = self.config_manager.get_screen_region()
        logger.info(f"[INFO] Using capture region: {region}")
        # Normalize config to dict with correct types for all expected keys
        config_dict = self.config_manager.as_dict()
        # If using INI, flatten and convert types as needed
        if isinstance(self.config_manager.config, configparser.ConfigParser):
            # Convert all values to dicts with correct types
            def parse_int(val, default=0):
                try:
                    return int(val)
                except Exception:
                    return default
            def parse_float(val, default=0.0):
                try:
                    return float(val)
                except Exception:
                    return default
            # Build a normalized config dict
            config_dict = {
                "yolo": {
                    "model_path": self.config_manager.get("model_settings", "model_path", "thebot/src/models/best.pt"),
                    "confidence_threshold": parse_float(self.config_manager.get("model_settings", "confidence", 0.4)),
                    "device": self.config_manager.get("model_settings", "device", "cuda")
                },
                "target_class_id": parse_int(self.config_manager.get("Application", "target_class_id", 0)),
                "target_class_name": "player",
                "fallback_model_path": self.config_manager.get("model_settings", "fallback_model_path", "thebot/src/models/yolo/yolov8n.pt"),
                "precision_mode": self.config_manager.get("model_settings", "precision_mode", "float32"),
                "warmup_iterations": parse_int(self.config_manager.get("model_settings", "warmup_iterations", 10)),
                "target_priority": parse_int(self.config_manager.get("model_settings", "target_priority", 1)),
                "detection_mode": self.config_manager.get("model_settings", "detection_mode", "tracking"),
                "min_player_size": [parse_int(x) for x in str(self.config_manager.get("model_settings", "min_player_size", "10,10")).split(",")],
                "max_player_size": [parse_int(x) for x in str(self.config_manager.get("model_settings", "max_player_size", "500,500")).split(",")],
                "ethical_mode": self.config_manager.get("model_settings", "ethical_mode", "production"),
                "arduino_port": self.config_manager.get("arduino", "arduino_port", "COM5"),
                "screen_region": region,
                "fov_size": region["width"],
            }
        else:
            # JSON config, just update region
            config_dict["screen_region"] = region
            config_dict["fov_size"] = {"width": region["width"]}
        self.config = config_dict

    def validate_model(self):
        if self.model is None:
            logger.error("Model not loaded for validation")
            return False
        test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        try:
            results = self.model(test_img, verbose=False)
            if results is not None:
                valid = results[0] is not None if isinstance(results, list) else True
                if valid:
                    logger.info(f"Model validation successful. Model classes: {getattr(self.model, 'names', None)}")
                    return True
                else:
                    logger.error("Model did not return results during validation")
                    return False
            else:
                logger.error("Model did not return results during validation")
                return False
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False

    def run(self):
        print("CVTargetingSystem.run() called")
        logger.info("Starting CVTargetingSystem.run()")
        self.load_config()
        print("Config loaded")
        self.input_ctrl = InputController(self.config)
        self.input_ctrl.connect()
        print("InputController connected")
        if not self.input_ctrl.is_connected():
            print("InputController not connected, exiting")
            logger.error("[ERROR] Failed to connect to Arduino. Exiting.")
            return
        print("InputController is connected")
        self.detector = YOLOv8Detector("configs/default_config.json")
        print("YOLOv8Detector instantiated")
        self.model = self.detector.model if hasattr(self.detector, 'model') else None
        logger.info("Loaded model, about to validate...")
        print("About to validate model")
        if not self.validate_model():
            print("Model validation failed. Exiting.")
            logger.error("Model validation failed. Exiting.")
            return
        print("Model validated")
        # --- GUI/Overlay selection ---
        use_pygame = False
        while True:
            mode = input("Select mode: [1] GUI  [2] Pygame Overlay  [q] Quit: ").strip()
            if mode == "1":
                print("Launching configuration GUI...")
                from gui import CVTargetingGUI
                gui = CVTargetingGUI(self.config)
                # Optionally, enable overlay after GUI closes
                popup = input("Enable overlay after closing GUI? [y/N]: ").strip().lower()
                if popup == 'y':
                    gui.enable_pygame_overlay()
                gui.run()
                break
            elif mode == "2":
                use_pygame = True
                break
            elif mode.lower() == "q":
                print("Exiting.")
                return
        if use_pygame:
            self.start_detection()  # Ensure detection/AI loop runs in background
            region = self.config_manager.get_screen_region()
            fov = region.get('width', 280)
            self.overlay = PygameOverlay(width=region.get('width', 960), height=region.get('height', 720), fov=fov)
            def overlay_loop():
                while self.overlay is not None and getattr(self.overlay, 'running', False):
                    frame = get_frame(region)
                    if frame is not None and isinstance(frame, np.ndarray):
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
                        if self.overlay is not None and hasattr(self.overlay, 'screen') and self.overlay.screen is not None:
                            self.overlay.screen.blit(surf, (0, 0))
                    else:
                        # If no frame, fill with a blank background
                        self.overlay.screen.fill((30, 30, 30))
                    pygame.display.flip()
                    if self.overlay is not None and hasattr(self.overlay, 'clock') and self.overlay.clock is not None:
                        self.overlay.clock.tick(60)
            t = threading.Thread(target=overlay_loop, daemon=True)
            t.start()
            # Do not block main thread; let overlay run in background
            print("Pygame overlay started in background thread")

    def toggle_system(self):
        self.toggle_aimbot(not self.running)

    def toggle_aimbot(self, state):
        if state:
            self.start_detection()
        else:
            self.stop_detection()

    def start_detection(self):
        if not self.running:
            self.running = True
            if self.model is None:
                self.detector = YOLOv8Detector("configs/default_config.json")
                self.model = self.detector.model if hasattr(self.detector, 'model') else None
            if self.predictor is None:
                self.predictor = TargetPredictor(self.config)
            region = self.config.get("screen_region") or {"left": 0, "top": 0, "width": 280, "height": 280}
            self.monitor = {
                "top": int(region.get("top", 0)),
                "left": int(region.get("left", 0)),
                "width": int(region.get("width", 280)),
                "height": int(region.get("height", 280))
            }
            if self.frame_queue is None:
                self.frame_queue = queue.Queue(maxsize=2)
            # Only start detection_thread, not capture_thread
            self.detect_thread = threading.Thread(target=self.detection_loop, daemon=True)
            self.detect_thread.start()

    def stop_detection(self):
        self.running = False

    def get_live_detections(self, frame):
        if self.detector is None:
            return []
        return self.detector.detect(frame)

    def detection_loop(self):
        logger.info("Detection loop started")
        if self.input_ctrl is None:
            self.input_ctrl = InputController(self.config)
            self.input_ctrl.connect()
        if self.predictor is None:
            self.predictor = TargetPredictor(self.config)
        region = self.config.get("screen_region") or {"left": 0, "top": 0, "width": 280, "height": 280}
        left = int(region.get("left", 0))
        top = int(region.get("top", 0))
        w = int(region.get("width", 280))
        h = int(region.get("height", 280))
        monitor = {"left": left, "top": top, "width": w, "height": h}
        with mss.mss() as sct:
            while self.running:
                try:
                    frame = None
                    try:
                        img = sct.grab(monitor)
                        frame = np.array(img)
                        if frame.shape[2] == 4:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                        frame = np.ascontiguousarray(frame)
                    except Exception as e:
                        logger.error(f"Screen capture error: {e}")
                        frame = None
                    detections = []
                    results = None
                    if frame is not None and self.detector is not None:
                        try:
                            results = self.detector.model(frame, verbose=False)[0]
                        except Exception as e:
                            logger.error(f"YOLOv8 inference error: {e}")
                            results = None
                    # Robust detection extraction
                    if results is not None and hasattr(results, 'boxes') and results.boxes is not None:
                        boxes = results.boxes
                        if hasattr(boxes, 'xyxy') and boxes.xyxy is not None and len(boxes.xyxy) > 0:
                            for box in boxes:
                                if box.xyxy is None or len(box.xyxy.shape) != 2 or box.xyxy.shape[0] < 1:
                                    continue
                                class_id = int(box.cls.item()) if hasattr(box, 'cls') else -1
                                confidence = float(box.conf.item()) if hasattr(box, 'conf') else 0.0
                                x1, y1, x2, y2 = box.xyxy[0].tolist()
                                detections.append({
                                    'box': (x1, y1, x2, y2),
                                    'confidence': confidence,
                                    'class_id': class_id
                                })
                        else:
                            logger.debug("No detections found in this frame.")
                    else:
                        logger.debug("No boxes found in results.")
                    # --- Send detections to PygameOverlay if available ---
                    if hasattr(self, 'overlay') and self.overlay:
                        self.overlay.set_detections(detections)
                    # Robust target selection and aiming
                    if detections:
                        screen_center = (w // 2, h // 2)
                        best = self.predictor.select_best_target(detections, screen_center)
                        if best:
                            predicted_pos = self.predictor.predict(best)
                            self.input_ctrl.move_to_target(predicted_pos, best)
                    show_capture = self.gui.show_capture_window.get() if self.gui else False
                    if show_capture and frame is not None:
                        for det in detections:
                            x1, y1, x2, y2 = det['box']
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        if isinstance(frame, np.ndarray):
                            hh, ww = frame.shape[:2]
                            cv2.drawMarker(frame, (ww//2, hh//2), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=30, thickness=2)
                            cv2.imshow(self.capture_window_name, frame)
                            cv2.waitKey(1)
                        else:
                            logger.error(f"Frame is not a valid numpy array: type={type(frame)}, value={frame}")
                    else:
                        try:
                            cv2.destroyWindow(self.capture_window_name)
                        except:
                            pass
                    time.sleep(0.016)
                except Exception as e:
                    logger.error(f"Detection loop error: {e}")
                    time.sleep(0.05)

    def __enter__(self):
        # Context manager entry: start system
        logger.info("CVTargetingSystem context entered.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Context manager exit: cleanup resources
        logger.info("CVTargetingSystem context exiting. Cleaning up resources...")
        self.cleanup()
        return False  # Do not suppress exceptions

    def cleanup(self):
        # Proper resource cleanup for overlay, OpenCV windows, threads, etc.
        if self.overlay:
            try:
                self.overlay.running = False
                # If overlay has a cleanup method, call it
                if hasattr(self.overlay, 'cleanup'):
                    self.overlay.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up overlay: {e}")
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        # Stop detection thread
        self.running = False
        if self.detect_thread and self.detect_thread.is_alive():
            self.detect_thread.join(timeout=2)
        # Disconnect input controller
        if self.input_ctrl and hasattr(self.input_ctrl, 'disconnect'):
            try:
                self.input_ctrl.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting input controller: {e}")
        logger.info("CVTargetingSystem cleanup complete.")

def np_to_surface(frame):
    # Convert BGR (OpenCV) to RGB (Pygame)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))

# Removed local PygameOverlay class to avoid type conflict with imported PygameOverlay.

if __name__ == "__main__":
    print("=== Entered main ===")
    logger.info("Script started")
    system = CVTargetingSystem()
    print("CVTargetingSystem instantiated")
    logger.info("CVTargetingSystem instantiated")
    system.run()
    print("system.run() returned")
    logger.info("system.run() returned")
