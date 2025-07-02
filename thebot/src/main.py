import json
import logging
import threading
import time
import tkinter as tk
import numpy as np
import mss
import queue
import cv2
import configparser  # <-- Add this
# Add win32api for screen size detection
try:
    import win32api
except ImportError:
    win32api = None

from input_handler import InputController
from capture import ScreenCapture
from detect import TargetPredictor, YOLOv8Detector
from gui import CVTargetingGUI
from toggle import ToggleManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')


class CVTargetingSystem:
    """Main class for the AI Aim Assist System. Handles config, detection, and hardware integration."""

    def __init__(self):
        self.config = {}
        self.model = None
        self.capture = None
        self.detector = None
        self.toggle_mgr = None
        self.frame_queue = queue.Queue(maxsize=2)  # Always initialized
        self.result_queue = None
        self.capture_thread = None
        self.detect_thread = None
        self.monitor = None
        self.running = False
        self.detection_thread = None
        self.predictor = None
        self.input_ctrl = None
        self.gui = None
        self.capture_window_name = "Live Detection"

    def get_screen_size(self):
        """Detect the real screen size using win32api, fallback to mss if unavailable."""
        if win32api:
            return win32api.GetSystemMetrics(0), win32api.GetSystemMetrics(1)
        else:
            with mss.mss() as sct:
                mon = sct.monitors[1]
                return mon['width'], mon['height']

    def get_centered_region(self, fov=None):
        """Return a centered region dict using the real screen size and given FOV."""
        screen_w, screen_h = self.get_screen_size()
        if fov is None:
            fov = self.config.get('fov_size', 280)
        left = (screen_w - fov) // 2
        top = (screen_h - fov) // 2
        # Clamp to ensure inside bounds
        if left < 0:
            left = 0
        if top < 0:
            top = 0
        if left + fov > screen_w:
            left = screen_w - fov
        if top + fov > screen_h:
            top = screen_h - fov
        region = {
            'left': int(left),
            'top': int(top),
            'width': int(fov),
            'height': int(fov)
        }
        logger.info(
            f"[AUTO] Centered region for {screen_w}x{screen_h} FOV={fov}: {region}"
        )
        return region

    def load_config(self):
        """Load configuration from INI file. Always use a centered region and all advanced settings."""
        config_path = "configs/config.ini"
        parser = configparser.ConfigParser()
        parser.read(config_path)
        # Build config dict from INI
        # Use FOV from config, but always recalculate region at runtime
        fov = parser.getint('Application',
                            'capture_region_width',
                            fallback=280)
        region = self.get_centered_region(fov)
        logger.info(f"[INFO] Using capture region: {region}")

        # Parse advanced settings from INI
        def parse_json_or_str(val, fallback):
            try:
                return json.loads(val)
            except Exception:
                return fallback

        self.config = {
            "yolo": {
                "model_path":
                parser.get('model_settings',
                           'model_path',
                           fallback="thebot/src/models/best.pt"),
                "confidence_threshold":
                parser.getfloat('model_settings', 'confidence', fallback=0.4),
                "device":
                parser.get('model_settings', 'device', fallback="cuda")
            },
            "target_class_id":
            parser.getint('Application', 'target_class_id', fallback=0),
            "target_class_name":
            "player",
            "fallback_model_path":
            parser.get('model_settings',
                       'fallback_model_path',
                       fallback="thebot/src/models/yolo/yolov8n.pt"),
            "precision_mode":
            parser.get('model_settings', 'precision_mode', fallback="float32"),
            "warmup_iterations":
            parser.getint('model_settings', 'warmup_iterations', fallback=10),
            "target_priority":
            parser.getint('model_settings', 'target_priority', fallback=1),
            "detection_mode":
            parser.get('model_settings', 'detection_mode',
                       fallback="tracking"),
            "min_player_size": [
                int(x) for x in parser.get('model_settings',
                                           'min_player_size',
                                           fallback="10,10").split(',')
            ],
            "max_player_size": [
                int(x) for x in parser.get('model_settings',
                                           'max_player_size',
                                           fallback="500,500").split(',')
            ],
            "ethical_mode":
            parser.get('model_settings', 'ethical_mode',
                       fallback="production"),
            "arduino_port":
            parser.get('arduino', 'arduino_port', fallback="COM5"),
            "screen_region":
            region,
            "fov_size":
            fov,
            # Advanced settings
            "aim_settings": {
                "sensitivity":
                parser.getfloat('aim_settings', 'sensitivity', fallback=1.0),
                "max_distance":
                parser.getint('aim_settings', 'max_distance', fallback=500),
                "shooting_height_ratios":
                parse_json_or_str(
                    parser.get(
                        'aim_settings',
                        'shooting_height_ratios',
                        fallback='{"head":0.15,"neck":0.25,"chest":0.35}'), {
                            "head": 0.15,
                            "neck": 0.25,
                            "chest": 0.35
                        }),
                "altura_tiro":
                parser.getfloat('aim_settings', 'altura_tiro', fallback=1.5),
                "target_areas":
                parse_json_or_str(
                    parser.get('aim_settings',
                               'target_areas',
                               fallback='["head","neck","chest"]'),
                    ["head", "neck", "chest"]),
                "smoothing_factor":
                parser.getfloat('aim_settings', 'smoothing_factor',
                                fallback=5),
                "fov_size":
                parser.getfloat('aim_settings', 'fov_size', fallback=280)
            },
            "mode_settings": {
                "right_click":
                parse_json_or_str(
                    parser.get(
                        'mode_settings',
                        'right_click',
                        fallback='{"click_button": "0x02", "sensitivity": 1.0}'
                    ), {
                        "click_button": "0x02",
                        "sensitivity": 1.0
                    }),
                "left_click":
                parse_json_or_str(
                    parser.get(
                        'mode_settings',
                        'left_click',
                        fallback='{"click_button": "0x01", "sensitivity": 0.5}'
                    ), {
                        "click_button": "0x01",
                        "sensitivity": 0.5
                    })
            },
            "key_bindings": {
                "activation_key":
                parser.get('key_bindings', 'activation_key', fallback='f1'),
                "deactivation_key":
                parser.get('key_bindings', 'deactivation_key', fallback='f2')
            },
            "anti_recoil": {
                "enabled":
                parser.getboolean('anti_recoil', 'enabled', fallback=True),
                "vertical_strength":
                parser.getfloat('anti_recoil',
                                'vertical_strength',
                                fallback=0.5),
                "horizontal_strength":
                parser.getfloat('anti_recoil',
                                'horizontal_strength',
                                fallback=0.2),
                "pattern_enabled":
                parser.getboolean('anti_recoil',
                                  'pattern_enabled',
                                  fallback=False)
            },
            "rapid_fire": {
                "enabled":
                parser.getboolean('rapid_fire', 'enabled', fallback=False),
                "fire_rate":
                parser.getfloat('rapid_fire', 'fire_rate', fallback=0.1)
            },
            "hip_mode_enabled":
            parser.getboolean('hip_mode_enabled', 'enabled', fallback=True),
            "delay":
            parser.getfloat('aim_settings', 'delay', fallback=5e-05),
            "fov_size":
            parser.getfloat('aim_settings', 'fov_size', fallback=280),
            "aim_height":
            parser.getfloat('aim_settings', 'aim_height', fallback=0.25),
            "kalman_filter": {
                "transition_covariance":
                parser.getfloat('aim_settings',
                                'kalman_transition_cov',
                                fallback=0.01),
                "observation_covariance":
                parser.getfloat('aim_settings',
                                'kalman_observation_cov',
                                fallback=0.01)
            }
        }
        self.config["screen_region"] = region

    def validate_model(self):
        """Validate YOLO model is working with test data."""
        if self.model is None:
            logger.error("Model not loaded for validation")
            return False
        test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        try:
            results = self.model(test_img, verbose=False)
            if results is not None:
                if isinstance(results, list):
                    valid = results[0] is not None
                else:
                    valid = True
                if valid:
                    logger.info(
                        f"Model validation successful. Model classes: {getattr(self.model, 'names', None)}"
                    )
                    return True
                else:
                    logger.error(
                        "Model did not return results during validation")
                    return False
            else:
                logger.error("Model did not return results during validation")
                return False
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False

    def run(self):
        logger.info("Starting CVTargetingSystem.run()")
        
        # Check dependencies first
        from utils import check_dependencies
        missing_deps = check_dependencies()
        if missing_deps:
            logger.error(f"Missing critical dependencies: {missing_deps}")
            logger.error("Please install missing packages with: pip install -r requirements.txt")
            return
            
        self.load_config()
        # Connect to Arduino before GUI or detection
        self.input_ctrl = InputController(self.config)
        self.input_ctrl.connect()
        if self.input_ctrl.is_connected():
            logger.info(
                f"[INFO] Successfully connected to Arduino on {self.config.get('arduino_port', 'COM5')}."
            )
        else:
            logger.error("[ERROR] Failed to connect to Arduino. Exiting.")
            return
        self.capture = ScreenCapture(region=self.config.get("screen_region"))
        self.detector = YOLOv8Detector("configs/default_config.json")
        self.model = self.detector.model if hasattr(self.detector,
                                                    'model') else None
        logger.info("Loaded model, about to validate...")
        print("Loaded model, about to validate...")
        if not self.validate_model():
            logger.error("Model validation failed. Exiting.")
            print("Model validation failed. Exiting.")
            return
        logger.info("Model validated, starting GUI...")
        print("Model validated, starting GUI...")
        self.gui = CVTargetingGUI(tk.Tk(), config=self.config)
        self.gui.root.after(100, self.gui.root.update)
        self.toggle_mgr = ToggleManager(on_aimbot_toggle=self.toggle_aimbot)
        self.toggle_mgr.start()
        self.gui.toggle_button.config(command=self.toggle_system)
        self.gui.root.mainloop()

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
                self.model = self.detector.model if hasattr(
                    self.detector, 'model') else None
            if self.predictor is None:
                self.predictor = TargetPredictor(self.config)
            # Arduino is already connected in run()
            region = self.config.get("screen_region", {
                "left": 650,
                "top": 362,
                "width": 300,
                "height": 300
            })
            self.monitor = {
                "top": region.get("top", 362),
                "left": region.get("left", 650),
                "width": region.get("width", 300),
                "height": region.get("height", 300)
            }
            if self.frame_queue is None:
                self.frame_queue = queue.Queue(maxsize=2)
            self.capture_thread = threading.Thread(target=self.capture_loop,
                                                   daemon=True)
            self.detect_thread = threading.Thread(target=self.detect_loop,
                                                  daemon=True)
            self.capture_thread.start()
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
        while self.running:
            if not self.capture or not self.detector:
                logger.error("Capture or detector not initialized!")
                break
            frame = self.capture.capture()
            detections = self.get_live_detections(
                frame) if frame is not None else []
            if detections:
                region = self.config.get("screen_region", {
                    "left": 650,
                    "top": 362,
                    "width": 300,
                    "height": 300
                })
                screen_center = (region["width"] // 2, region["height"] // 2)
                best = self.predictor.select_best_target(
                    detections, screen_center)
                if best:
                    predicted_pos = self.predictor.predict(best)
                    self.input_ctrl.move_to_target(predicted_pos, best)
            show_capture = self.gui.show_capture_window.get(
            ) if self.gui else False
            if show_capture and frame is not None:
                for det in detections:
                    x1, y1, x2, y2 = det['box']
                    cv2.rectangle(frame, (int(x1), int(y1)),
                                  (int(x2), int(y2)), (0, 255, 0), 2)
                if isinstance(frame, np.ndarray):
                    h, w = frame.shape[:2]
                    cv2.drawMarker(frame, (w // 2, h // 2), (0, 0, 255),
                                   markerType=cv2.MARKER_CROSS,
                                   markerSize=30,
                                   thickness=2)
                    cv2.imshow(self.capture_window_name, frame)
                    cv2.waitKey(1)
                else:
                    logger.error(
                        f"Frame is not a valid numpy array: type={type(frame)}, value={frame}"
                    )
            else:
                try:
                    cv2.destroyWindow(self.capture_window_name)
                except:
                    pass
            time.sleep(0.016)

    def capture_loop(self):
        """Fast screen capture loop using bettercam with frame interpolation for target FPS."""
        try:
            import bettercam
        except ImportError:
            logger.error("bettercam module not found. Please install it with: pip install bettercam")
            return
        target_fps = 120  # You can adjust this as needed
        frame_interval = 1.0 / target_fps
        if self.frame_queue is None:
            logger.error("Frame queue is not initialized!")
            return
        try:
            cam = bettercam.create()
        except Exception as e:
            logger.error(
                f"Failed to initialize bettercam in capture thread: {e}")
            return
        last_frame = None
        last_time = time.perf_counter()
        # Always get real screen size and region at runtime
        screen_w, screen_h = self.get_screen_size()
        fov = self.config.get('fov_size', 280)
        while self.running and self.monitor:
            try:
                now = time.perf_counter()
                elapsed = now - last_time
                # Calculate centered region
                left = (screen_w - fov) // 2
                top = (screen_h - fov) // 2
                if left < 0:
                    left = 0
                if top < 0:
                    top = 0
                if left + fov > screen_w:
                    left = screen_w - fov
                if top + fov > screen_h:
                    top = screen_h - fov
                w = min(fov, screen_w - left)
                h = min(fov, screen_h - top)
                region_tuple = (int(left), int(top), int(w), int(h))
                logger.info(f"Sanitized region: {region_tuple}")
                logger.info(f"Region types: {[type(x) for x in region_tuple]}")
                assert all(
                    isinstance(x, int) and x >= 0 for x in region_tuple
                ), f"Region values must be int and >=0: {region_tuple}"
                if elapsed < frame_interval:
                    if last_frame is not None:
                        frame = last_frame.copy()
                    else:
                        time.sleep(frame_interval - elapsed)
                        continue
                else:
                    logger.info(f"About to capture region: {region_tuple}")
                    try:
                        frame = cam.grab(region=region_tuple)
                    except Exception as e:
                        logger.error(
                            f"Grab failed with region={region_tuple}: {e}")
                        if w > 1 and h > 1:
                            logger.info(
                                f"Retrying with w-1/h-1: ({left},{top},{w-1},{h-1})"
                            )
                            try:
                                frame = cam.grab(region=(int(left), int(top),
                                                         int(w - 1),
                                                         int(h - 1)))
                            except Exception as e2:
                                logger.error(
                                    f"Second grab attempt failed: {e2}")
                                frame = None
                        else:
                            frame = None
                    last_time = now
                    if frame is not None:
                        last_frame = frame.copy()
                if frame is not None:
                    if frame.shape[2] == 4:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    frame = np.ascontiguousarray(frame)
                    if not self.frame_queue.full():
                        self.frame_queue.put(frame)
                else:
                    logger.error("Screen capture returned None frame.")
            except Exception as e:
                logger.error(f"Capture error: {e}")
            time.sleep(0)  # Yield thread

    def detect_loop(self):
        """Fast YOLO detection loop, decoupled from capture."""
        if self.frame_queue is None:
            logger.error("Frame queue is not initialized!")
            return
        while self.running and self.model and self.predictor:
            try:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()
                    start = time.perf_counter()
                    results = self.model(frame,
                                         imgsz=128,
                                         device=0,
                                         verbose=False)[0]
                    end = time.perf_counter()
                    logger.info(f"Inference time: {(end-start)*1000:.2f} ms")
                    detections = []
                    if results is not None and hasattr(
                            results, 'boxes') and results.boxes is not None:
                        for box in results.boxes:
                            class_id = int(box.cls.item())
                            confidence = float(box.conf.item())
                            if (class_id == self.config.get(
                                    "target_class_id", 0)
                                    and confidence >= self.config.get(
                                        "yolo", {}).get(
                                            "confidence_threshold", 0.4)):
                                x1, y1, x2, y2 = box.xyxy[0].tolist()
                                detections.append({
                                    'box': (x1, y1, x2, y2),
                                    'confidence': confidence,
                                    'class_id': class_id
                                })
                    # Only send to Arduino if target changed
                    if detections and self.monitor:
                        screen_center = (self.monitor["width"] // 2,
                                         self.monitor["height"] // 2)
                        best = self.predictor.select_best_target(
                            detections, screen_center)
                        if best:
                            try:
                                predicted_pos = self.predictor.predict(best)
                                if self.input_ctrl:
                                    self.input_ctrl.move_to_target(
                                        predicted_pos, best)
                            except Exception as e:
                                logger.error(
                                    f"Arduino communication error: {e}")
                    # Optionally, throttle Arduino writes to 100 Hz
                    time.sleep(0.01)
            except Exception as e:
                logger.error(f"Detection error: {e}")
                time.sleep(0.01)


if __name__ == "__main__":
    system = CVTargetingSystem()
    system.run()
