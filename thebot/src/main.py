import os
import sys
import json
import logging
import threading
import time
from datetime import datetime
import tkinter as tk
import cv2
import numpy as np
from input_handler import InputController
from capture import ScreenCapture
from detect import TargetPredictor, YOLOv8Detector
from gui import CVTargetingGUI
from toggle import ToggleManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('main')

class CVTargetingSystem:
    def __init__(self):
        self.config = {}  # Will be populated by load_config()
        self.model = None
        self.model_type = None
        self.gui = None
        self.running = False
        self.detection_thread = None
        self.arduino_connected = False
        self.capture_window_enabled = False
        self.capture_window_name = "Live Detection"
        self.last_frame = None
        self.input_ctrl = None
        self.predictor = None
        self.capture = None
        self.detector = None
        self.toggle_mgr = None

    def load_config(self):
        screen_w, screen_h = 1600, 1024
        radius = 300
        center_x, center_y = screen_w // 2, screen_h // 2
        region = {
            "left": max(center_x - radius, 0),
            "top": max(center_y - radius, 0),
            "width": min(radius * 2, screen_w),
            "height": min(radius * 2, screen_h)
        }

        default_config = {
            "yolo": {
                "model_path": "thebot/src/models/best.pt",
                "confidence_threshold": 0.4,
                "device": "cuda"
            },
            "target_class_id": 0,
            "target_class_name": "player",
            "fallback_model_path": "thebot/src/models/yolo/yolov8n.pt",
            "precision_mode": "float32",
            "warmup_iterations": 10,
            "target_priority": 1,
            "detection_mode": "tracking",
            "min_player_size": [10, 10],
            "max_player_size": [500, 500],
            "ethical_mode": "production",
            "arduino_port": "COM5",
            "screen_region": region
        }
        config_path = "configs/default_config.json"
        os.makedirs("configs", exist_ok=True)
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = default_config

    def run(self):
        self.load_config()
        self.capture = ScreenCapture(region=self.config.get("screen_region"))
        self.detector = YOLOv8Detector("configs/default_config.json")

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
            self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
            self.detection_thread.start()

    def stop_detection(self):
        self.running = False

    def get_live_detections(self, frame):
        """Run YOLOv8 on frame and return list of dicts with 'box', 'confidence', 'class_id'."""
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
            detections = self.get_live_detections(frame) if frame is not None else []

            if detections:
                screen_center = (1600 // 2, 1024 // 2)
                best = self.predictor.select_best_target(detections, screen_center)
                if best:
                    predicted_pos = self.predictor.predict(best)
                    self.input_ctrl.move_to_target(predicted_pos, best)

            show_capture = self.gui.show_capture_window.get() if self.gui else False
            if show_capture and frame is not None:
                for det in detections:
                    x1, y1, x2, y2 = det['box']
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                h, w = frame.shape[:2]
                cv2.drawMarker(frame, (w//2, h//2), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=30, thickness=2)
                cv2.imshow(self.capture_window_name, frame)
                cv2.waitKey(1)
            else:
                try:
                    cv2.destroyWindow(self.capture_window_name)
                except:
                    pass

            time.sleep(0.016)

if __name__ == "__main__":
    system = CVTargetingSystem()
    system.run()
