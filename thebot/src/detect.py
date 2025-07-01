"""
YOLOv8 Detector Module
Purpose: Provides clean interface for YOLOv8 object detection and prediction in mouse control system
"""

from ultralytics import YOLO
import json
from pathlib import Path
import numpy as np
import cv2
from utils import Logger


# NOTE: The following values are the model's trained targeting info for detections:
# names: ['fn - v1 2023-11-26 7-24am']
# target_class_id: 0
# classes: ['fn - v1 2023-11-26 7-24am']

class YOLOv8Detector:
    """
    YOLOv8 Detection Engine

    Key Features:
    - Streamlined model initialization
    - Configurable detection parameters
    - Optimized inference pipeline
    - Integrated prediction logic (target selection, smoothing, Kalman filter)
    """

    def __init__(self, config_path="config/settings.json"):
        """Initialize YOLOv8 detector with configuration"""
        self.config = self._load_config(config_path)
        self.model = self._initialize_model()
        # --- Prediction logic state ---
        self.smoothing_factor = self.config['yolo'].get('smoothing_factor', 0.3)
        self.aim_height_offset = self.config['yolo'].get('aim_height_offset', 0.0)
        self.target_class_ids = set(self.config['yolo'].get('target_classes', [0]))
        self.last_target_pos = None
        self.kalman = cv2.KalmanFilter(4, 2)
        self._init_kalman()

    def _load_config(self, config_path):
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            return json.load(f)

    def _initialize_model(self):
        """Initialize YOLOv8 model"""
        model_path = self.config['yolo']['model_path']
        device = self.config['yolo']['device']
        model = YOLO(model_path)
        model.to(device)
        return model

    def _init_kalman(self):
        """Initializes the Kalman filter."""
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32) * 1

    def detect(self, input_image):
        """
        Run detection on image

        Args:
            input_image: numpy array (BGR format)

        Returns:
            list: Filtered detections [{'box': (x1, y1, x2, y2), 'confidence': conf, 'class_id': class_id}, ...]
        """
        print(f"[DEBUG] detection input shape: {input_image.shape}")
        conf_threshold = self.config['yolo']['confidence_threshold']
        target_classes = self.target_class_ids
        imgsz = self.config['yolo'].get('imgsz', 640)

        results = self.model(input_image, conf=conf_threshold, imgsz=imgsz)
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                class_id = int(box.cls)
                confidence = float(box.conf)
                if class_id in target_classes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append({
                        'box': (x1, y1, x2, y2),
                        'confidence': confidence,
                        'class_id': class_id
                    })
        print(f"[DEBUG] detection result: {results}")
        return results

    # --- Prediction logic integration ---

    def select_best_target(self, detections, screen_center=None):
        """
        Selects the best target from a list of detections within a 300px radius center region.
        Returns the detection dict with added 'center', 'distance', 'height'.
        """
        if not detections:
            return None
        # Use a 300px radius from the center of a 1600x1024 screen (or override)
        if screen_center is None:
            center_x, center_y = 1600 // 2, 1024 // 2
        else:
            center_x, center_y = screen_center
        radius = 300

        valid_targets = []
        for det in detections:
            x1, y1, x2, y2 = det['box']
            center_x_det = (x1 + x2) / 2
            center_y_det = (y1 + y2) / 2
            distance = np.sqrt((center_x_det - center_x) ** 2 + (center_y_det - center_y) ** 2)
            if distance <= radius:
                det['center'] = (center_x_det, center_y_det)
                det['distance'] = distance
                det['height'] = y2 - y1
                valid_targets.append(det)
        if not valid_targets:
            return None
        return min(valid_targets, key=lambda t: t['distance'])

    def predict(self, target):
        """
        Predicts and smooths the target's position using Kalman filter and smoothing.
        Returns: {'x': int, 'y': int}
        """
        target_center_x, target_center_y = target['center']
        aim_height = target['height'] * self.aim_height_offset
        current_pos = np.array([target_center_x, target_center_y - aim_height], dtype=np.float32)
        self.kalman.correct(current_pos)
        prediction = self.kalman.predict()
        if self.last_target_pos is not None:
            smoothed_pos = self.last_target_pos + self.smoothing_factor * (prediction[:2].flatten() - self.last_target_pos)
        else:
            smoothed_pos = prediction[:2].flatten()
        self.last_target_pos = smoothed_pos
        return {'x': int(smoothed_pos[0]), 'y': int(smoothed_pos[1])}

    def reset_predictor(self):
        """Resets the predictor's state."""
        self.last_target_pos = None
        self._init_kalman()

    def get_best_detection(self, detections):
        """
        Select highest confidence detection (legacy API).
        """
        if not detections:
            return None
        return max(detections, key=lambda d: d['confidence'])

    def calculate_offset(self, detection, screen_width, screen_height):
        """
        Calculate mouse movement offset from detection.
        Returns: (offset_x, offset_y) from screen center.
        """
        if not detection:
            return (0, 0)
        x1, y1, x2, y2 = detection['box']
        det_center_x = (x1 + x2) / 2
        det_center_y = (y1 + y2) / 2
        screen_center_x = screen_width / 2
        screen_center_y = screen_height / 2
        offset_x = det_center_x - screen_center_x
        offset_y = det_center_y - screen_center_y
        return (offset_x, offset_y)

    # Multiplayer Prediction Model stub
    def set_multiplayer_predictor(self, predictor):
        """Attach a multiplayer movement prediction model (e.g., LSTM, transformer)."""
        self.multiplayer_predictor = predictor

    def predict_multiplayer_movement(self, detection_sequence):
        """Predict future positions for multiple players (stub for extension)."""
        if hasattr(self, 'multiplayer_predictor'):
            return self.multiplayer_predictor.predict(detection_sequence)
        # Default: no prediction
        return detection_sequence[-1] if detection_sequence else None


class TargetPredictor:
    """
    Selects the best target and predicts its future position using Kalman filter and smoothing.
    """

    def __init__(self, config):
        self.config = config
        self.max_distance = self.config.get('max_tracking_distance', 300)
        self.aim_offset = self.config.get('aim_height_offset', 0.0)
        self.smoothing_factor = self.config.get('smoothing_factor', 0.3)
        self.target_class_id = self.config.get('target_class_id', 0)
        self.target_class_real_name = self.config.get('target_class_real_name', 'player')
        self.target_class_name = self.config.get('target_class_name', 'player')
        self.last_target_pos = None
        self.kalman = cv2.KalmanFilter(4, 2)
        self._init_kalman()

    def _init_kalman(self):
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32) * 1

    def select_best_target(self, detections, screen_center=None):
        """
        Selects the best target from a list of detections within a 300px radius center region.
        Returns the detection dict with added 'center', 'distance', 'height'.
        """
        if not detections:
            return None
        # Use a 300px radius from the center of a 1600x1024 screen (or override)
        if screen_center is None:
            center_x, center_y = 1600 // 2, 1024 // 2
        else:
            center_x, center_y = screen_center
        radius = 300

        valid_targets = []
        for det in detections:
            # If detection has class info, check against real name or id
            if 'class_id' in det:
                if det['class_id'] != self.target_class_id:
                    continue
            elif 'class_name' in det:
                if det['class_name'] != self.target_class_real_name:
                    continue
            x1, y1, x2, y2 = det['box']
            center_x_det = (x1 + x2) / 2
            center_y_det = (y1 + y2) / 2
            distance = np.sqrt((center_x_det - center_x) ** 2 + (center_y_det - center_y) ** 2)
            if distance <= radius:
                det['center'] = (center_x_det, center_y_det)
                det['distance'] = distance
                det['height'] = y2 - y1
                valid_targets.append(det)
        if not valid_targets:
            return None
        return min(valid_targets, key=lambda t: t['distance'])

    def predict(self, target):
        """
        Predicts and smooths the target's position using Kalman filter and smoothing.
        Returns: {'x': int, 'y': int}
        """
        target_center_x, target_center_y = target['center']
        aim_height = target['height'] * self.aim_offset
        current_pos = np.array([target_center_x, target_center_y - aim_height], dtype=np.float32)
        self.kalman.correct(current_pos)
        prediction = self.kalman.predict()
        if self.last_target_pos is not None:
            smoothed_pos = self.last_target_pos + self.smoothing_factor * (prediction[:2].flatten() - self.last_target_pos)
        else:
            smoothed_pos = prediction[:2].flatten()
        self.last_target_pos = smoothed_pos
        return {'x': int(smoothed_pos[0]), 'y': int(smoothed_pos[1])}

    def reset(self):
        """Resets the predictor's state."""
        self.last_target_pos = None
        self._init_kalman()


# Ensure both YOLOv8Detector and TargetPredictor are defined in this file.
# If you want to use YOLOv8Detector for detection, import and use it in main.py as shown above.
# TargetPredictor is used for prediction/smoothing and is called from main.py.