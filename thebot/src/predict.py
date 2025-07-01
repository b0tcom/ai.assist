import numpy as np
import time
from typing import Tuple, Optional
from capture import CONFIG
from utils import logger

class TargetPredictor:
    """Handles target trajectory prediction and cursor smoothing."""
    def __init__(self) -> None:
        self.last_target_pos: Optional[Tuple[int, int]] = None
        self.last_prediction_time: Optional[float] = None
        self.last_velocity: Optional[Tuple[float, float]] = None
        logger.info("Target predictor initialized.")

    def predict_target_position(self, target_coords: Tuple[int, int], frame_latency_ms: float) -> Tuple[int, int]:
        """
        Predicts target position using velocity and acceleration.
        Replace this logic with a Kalman filter for better prediction if needed.

        Parameters:
            target_coords (Tuple[int, int]): Current target coordinates.
            frame_latency_ms (float): Frame latency in milliseconds.

        Returns:
            Tuple[int, int]: Predicted target coordinates.
        """
        # Input validation
        if not (isinstance(target_coords, (list, tuple)) and len(target_coords) == 2):
            raise ValueError("target_coords must be a tuple or list of length 2.")
        if not isinstance(frame_latency_ms, (int, float)):
            raise ValueError("frame_latency_ms must be a number.")

        if self.last_target_pos is None or self.last_prediction_time is None:
            self.last_target_pos = target_coords
            self.last_prediction_time = time.perf_counter()
            self.last_velocity = (0.0, 0.0)
            return target_coords

        current_time = time.perf_counter()
        time_delta = current_time - self.last_prediction_time

        # Simple velocity calculation
        if time_delta > 0:
            velocity_x = (target_coords[0] - self.last_target_pos[0]) / time_delta
            velocity_y = (target_coords[1] - self.last_target_pos[1]) / time_delta
        else:
            velocity_x = velocity_y = 0.0

        # Calculate acceleration if previous velocity exists
        if self.last_velocity is not None and time_delta > 0:
            accel_x = (velocity_x - self.last_velocity[0]) / time_delta
            accel_y = (velocity_y - self.last_velocity[1]) / time_delta
        else:
            accel_x = accel_y = 0.0

        latency_seconds = frame_latency_ms / 1000.0

        # --- FIX: CONFIG may be None or not subscriptable, so use defaults if needed ---
        prediction_factor = 0.1
        if CONFIG and isinstance(CONFIG, dict) and 'prediction_factor' in CONFIG:
            prediction_factor = float(CONFIG['prediction_factor'])

        predicted_x = (target_coords[0] +
                       velocity_x * latency_seconds * prediction_factor +
                       0.5 * accel_x * (latency_seconds ** 2))
        predicted_y = (target_coords[1] +
                       velocity_y * latency_seconds * prediction_factor +
                       0.5 * accel_y * (latency_seconds ** 2))

        self.last_target_pos = target_coords
        self.last_prediction_time = current_time
        self.last_velocity = (velocity_x, velocity_y)

        return int(round(predicted_x)), int(round(predicted_y))

    def smooth_cursor_movement(self, current_pos: Tuple[int, int], target_pos: Tuple[int, int]) -> Tuple[int, int]:
        """
        Calculates smoothed movement steps towards the target position.

        Parameters:
            current_pos (Tuple[int, int]): Current cursor position.
            target_pos (Tuple[int, int]): Target cursor position.

        Returns:
            Tuple[int, int]: Movement step (x, y) applied after smoothing.
        """
        # Input validation for positions
        for pos in (current_pos, target_pos):
            if not (isinstance(pos, (list, tuple)) and len(pos) == 2):
                raise ValueError("Positions must be a tuple or list of length 2.")

        # --- FIX: CONFIG may be None or not subscriptable, so use defaults if needed ---
        smoothing_factor = 0.1
        if CONFIG and isinstance(CONFIG, dict) and 'smoothing_factor' in CONFIG:
            smoothing_factor = float(CONFIG['smoothing_factor'])

        diff_x = target_pos[0] - current_pos[0]
        diff_y = target_pos[1] - current_pos[1]

        move_x = diff_x * smoothing_factor
        move_y = diff_y * smoothing_factor

        return int(round(move_x)), int(round(move_y))

    def reset_prediction(self) -> None:
        """
        Resets the target prediction state.
        """
        self.last_target_pos = None
        self.last_prediction_time = None
        self.last_velocity = None