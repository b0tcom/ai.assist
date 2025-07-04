"""
Prediction Backward Compatibility Module
Purpose: Maintains compatibility with legacy code expecting the old TargetPredictor class
Author: AI Gaming Analysis System
Version: 2.0.0
"""

import warnings
import numpy as np
import time
from typing import Tuple, Optional, Dict, Any, Union, List
from dataclasses import dataclass

# Import from new modules
try:
    from .detect import Detection, YOLOv8Detector
    from .config_manager import ConfigManager
    from .logger_util import get_logger
except ImportError as e:
    raise ImportError(f"Required modules not found: {e}. Ensure detect.py and config modules are available.")

# Constants
DEFAULT_BOUNDING_BOX_SIZE = 50
DEFAULT_SMOOTHING_FACTOR = 0.1
DEFAULT_FRAME_LATENCY_MS = 16.67  # ~60 FPS
MIN_SMOOTHING_FACTOR = 0.01
MAX_SMOOTHING_FACTOR = 1.0

# Show deprecation warning once per session
warnings.warn(
    "predict.py is deprecated. Target prediction is now integrated into detect.py. "
    "Please migrate to the new API for future compatibility.",
    DeprecationWarning,
    stacklevel=2
)

class SimpleTargetPredictor:
    """Simple target predictor for backward compatibility"""
    
    def __init__(self, config_manager: Optional[ConfigManager]):
        self.config_manager = config_manager
        self.logger = get_logger(__name__)
    
    def predict(self, detection: Detection) -> Dict[str, Any]:
        """Simple prediction based on center of detection"""
        x1, y1, x2, y2 = detection.box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return {'x': center_x, 'y': center_y}
    
    def select_best_target(self, detections: List[Detection], screen_center: Tuple[int, int]) -> Optional[Detection]:
        """Select closest target to screen center"""
        if not detections:
            return None
        
        best_detection = None
        min_distance = float('inf')
        
        for detection in detections:
            x1, y1, x2, y2 = detection.box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            distance = ((center_x - screen_center[0])**2 + (center_y - screen_center[1])**2)**0.5
            
            if distance < min_distance:
                min_distance = distance
                best_detection = detection
        
        return best_detection
    
    def reset(self):
        """Reset predictor state"""
        pass

@dataclass
class LegacyDetection:
    """Legacy detection format for backward compatibility"""
    box: Tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str = 'unknown'
    center: Optional[Tuple[float, float]] = None
    distance: Optional[float] = None
    height: Optional[float] = None

    def __post_init__(self):
        """Calculate derived properties"""
        if self.center is None:
            x1, y1, x2, y2 = self.box
            self.center = ((x1 + x2) / 2, (y1 + y2) / 2)
        
        if self.height is None:
            _, y1, _, y2 = self.box
            self.height = y2 - y1


class ConfigCompatibilityError(Exception):
    """Raised when configuration compatibility issues occur"""
    pass


class TargetPredictor:
    """
    Legacy TargetPredictor wrapper around new detection system.
    Provides backward compatibility for existing code while maintaining
    professional standards and error handling.
    """
    
    def __init__(self, config: Optional[Union[Dict, Any]] = None) -> None:
        """
        Initialize legacy target predictor with robust config handling.
        
        Args:
            config: Configuration object (dict, Config wrapper, or ConfigManager)
            
        Raises:
            ConfigCompatibilityError: If config cannot be properly initialized
        """
        self.logger = get_logger(__name__)
        self._initialize_config(config)
        self._initialize_predictor()
        self._reset_state()
        
        self.logger.info("Legacy TargetPredictor initialized successfully")
    
    def _initialize_config(self, config: Optional[Union[Dict, Any]]) -> None:
        """Initialize configuration manager with proper error handling."""
        try:
            # Try to import legacy CONFIG as fallback
            if config is None:
                try:
                    from config import CONFIG
                    config = CONFIG
                except ImportError:
                    config = {}
            
            # Handle different config types
            if isinstance(config, dict):
                self.config_manager = ConfigManager()
                self._map_legacy_config(config)
            elif config is not None and hasattr(config, '_manager'):
                # New Config wrapper detected
                self.config_manager = config._manager
            elif hasattr(config, 'get_aim_config'):
                # Assume it's already a ConfigManager
                self.config_manager = config
            else:
                raise ConfigCompatibilityError(f"Unsupported config type: {type(config)}")
                
        except Exception as e:
            self.logger.error(f"Config initialization failed: {e}")
            # Fallback to default config
            self.config_manager = ConfigManager()
    
    def _map_legacy_config(self, legacy_config: Dict) -> None:
        """Map legacy configuration values to new format."""
        mapping = {
            'prediction_factor': ('aim_settings', 'smoothing_factor'),
            'smoothing_factor': ('aim_settings', 'smoothing_factor'),
            'max_tracking_distance': ('aim_settings', 'max_distance'),
            'aim_height_offset': ('aim_settings', 'altura_tiro'),
        }
        
        for legacy_key, (section, new_key) in mapping.items():
            if legacy_key in legacy_config:
                try:
                    if self.config_manager is not None:
                        self.config_manager.set(section, new_key, legacy_config[legacy_key])
                except Exception as e:
                    self.logger.warning(f"Failed to map config {legacy_key}: {e}")
    
    def _initialize_predictor(self) -> None:
        """Initialize the new predictor with error handling."""
        try:
            if self.config_manager is not None:
                self._predictor = SimpleTargetPredictor(self.config_manager)
            else:
                raise ConfigCompatibilityError("Config manager is not initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize new predictor: {e}")
            raise ConfigCompatibilityError(f"Predictor initialization failed: {e}")
    
    def _reset_state(self) -> None:
        """Reset internal state variables."""
        self.last_target_pos: Optional[Tuple[int, int]] = None
        self.last_prediction_time: Optional[float] = None
        self.last_velocity: Optional[Tuple[float, float]] = None
    
    def predict_target_position(self, 
                              target_coords: Tuple[int, int], 
                              frame_latency_ms: float) -> Tuple[int, int]:
        """
        Legacy prediction method with enhanced input validation.
        
        Args:
            target_coords: Current target coordinates (x, y)
            frame_latency_ms: Frame latency in milliseconds
            
        Returns:
            Predicted target coordinates (x, y)
            
        Raises:
            ValueError: If input parameters are invalid
        """
        # Comprehensive input validation
        self._validate_coordinates(target_coords)
        self._validate_latency(frame_latency_ms)
        
        try:
            # Create detection object for new system
            fake_detection = self._create_fake_detection(target_coords)
            
            # Use new predictor
            result = self._predictor.predict(fake_detection)
            
            # Update legacy state
            self._update_legacy_state(target_coords)
            
            return int(result['x']), int(result['y'])
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            # Fallback to simple prediction
            return self._fallback_prediction(target_coords, frame_latency_ms)
    
    def _validate_coordinates(self, coords: Tuple[int, int]) -> None:
        """Validate coordinate input."""
        if not isinstance(coords, (list, tuple)) or len(coords) != 2:
            raise ValueError("target_coords must be a tuple or list of length 2")
        
        if not all(isinstance(coord, (int, float)) for coord in coords):
            raise ValueError("Coordinates must be numeric values")
        
        # Basic range validation (assuming standard screen sizes)
        if not (0 <= coords[0] <= 8192 and 0 <= coords[1] <= 8192):
            self.logger.warning(f"Coordinates may be out of normal range: {coords}")
    
    def _validate_latency(self, latency: float) -> None:
        """Validate latency input."""
        if not isinstance(latency, (int, float)):
            raise ValueError("frame_latency_ms must be a number")
        
        if latency < 0:
            raise ValueError("frame_latency_ms cannot be negative")
        
        if latency > 1000:  # 1 second seems excessive
            self.logger.warning(f"Unusually high latency: {latency}ms")
    
    def _create_fake_detection(self, coords: Tuple[int, int]) -> Detection:
        """Create a detection object from coordinates."""
        x, y = coords
        half_size = DEFAULT_BOUNDING_BOX_SIZE // 2
        
        return Detection(
            box=(x - half_size, y - half_size, x + half_size, y + half_size),
            confidence=1.0,
            class_id=0,
            class_name='legacy_target'
        )
    
    def _update_legacy_state(self, coords: Tuple[int, int]) -> None:
        """Update legacy state variables."""
        current_time = time.perf_counter()
        
        if self.last_target_pos is not None and self.last_prediction_time is not None:
            time_delta = current_time - self.last_prediction_time
            if time_delta > 0:
                dx = coords[0] - self.last_target_pos[0]
                dy = coords[1] - self.last_target_pos[1]
                self.last_velocity = (dx / time_delta, dy / time_delta)
        
        self.last_target_pos = coords
        self.last_prediction_time = current_time
    
    def _fallback_prediction(self, coords: Tuple[int, int], latency_ms: float) -> Tuple[int, int]:
        """Simple fallback prediction when main predictor fails."""
        if self.last_velocity is None:
            return coords
        
        # Simple linear prediction
        time_delta = latency_ms / 1000.0  # Convert to seconds
        predicted_x = coords[0] + self.last_velocity[0] * time_delta
        predicted_y = coords[1] + self.last_velocity[1] * time_delta
        
        return int(predicted_x), int(predicted_y)
    
    def smooth_cursor_movement(self, 
                             current_pos: Tuple[int, int], 
                             target_pos: Tuple[int, int]) -> Tuple[int, int]:
        """
        Enhanced cursor movement smoothing with proper error handling.
        
        Args:
            current_pos: Current cursor position (x, y)
            target_pos: Target cursor position (x, y)
            
        Returns:
            Movement step (x, y)
            
        Raises:
            ValueError: If positions are invalid
        """
        # Validate inputs
        for pos_name, pos in [('current_pos', current_pos), ('target_pos', target_pos)]:
            if not isinstance(pos, (list, tuple)) or len(pos) != 2:
                raise ValueError(f"{pos_name} must be a tuple or list of length 2")
            if not all(isinstance(coord, (int, float)) for coord in pos):
                raise ValueError(f"{pos_name} coordinates must be numeric")
        
        try:
            # Get smoothing factor with safe fallback
            smoothing_factor = self._get_smoothing_factor()
            
            # Calculate movement
            diff_x = target_pos[0] - current_pos[0]
            diff_y = target_pos[1] - current_pos[1]
            
            move_x = diff_x * smoothing_factor
            move_y = diff_y * smoothing_factor
            
            return int(round(move_x)), int(round(move_y))
            
        except Exception as e:
            self.logger.error(f"Smoothing calculation failed: {e}")
            # Return minimal movement as fallback
            return (0, 0)
    
    def _get_smoothing_factor(self) -> float:
        """Get smoothing factor with safe fallback."""
        try:
            if self.config_manager is not None:
                aim_config = self.config_manager.get_aim_config()
                if hasattr(aim_config, 'smoothing_factor'):
                    factor = aim_config.smoothing_factor
                else:
                    factor = getattr(aim_config, 'smoothing_factor', DEFAULT_SMOOTHING_FACTOR)
                
                # Validate range
                return max(MIN_SMOOTHING_FACTOR, min(MAX_SMOOTHING_FACTOR, factor))
            else:
                return DEFAULT_SMOOTHING_FACTOR
            
        except Exception as e:
            self.logger.warning(f"Failed to get smoothing factor: {e}")
            return DEFAULT_SMOOTHING_FACTOR
    
    def reset_prediction(self) -> None:
        """Reset prediction state with proper error handling."""
        try:
            self._predictor.reset()
        except Exception as e:
            self.logger.error(f"Failed to reset predictor: {e}")
        
        self._reset_state()
        self.logger.debug("Prediction state reset")
    
    def select_best_target(self, 
                          detections: List[Union[Dict, Detection, LegacyDetection]], 
                          screen_center: Tuple[int, int]) -> Optional[Dict[str, Any]]:
        """
        Enhanced target selection with robust detection handling.
        
        Args:
            detections: List of detection objects in various formats
            screen_center: Screen center coordinates (x, y)
            
        Returns:
            Best detection in legacy format or None
        """
        if not detections:
            return None
        
        try:
            # Convert all detections to new format
            new_detections = []
            for det in detections:
                converted = self._convert_detection(det)
                if converted:
                    new_detections.append(converted)
            
            if not new_detections:
                return None
            
            # Use new predictor
            best = self._predictor.select_best_target(new_detections, screen_center)
            
            if best:
                return self._convert_to_legacy_format(best)
            
        except Exception as e:
            self.logger.error(f"Target selection failed: {e}")
        
        return None
    
    def _convert_detection(self, det: Union[Dict, Detection, Any]) -> Optional[Detection]:
        """Convert various detection formats to new Detection format."""
        try:
            if isinstance(det, Detection):
                return det
            elif isinstance(det, dict):
                return Detection(
                    box=det.get('box', (0, 0, 0, 0)),
                    confidence=det.get('confidence', 0.0),
                    class_id=det.get('class_id', 0),
                    class_name=det.get('class_name', 'unknown')
                )
            elif hasattr(det, 'box') and hasattr(det, 'confidence'):
                # Handle custom detection objects
                return Detection(
                    box=getattr(det, 'box', (0, 0, 0, 0)),
                    confidence=getattr(det, 'confidence', 0.0),
                    class_id=getattr(det, 'class_id', 0),
                    class_name=getattr(det, 'class_name', 'unknown')
                )
        except Exception as e:
            self.logger.warning(f"Failed to convert detection: {e}")
        
        return None
    
    def _convert_to_legacy_format(self, detection: Detection) -> Dict[str, Any]:
        """Convert new Detection to legacy format."""
        legacy_det = LegacyDetection(
            box=detection.box,
            confidence=detection.confidence,
            class_id=detection.class_id,
            class_name=getattr(detection, 'class_name', 'unknown')
        )
        
        # Calculate distance if screen center is available
        distance = getattr(detection, 'distance', None)
        if distance is None and hasattr(detection, 'center'):
            # Could calculate distance here if needed
            distance = 0
        
        return {
            'box': legacy_det.box,
            'confidence': legacy_det.confidence,
            'class_id': legacy_det.class_id,
            'class_name': legacy_det.class_name,
            'center': legacy_det.center,
            'distance': distance or 0,
            'height': legacy_det.height
        }


# Convenience functions for backward compatibility
def predict_target_position(target_coords: Tuple[int, int], 
                          frame_latency_ms: float,
                          config: Optional[Any] = None) -> Tuple[int, int]:
    """
    Legacy standalone prediction function with enhanced error handling.
    
    Args:
        target_coords: Current target coordinates
        frame_latency_ms: Frame latency in milliseconds
        config: Optional configuration
        
    Returns:
        Predicted target coordinates
        
    Raises:
        ValueError: If input parameters are invalid
    """
    predictor = TargetPredictor(config)
    return predictor.predict_target_position(target_coords, frame_latency_ms)


def smooth_cursor_movement(current_pos: Tuple[int, int], 
                         target_pos: Tuple[int, int],
                         smoothing_factor: float = DEFAULT_SMOOTHING_FACTOR) -> Tuple[int, int]:
    """
    Legacy standalone smoothing function with validation.
    
    Args:
        current_pos: Current cursor position
        target_pos: Target cursor position
        smoothing_factor: Smoothing factor (0-1)
        
    Returns:
        Movement step (x, y)
        
    Raises:
        ValueError: If parameters are invalid
    """
    # Validate smoothing factor
    if not isinstance(smoothing_factor, (int, float)):
        raise ValueError("smoothing_factor must be a number")
    
    smoothing_factor = max(MIN_SMOOTHING_FACTOR, min(MAX_SMOOTHING_FACTOR, smoothing_factor))
    
    # Validate positions
    for pos_name, pos in [('current_pos', current_pos), ('target_pos', target_pos)]:
        if not isinstance(pos, (list, tuple)) or len(pos) != 2:
            raise ValueError(f"{pos_name} must be a tuple or list of length 2")
    
    diff_x = target_pos[0] - current_pos[0]
    diff_y = target_pos[1] - current_pos[1]
    
    move_x = diff_x * smoothing_factor
    move_y = diff_y * smoothing_factor
    
    return int(round(move_x)), int(round(move_y))