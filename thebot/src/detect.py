"""
Production-Grade CUDA-Accelerated YOLOv8 Detection Engine with Enhanced Architecture
Purpose: High-performance object detection with improved dependency management, 
resource handling, and architectural patterns
"""
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple, Any, Protocol
from dataclasses import dataclass
from abc import ABC, abstractmethod
import time
from enum import Enum
from pathlib import Path
from collections import deque
import statistics
import contextlib
from contextlib import contextmanager
import traceback
import threading
import weakref

# Handle both direct execution and module import
try:
    from .logger_util import get_logger
    from .config_manager import ConfigManager, ModelConfig
except ImportError:
    from logger_util import get_logger
    from config_manager import ConfigManager, ModelConfig


class DependencyManager:
    """Centralized dependency management with safe importing"""
    
    def __init__(self):
        self.torch = self._safe_import('torch')
        self.cuda = self._safe_import_attr('torch', 'cuda') if self.torch else None
        self.yolo = self._safe_import('ultralytics', 'YOLO')
        self.onnx = self._safe_import('onnxruntime')
        self.tensorrt = self._safe_import('tensorrt')
        
        self.logger = get_logger(__name__)
        self._log_availability()
    
    def _safe_import(self, module_name: str, attr: Optional[str] = None):
        """Safely import a module or attribute"""
        try:
            module = __import__(module_name)
            if attr:
                return getattr(module, attr)
            return module
        except (ImportError, AttributeError) as e:
            self.logger.debug(f"Failed to import {module_name}.{attr or ''}: {e}")
            return None
    
    def _safe_import_attr(self, module_name: str, attr: str):
        """Safely import an attribute from a module"""
        try:
            module = __import__(module_name)
            return getattr(module, attr)
        except (ImportError, AttributeError):
            return None
    
    def is_available(self, dependency: str) -> bool:
        """Check if a dependency is available"""
        return getattr(self, dependency, None) is not None
    
    def require(self, dependency: str, error_msg: Optional[str] = None):
        """Require a dependency or raise an error"""
        if not self.is_available(dependency):
            msg = error_msg or f"Required dependency '{dependency}' is not available"
            raise RuntimeError(msg)
        return getattr(self, dependency)
    
    def _log_availability(self):
        """Log availability of dependencies"""
        deps = {
            'PyTorch': self.torch is not None,
            'CUDA': self.cuda is not None and (self.cuda.is_available() if hasattr(self.cuda, 'is_available') else False),
            'YOLO': self.yolo is not None,
            'ONNX': self.onnx is not None,
            'TensorRT': self.tensorrt is not None
        }
        
        available = [name for name, avail in deps.items() if avail]
        unavailable = [name for name, avail in deps.items() if not avail]
        
        if available:
            self.logger.info(f"Available dependencies: {', '.join(available)}")
        if unavailable:
            self.logger.debug(f"Unavailable dependencies: {', '.join(unavailable)}")


# Global dependency manager instance
deps = DependencyManager()

# Legacy compatibility exports
TENSORRT_AVAILABLE = deps.is_available('tensorrt')
YOLO_AVAILABLE = deps.is_available('yolo')


class DetectionPool:
    """Object pool for Detection instances to reduce allocations"""
    
    def __init__(self, pool_size: int = 100):
        self._pool_size = pool_size
        self._available = deque()
        self._in_use = weakref.WeakSet()
        self._lock = threading.Lock()
        
        # Pre-allocate pool
        for _ in range(pool_size):
            detection = Detection.__new__(Detection)
            self._available.append(detection)
    
    def get_detection(self, box: Tuple[float, float, float, float], 
                     confidence: float, class_id: int, **kwargs) -> 'Detection':
        """Get a Detection instance from the pool"""
        with self._lock:
            if self._available:
                detection = self._available.popleft()
                detection.__init__(box, confidence, class_id, **kwargs)
                self._in_use.add(detection)
                return detection
        
        # Pool exhausted, create new instance
        return Detection(box, confidence, class_id, **kwargs)
    
    def return_detection(self, detection: 'Detection'):
        """Return a Detection instance to the pool"""
        with self._lock:
            if detection in self._in_use and len(self._available) < self._pool_size:
                # Reset the detection for reuse
                detection._reset()
                self._available.append(detection)
                self._in_use.discard(detection)
    
    def get_stats(self) -> Dict[str, int]:
        """Get pool statistics"""
        with self._lock:
            return {
                'pool_size': self._pool_size,
                'available': len(self._available),
                'in_use': len(self._in_use),
                'total_allocated': self._pool_size + len(self._in_use) - len(self._available)
            }


class HardwareInterface(Protocol):
    """Protocol for hardware communication interfaces"""
    
    def connect(self) -> bool:
        """Connect to hardware"""
        ...
    
    def send_mouse_command(self, x: int, y: int) -> bool:
        """Send mouse movement command to hardware"""
        ...
    
    def is_connected(self) -> bool:
        """Check if hardware is connected"""
        ...
    
    def cleanup(self) -> None:
        """Cleanup hardware resources"""
        ...


class ArduinoMouseController:
    """Arduino-based mouse controller implementation"""
    
    def __init__(self, port: str, baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self.connection = None
        self.logger = get_logger(__name__)
        self._lock = threading.Lock()
        
    def connect(self) -> bool:
        """Connect to Arduino"""
        try:
            import serial
            with self._lock:
                if self.connection is None:
                    self.connection = serial.Serial(self.port, self.baudrate, timeout=0.1)
                    self.logger.info(f"Connected to Arduino on {self.port}")
                return True
        except Exception as e:
            self.logger.error(f"Failed to connect to Arduino: {e}")
            return False
    
    def send_mouse_command(self, x: int, y: int) -> bool:
        """Send mouse movement command to Arduino"""
        if not self.is_connected():
            return False
            
        try:
            command = f"MOVE,{x},{y}\n"
            with self._lock:
                if self.connection is not None:
                    self.connection.write(command.encode())
                    self.connection.flush()
                else:
                    return False
            return True
        except Exception as e:
            self.logger.error(f"Failed to send mouse command: {e}")
            return False
    
    def is_connected(self) -> bool:
        """Check if Arduino is connected"""
        with self._lock:
            return self.connection is not None and self.connection.is_open
    
    def cleanup(self) -> None:
        """Cleanup Arduino connection"""
        with self._lock:
            if self.connection:
                try:
                    self.connection.close()
                except Exception:
                    pass
                self.connection = None


class PrecisionMode(Enum):
    """Supported precision modes for inference"""
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    MIXED = "mixed"
    INT8 = "int8"


class InferenceBackend(Enum):
    """Available inference backends"""
    PYTORCH = "pytorch"
    TENSORRT = "tensorrt"
    ONNX = "onnx"


@dataclass
class Detection:
    """Structured detection result with validation and pooling support"""
    box: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: Optional[str] = None
    tracking_id: Optional[int] = None
    feature_vector: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Validate detection parameters"""
        self._validate()
    
    def _validate(self):
        """Validate detection parameters"""
        x1, y1, x2, y2 = self.box
        if x1 >= x2 or y1 >= y2:
            raise ValueError(f"Invalid box coordinates: {self.box}")
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Invalid confidence: {self.confidence}")
        if self.class_id < 0:
            raise ValueError(f"Invalid class ID: {self.class_id}")
    
    def _reset(self):
        """Reset detection for object pooling"""
        self.box = (0, 0, 0, 0)
        self.confidence = 0.0
        self.class_id = 0
        self.class_name = None
        self.tracking_id = None
        self.feature_vector = None
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get detection center point"""
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    @property
    def size(self) -> Tuple[float, float]:
        """Get detection size (width, height)"""
        x1, y1, x2, y2 = self.box
        return (x2 - x1, y2 - y1)
    
    @property
    def area(self) -> float:
        """Get detection area"""
        w, h = self.size
        return w * h
    
    def iou(self, other: 'Detection') -> float:
        """Calculate IoU with another detection"""
        x1a, y1a, x2a, y2a = self.box
        x1b, y1b, x2b, y2b = other.box
        
        # Intersection
        x1 = max(x1a, x1b)
        y1 = max(y1a, y1b)
        x2 = min(x2a, x2b)
        y2 = min(y2a, y2b)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        union = self.area + other.area - intersection
        
        return intersection / union if union > 0 else 0.0


@dataclass
class InferenceMetrics:
    """Detailed inference performance metrics"""
    preprocess_us: float = 0.0
    inference_us: float = 0.0
    postprocess_us: float = 0.0
    total_us: float = 0.0
    gpu_memory_mb: float = 0.0
    cpu_memory_mb: float = 0.0
    temperature_c: Optional[float] = None
    
    @property
    def total_ms(self) -> float:
        return self.total_us / 1000.0


class InputValidator:
    """Comprehensive input validation for detection pipeline"""
    
    @staticmethod
    def validate_image(image: np.ndarray) -> bool:
        """Validate input image"""
        if image is None:
            return False
        if not isinstance(image, np.ndarray):
            return False
        if len(image.shape) != 3:
            return False
        if image.shape[2] != 3:
            return False
        if image.dtype != np.uint8:
            return False
        if image.size == 0:
            return False
        return True
    
    @staticmethod
    def validate_detection_params(box: Tuple, confidence: float, class_id: int) -> bool:
        """Validate detection parameters"""
        if not isinstance(box, (tuple, list)) or len(box) != 4:
            return False
        x1, y1, x2, y2 = box
        if x1 >= x2 or y1 >= y2:
            return False
        if not 0 <= confidence <= 1:
            return False
        if class_id < 0:
            return False
        return True


class DetectorBackend(ABC):
    """Abstract base class for detection backends with enhanced error handling"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self._is_initialized = False
        self._error_count = 0
        self._max_errors = 10
    
    @abstractmethod
    def load_model(self, model_path: str, config: ModelConfig) -> None:
        """Load model with configuration"""
        pass
    
    @abstractmethod
    def infer(self, image: np.ndarray) -> Tuple[List[Detection], InferenceMetrics]:
        """Run inference on image"""
        pass
    
    @abstractmethod
    def warmup(self, input_shape: Tuple[int, int, int], iterations: int = 10) -> None:
        """Warmup model for consistent performance"""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup resources"""
        pass
    
    def _handle_error(self, error: Exception, operation: str) -> None:
        """Handle backend errors with recovery strategy"""
        self._error_count += 1
        self.logger.error(f"Backend error in {operation}: {error}")
        
        if self._error_count >= self._max_errors:
            self.logger.critical(f"Backend exceeded max errors ({self._max_errors})")
            raise RuntimeError(f"Backend failed after {self._max_errors} errors")
    
    def _reset_error_count(self):
        """Reset error count after successful operation"""
        self._error_count = 0


class YOLOBackend(DetectorBackend):
    """Enhanced YOLO backend with PyTorch CUDA support"""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.device = None
        deps.require('yolo', "YOLO backend requires ultralytics")
    
    def load_model(self, model_path: str, config: ModelConfig) -> None:
        """Load YOLO model with enhanced error handling"""
        try:
            yolo_class = deps.require('yolo', "YOLO backend requires ultralytics package")
            self.model = yolo_class(model_path)
            
            # Configure device
            if config.device == "cuda" and deps.is_available('cuda'):
                if deps.cuda and deps.cuda.is_available():
                    self.device = 'cuda'
                    self.model.to('cuda')
                    self.logger.info(f"YOLO model moved to CUDA device: {deps.cuda.get_device_name()}")
                else:
                    self.device = 'cpu'
                    self.logger.warning("CUDA requested but not available, using CPU")
            else:
                self.device = 'cpu'
                self.model.to('cpu')
            
            # Export to ONNX for future optimization
            self._try_export_onnx(model_path)
            
            self._is_initialized = True
            self.logger.info(f"YOLO model loaded: {model_path}")
            
        except Exception as e:
            self._handle_error(e, "model_loading")
            raise
    
    def _try_export_onnx(self, model_path: str):
        """Try to export model to ONNX (non-critical)"""
        try:
            if self.model is None:
                self.logger.warning("Cannot export ONNX: model is None")
                return
                
            onnx_path = Path(model_path).with_suffix('.onnx')
            if not onnx_path.exists():
                self.logger.info("Exporting YOLO model to ONNX...")
                self.model.export(format='onnx', dynamic=True, simplify=True)
        except Exception as e:
            self.logger.warning(f"Failed to export ONNX (non-critical): {e}")
    
    def infer(self, image: np.ndarray) -> Tuple[List[Detection], InferenceMetrics]:
        """Run YOLO inference with comprehensive error handling"""
        if not self._is_initialized:
            raise RuntimeError("Backend not initialized")
        
        # Validate input
        if not InputValidator.validate_image(image):
            raise ValueError(f"Invalid image: shape={image.shape if image is not None else None}")
        
        metrics = InferenceMetrics()
        
        try:
            start_total = time.perf_counter()
            
            # Check if model is loaded
            if self.model is None:
                raise RuntimeError("Model is not loaded")
            
            # Run inference
            results = self.model(image, verbose=False)
            
            metrics.total_us = (time.perf_counter() - start_total) * 1e6
            
            # Convert to detections with validation
            detections = self._postprocess_results(results)
            
            # Track GPU metrics
            if self.device == 'cuda' and deps.cuda:
                try:
                    metrics.gpu_memory_mb = deps.cuda.memory_allocated() / 1024 / 1024
                    metrics.temperature_c = deps.cuda.get_device_properties(0).temperature
                except Exception:
                    pass
            
            self._reset_error_count()
            return detections, metrics
            
        except Exception as e:
            self._handle_error(e, "inference")
            # Return fallback results
            return self._fallback_detection(image), metrics
    
    def _postprocess_results(self, results) -> List[Detection]:
        """Postprocess YOLO results with validation"""
        detections = []
        
        if not results or len(results) == 0:
            return detections
        
        result = results[0]
        if result.boxes is None:
            return detections
        
        boxes = result.boxes
        for i in range(len(boxes)):
            try:
                box = tuple(boxes.xyxy[i].cpu().numpy())
                conf = float(boxes.conf[i])
                cls_id = int(boxes.cls[i])
                
                # Validate detection parameters
                if not InputValidator.validate_detection_params(box, conf, cls_id):
                    continue
                
                detection = Detection(
                    box=box,
                    confidence=conf,
                    class_id=cls_id,
                    class_name=self.model.names.get(cls_id, str(cls_id)) if self.model and hasattr(self.model, 'names') else str(cls_id)
                )
                
                detections.append(detection)
                
            except Exception as e:
                self.logger.warning(f"Failed to process detection {i}: {e}")
                continue
        
        return detections
    
    def _fallback_detection(self, image: np.ndarray) -> List[Detection]:
        """Fallback detection method"""
        self.logger.warning("Using fallback detection (empty results)")
        return []
    
    def warmup(self, input_shape: Tuple[int, int, int], iterations: int = 10) -> None:
        """Warmup YOLO model"""
        if not self._is_initialized or self.model is None:
            return
        
        h, w, c = input_shape
        dummy_input = np.random.randint(0, 255, (h, w, c), dtype=np.uint8)
        
        for i in range(iterations):
            try:
                self.model(dummy_input, verbose=False)
            except Exception as e:
                self.logger.warning(f"Warmup iteration {i} failed: {e}")
        
        self.logger.info(f"YOLO warmup completed ({iterations} iterations)")
    
    def cleanup(self) -> None:
        """Cleanup YOLO resources"""
        try:
            self.model = None
            self.device = None
            self._is_initialized = False
            if deps.cuda and deps.cuda.is_available():
                deps.cuda.empty_cache()
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")


class YOLOv8Detector:
    """
    Enhanced production-grade YOLOv8 detector with improved architecture
    """
    
    def __init__(self, 
                 config_manager: Optional[ConfigManager] = None,
                 backend: Optional[InferenceBackend] = None,
                 hardware_interface: Optional[HardwareInterface] = None,
                 enable_pooling: bool = True):
        
        self.logger = get_logger(__name__)
        self.config_manager = config_manager or ConfigManager()
        self.model_config = self.config_manager.get_model_config()
        
        # Object pooling for high-performance scenarios
        self.detection_pool = DetectionPool() if enable_pooling else None
        
        # Hardware interface for mouse control
        self.hardware = hardware_interface
        
        # Select and create backend
        self.backend_type = backend or self._select_best_backend()
        self.backend = self._create_backend(self.backend_type)
        
        # Performance tracking
        self.inference_metrics = deque(maxlen=100)
        self._total_detections = 0
        self._failed_detections = 0
        
        # Resource management
        self._cleanup_callbacks = []
        
        # Initialize
        self._initialize()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with guaranteed cleanup"""
        self.cleanup()
        return False
    
    @contextmanager
    def managed_inference(self):
        """Context manager for managed inference sessions"""
        try:
            yield self
        finally:
            # Perform any session-specific cleanup
            if self.detection_pool:
                stats = self.detection_pool.get_stats()
                self.logger.debug(f"Pool stats after session: {stats}")
    
    def _initialize(self):
        """Initialize the detector"""
        try:
            # Load model
            self.backend.load_model(self.model_config.model_path, self.model_config)
            
            # Initialize hardware if provided
            if self.hardware:
                if hasattr(self.hardware, 'connect'):
                    self.hardware.connect()
            
            # Warmup
            self._warmup()
            
            self.logger.info(f"YOLOv8 detector initialized with {self.backend_type.value} backend")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize detector: {e}")
            raise
    
    def _select_best_backend(self) -> InferenceBackend:
        """Select optimal inference backend"""
        if self.model_config.model_path.endswith('.onnx') and deps.is_available('onnx'):
            return InferenceBackend.ONNX
        elif deps.is_available('yolo'):
            return InferenceBackend.PYTORCH
        else:
            raise RuntimeError("No suitable backend available")
    
    def _create_backend(self, backend_type: InferenceBackend) -> DetectorBackend:
        """Create backend instance"""
        if backend_type == InferenceBackend.PYTORCH:
            return YOLOBackend()
        else:
            raise ValueError(f"Unsupported backend: {backend_type}")
    
    def _warmup(self) -> None:
        """Warmup model for consistent performance"""
        input_shape = (640, 640, 3)  # Default YOLO input
        iterations = getattr(self.model_config, 'warmup_iterations', 10)
        
        self.logger.info(f"Starting warmup ({iterations} iterations)...")
        self.backend.warmup(input_shape, iterations)
    
    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Detect objects in the given image
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of Detection objects
        """
        try:
            with self.managed_inference():
                detections, metrics = self.backend.infer(image)
                
                # Update metrics
                self.inference_metrics.append(metrics)
                self._total_detections += 1
                
                return detections
                
        except Exception as e:
            self.logger.error(f"Detection failed: {e}")
            self._failed_detections += 1
            return []
    
    def select_best_target(self, detections: List[Detection], screen_center: Tuple[int, int]) -> Optional[Detection]:
        """
        Select the best target from available detections
        
        Args:
            detections: List of available detections
            screen_center: Center of the screen for distance calculation
            
        Returns:
            Best target detection or None
        """
        if not detections:
            return None
        
        # Filter by confidence threshold
        config = self.config_manager.get_model_config()
        valid_detections = [d for d in detections if d.confidence >= config.confidence_threshold]
        
        if not valid_detections:
            return None
        
        # Select closest to screen center
        def distance_to_center(detection: Detection) -> float:
            center_x, center_y = detection.center
            dx = center_x - screen_center[0]
            dy = center_y - screen_center[1]
            return (dx * dx + dy * dy) ** 0.5
        
        return min(valid_detections, key=distance_to_center)
    
    def predict(self, detection: Detection) -> Dict[str, float]:
        """
        Predict target position with movement compensation
        
        Args:
            detection: Current detection
            
        Returns:
            Dictionary with predicted coordinates
        """
        # For now, return current position
        # In a full implementation, this would include:
        # - Movement prediction based on velocity
        # - Aim point calculation (head/chest offset)
        # - Smoothing and interpolation
        
        center_x, center_y = detection.center
        aim_config = self.config_manager.get_aim_config()
        
        # Apply aim height offset
        predicted_y = center_y - (detection.size[1] * aim_config.aim_height_offset)
        
        return {
            'x': center_x,
            'y': predicted_y,
            'confidence': detection.confidence
        }

    # ...existing code...
    
    def send_mouse_command(self, x: int, y: int) -> bool:
        """Send mouse command to hardware interface"""
        if self.hardware:
            return self.hardware.send_mouse_command(x, y)
        return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        if not self.inference_metrics:
            base_metrics = {
                'backend': self.backend_type.value,
                'total_detections': self._total_detections,
                'failed_detections': self._failed_detections,
                'success_rate': 0.0
            }
        else:
            # Calculate statistics
            all_times = [m.total_us for m in self.inference_metrics]
            success_rate = (self._total_detections - self._failed_detections) / max(1, self._total_detections)
            
            base_metrics = {
                'backend': self.backend_type.value,
                'cuda_available': deps.is_available('cuda') and deps.cuda.is_available() if deps.cuda else False,
                'total_detections': self._total_detections,
                'failed_detections': self._failed_detections,
                'success_rate': success_rate,
                'inference': {
                    'avg_us': statistics.mean(all_times),
                    'min_us': min(all_times),
                    'max_us': max(all_times),
                    'std_us': statistics.stdev(all_times) if len(all_times) > 1 else 0
                }
            }
        
        # Add GPU metrics if available
        if (deps.cuda and deps.cuda.is_available() and 
            self.inference_metrics and self.inference_metrics[-1].gpu_memory_mb):
            base_metrics['gpu'] = {
                'memory_mb': self.inference_metrics[-1].gpu_memory_mb,
                'temperature_c': self.inference_metrics[-1].temperature_c,
                'device_name': deps.cuda.get_device_name() if hasattr(deps.cuda, 'get_device_name') else 'Unknown'
            }
        
        # Add pool metrics if enabled
        if self.detection_pool:
            base_metrics['pool'] = self.detection_pool.get_stats()
        
        # Add hardware status
        if self.hardware:
            base_metrics['hardware'] = {
                'connected': self.hardware.is_connected()
            }
        
        return base_metrics
    
    def cleanup(self) -> None:
        """Comprehensive cleanup with error handling"""
        errors = []
        
        # Backend cleanup
        try:
            if hasattr(self, 'backend'):
                self.backend.cleanup()
        except Exception as e:
            errors.append(f"Backend cleanup: {e}")
        
        # Hardware cleanup
        try:
            if self.hardware:
                self.hardware.cleanup()
        except Exception as e:
            errors.append(f"Hardware cleanup: {e}")
        
        # Run additional cleanup callbacks
        for callback in self._cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                errors.append(f"Callback cleanup: {e}")
        
        if errors:
            self.logger.warning(f"Cleanup errors: {'; '.join(errors)}")
        else:
            self.logger.info("YOLOv8 detector cleaned up successfully")
    
    def add_cleanup_callback(self, callback):
        """Add a cleanup callback"""
        self._cleanup_callbacks.append(callback)


# Factory functions for backward compatibility
def create_detector(config_path: str = "configs/config.ini", 
                   hardware_port: Optional[str] = None) -> YOLOv8Detector:
    """Create detector instance with optional hardware interface"""
    config_manager = ConfigManager(config_path)
    
    # Create hardware interface if port specified
    hardware = None
    if hardware_port:
        hardware = ArduinoMouseController(hardware_port)
    
    return YOLOv8Detector(config_manager, hardware_interface=hardware)


def create_managed_detector(config_path: str = "configs/config.ini") -> contextlib.AbstractContextManager[YOLOv8Detector]:
    """Create a managed detector with automatic cleanup"""
    return YOLOv8Detector(ConfigManager(config_path))


# Export enhanced dependencies info
def get_dependency_info() -> Dict[str, Any]:
    """Get comprehensive dependency information"""
    return {
        'available': {
            'torch': deps.is_available('torch'),
            'cuda': deps.is_available('cuda') and (deps.cuda.is_available() if deps.cuda else False),
            'yolo': deps.is_available('yolo'),
            'onnx': deps.is_available('onnx'),
            'tensorrt': deps.is_available('tensorrt')
        },
        'versions': {
            'torch': getattr(deps.torch, '__version__', 'Unknown') if deps.torch else None,
            'cuda': getattr(deps.cuda, 'version', 'Unknown') if deps.cuda else None,
        },
        'gpu_info': {
            'device_name': deps.cuda.get_device_name() if deps.cuda and hasattr(deps.cuda, 'get_device_name') else None,
            'memory_total': deps.cuda.get_device_properties(0).total_memory // 1024**2 if deps.cuda and hasattr(deps.cuda, 'get_device_properties') else None
        } if deps.cuda and deps.cuda.is_available() else None
    }
