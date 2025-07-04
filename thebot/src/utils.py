"""
Consolidated Production Utilities Module
Purpose: High-performance utilities with hardware acceleration support
"""
import time
import psutil
import GPUtil
import numpy as np
from typing import Tuple, Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from collections import deque
import threading
from functools import wraps
import traceback
import sys

try:
    import torch
    import torch.cuda as cuda
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    cuda = None

# CuPy removed - not used in current implementation

try:
    import win32api
    import win32con
    WIN32_AVAILABLE = True
except ImportError:
    WIN32_AVAILABLE = False
    win32api = None

from .logger_util import get_logger


# Re-export logger for backward compatibility
from .logger_util import Logger, setup_logging, get_logger as logger


@dataclass
class TimingResult:
    """Detailed timing result"""
    operation: str
    duration_ms: float
    cpu_time_ms: float
    gpu_time_ms: Optional[float] = None
    memory_used_mb: Optional[float] = None
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class PerformanceTimer:
    """High-precision performance timer with GPU support"""
    
    def __init__(self, use_cuda: bool = True):
        self.use_cuda = use_cuda and TORCH_AVAILABLE and cuda and cuda.is_available()
        self.logger = get_logger(__name__)
        
        # CUDA events for GPU timing
        if self.use_cuda and cuda:
            self.start_event = cuda.Event(enable_timing=True)
            self.end_event = cuda.Event(enable_timing=True)
        
        # CPU timing
        self.cpu_start: Optional[float] = None
        self.wall_start: Optional[float] = None
        
        # Memory tracking
        self.start_memory: Optional[float] = None
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        # Timer automatically stopped when getting result
        pass
    
    def start(self) -> None:
        """Start timing"""
        # CPU timing
        self.cpu_start = time.process_time()
        self.wall_start = time.perf_counter()
        
        # GPU timing
        if self.use_cuda and cuda:
            self.start_event.record(cuda.current_stream())
            self.start_memory = cuda.memory_allocated() / 1024 / 1024  # MB
    
    def stop(self) -> TimingResult:
        """Stop timing and return result"""
        if self.cpu_start is None or self.wall_start is None:
            raise RuntimeError("Timer not started")
        
        # CPU timing
        cpu_time = (time.process_time() - self.cpu_start) * 1000
        wall_time = (time.perf_counter() - self.wall_start) * 1000
        
        # GPU timing
        gpu_time = None
        memory_used = None
        
        if self.use_cuda and cuda:
            self.end_event.record(cuda.current_stream())
            cuda.synchronize()
            gpu_time = self.start_event.elapsed_time(self.end_event)
            
            current_memory = cuda.memory_allocated() / 1024 / 1024
            memory_used = current_memory - (self.start_memory or 0)
        
        return TimingResult(
            operation="unnamed",
            duration_ms=wall_time,
            cpu_time_ms=cpu_time,
            gpu_time_ms=gpu_time,
            memory_used_mb=memory_used
        )


def timed_operation(name: str = "", log_result: bool = True):
    """Decorator for timing operations"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            timer = PerformanceTimer()
            
            try:
                with timer:
                    result = func(*args, **kwargs)
                
                timing = timer.stop()
                timing.operation = name or func.__name__
                
                if log_result:
                    logger = get_logger(func.__module__)
                    logger.debug(f"[TIMING] {timing.operation}: "
                               f"{timing.duration_ms:.2f}ms "
                               f"(CPU: {timing.cpu_time_ms:.2f}ms"
                               f"{f', GPU: {timing.gpu_time_ms:.2f}ms' if timing.gpu_time_ms else ''})")
                
                # Attach timing to result if possible
                if hasattr(result, '_timing'):
                    result._timing = timing
                
                return result
                
            except Exception as e:
                logger = get_logger(func.__module__)
                logger.error(f"Error in {name or func.__name__}: {e}")
                raise
        
        return wrapper
    return decorator


class SystemMonitor:
    """Comprehensive system resource monitoring"""
    
    def __init__(self, history_size: int = 60):
        self.history_size = history_size
        self.logger = get_logger(__name__)
        
        # Metrics history
        self.cpu_history = deque(maxlen=history_size)
        self.memory_history = deque(maxlen=history_size)
        self.gpu_history = deque(maxlen=history_size)
        self.temperature_history = deque(maxlen=history_size)
        
        # Get process handle
        self.process = psutil.Process()
        
        # GPU monitoring
        self.gpu_available = False
        self.gpu_handles = []
        self._init_gpu_monitoring()
        
        # Background monitoring thread
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
    
    def _init_gpu_monitoring(self) -> None:
        """Initialize GPU monitoring"""
        try:
            self.gpu_handles = GPUtil.getGPUs()
            self.gpu_available = len(self.gpu_handles) > 0
            
            if self.gpu_available:
                self.logger.info(f"GPU monitoring enabled: {len(self.gpu_handles)} GPU(s) found")
        except Exception as e:
            self.logger.warning(f"GPU monitoring not available: {e}")
    
    def start_monitoring(self, interval: float = 1.0) -> None:
        """Start background monitoring"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop background monitoring"""
        self._monitoring = False
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
    
    def _monitor_loop(self, interval: float) -> None:
        """Background monitoring loop"""
        while self._monitoring:
            try:
                metrics = self.get_current_metrics()
                
                # Update history
                self.cpu_history.append(metrics['cpu']['total'])
                self.memory_history.append(metrics['memory']['percent'])
                
                if self.gpu_available:
                    self.gpu_history.append(metrics['gpu'][0]['usage'])
                    self.temperature_history.append(metrics['gpu'][0]['temperature'])
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
            
            time.sleep(interval)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        metrics = {}
        
        # CPU metrics
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
            metrics['cpu'] = {
                'total': sum(cpu_percent) / len(cpu_percent),
                'per_core': cpu_percent,
                'count': psutil.cpu_count(),
                'frequency': psutil.cpu_freq().current if psutil.cpu_freq() else 0
            }
        except Exception as e:
            self.logger.error(f"CPU metrics error: {e}")
            metrics['cpu'] = {'total': 0, 'per_core': [], 'count': 0, 'frequency': 0}
        
        # Memory metrics
        try:
            mem = psutil.virtual_memory()
            metrics['memory'] = {
                'total_mb': mem.total / 1024 / 1024,
                'used_mb': mem.used / 1024 / 1024,
                'available_mb': mem.available / 1024 / 1024,
                'percent': mem.percent
            }
        except Exception as e:
            self.logger.error(f"Memory metrics error: {e}")
            metrics['memory'] = {'total_mb': 0, 'used_mb': 0, 'available_mb': 0, 'percent': 0}
        
        # GPU metrics
        if self.gpu_available:
            try:
                gpu_metrics = []
                
                for gpu in GPUtil.getGPUs():
                    gpu_metrics.append({
                        'id': gpu.id,
                        'name': gpu.name,
                        'usage': gpu.load * 100,
                        'memory_used_mb': gpu.memoryUsed,
                        'memory_total_mb': gpu.memoryTotal,
                        'temperature': gpu.temperature
                    })
                
                metrics['gpu'] = gpu_metrics
                
            except Exception as e:
                self.logger.error(f"GPU metrics error: {e}")
                metrics['gpu'] = []
        else:
            metrics['gpu'] = []
        
        # Process-specific metrics
        try:
            with self.process.oneshot():
                metrics['process'] = {
                    'cpu_percent': self.process.cpu_percent(),
                    'memory_mb': self.process.memory_info().rss / 1024 / 1024,
                    'threads': self.process.num_threads()
                }
        except Exception as e:
            self.logger.error(f"Process metrics error: {e}")
            metrics['process'] = {'cpu_percent': 0, 'memory_mb': 0, 'threads': 0}
        
        return metrics
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics"""
        summary = {}
        
        if self.cpu_history:
            summary['cpu_avg'] = sum(self.cpu_history) / len(self.cpu_history)
            summary['cpu_max'] = max(self.cpu_history)
        
        if self.memory_history:
            summary['memory_avg'] = sum(self.memory_history) / len(self.memory_history)
            summary['memory_max'] = max(self.memory_history)
        
        if self.gpu_history:
            summary['gpu_avg'] = sum(self.gpu_history) / len(self.gpu_history)
            summary['gpu_max'] = max(self.gpu_history)
        
        if self.temperature_history:
            summary['temp_avg'] = sum(self.temperature_history) / len(self.temperature_history)
            summary['temp_max'] = max(self.temperature_history)
        
        return summary


class CudaAccelerator:
    """CUDA acceleration utilities"""
    
    def __init__(self):
        self.available = TORCH_AVAILABLE and cuda and cuda.is_available()
        self.device = None
        self.logger = get_logger(__name__)
        
        if self.available and torch and cuda:
            self.device = torch.device('cuda')
            self.properties = cuda.get_device_properties(0)
            self.logger.info(f"CUDA device: {self.properties.name} "
                           f"({self.properties.total_memory / 1024**3:.1f} GB)")
    
    @timed_operation("cuda_nms")
    def nms_cuda(self, 
                  boxes: np.ndarray, 
                  scores: np.ndarray, 
                  threshold: float = 0.5) -> np.ndarray:
        """CUDA-accelerated Non-Maximum Suppression"""
        if not self.available:
            return self.nms_cpu(boxes, scores, threshold)
        
        try:
            # Convert to tensors
            if torch:
                boxes_tensor = torch.from_numpy(boxes).float().cuda()
                scores_tensor = torch.from_numpy(scores).float().cuda()
                
                # Perform NMS
                from torchvision.ops import nms
                keep_indices = nms(boxes_tensor, scores_tensor, threshold)
                
                return keep_indices.cpu().numpy()
            else:
                return self.nms_cpu(boxes, scores, threshold)
            
        except Exception as e:
            self.logger.error(f"CUDA NMS failed: {e}, falling back to CPU")
            return self.nms_cpu(boxes, scores, threshold)
    
    def nms_cpu(self, 
                boxes: np.ndarray, 
                scores: np.ndarray, 
                threshold: float = 0.5) -> np.ndarray:
        """CPU Non-Maximum Suppression"""
        # Simple CPU NMS implementation
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(ovr <= threshold)[0]
            order = order[inds + 1]
        
        return np.array(keep)


def get_current_mouse_position() -> Tuple[int, int]:
    """Get current mouse position"""
    if WIN32_AVAILABLE and win32api:
        return win32api.GetCursorPos()
    else:
        # Fallback - would need other implementation
        return (0, 0)


def screen_to_target_coords(bbox: Tuple[float, float, float, float], 
                          aim_height_offset: float) -> Tuple[int, int]:
    """Calculate target coordinates from bounding box"""
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    
    # Apply aim offset (percentage of box height)
    box_height = y2 - y1
    offset_y = int(box_height * aim_height_offset)
    target_y = center_y - offset_y
    
    return center_x, target_y


def calculate_angle_to_target(current_pos: Tuple[int, int], 
                            target_pos: Tuple[int, int],
                            fov: float = 90.0) -> Tuple[float, float]:
    """Calculate angles needed to reach target"""
    dx = target_pos[0] - current_pos[0]
    dy = target_pos[1] - current_pos[1]
    
    # Convert to angles (simplified)
    # In a real implementation, this would account for game-specific FOV
    angle_x = np.arctan2(dx, 1000) * 180 / np.pi
    angle_y = np.arctan2(dy, 1000) * 180 / np.pi
    
    return angle_x, angle_y


def smooth_movement(current: Tuple[float, float], 
                   target: Tuple[float, float], 
                   smoothing: float = 0.3) -> Tuple[float, float]:
    """Apply smoothing to movement"""
    # Exponential smoothing
    smooth_x = current[0] + (target[0] - current[0]) * smoothing
    smooth_y = current[1] + (target[1] - current[1]) * smoothing
    
    return smooth_x, smooth_y


def clamp_movement(dx: float, dy: float, max_speed: float = 500) -> Tuple[int, int]:
    """Clamp movement to maximum speed"""
    magnitude = np.sqrt(dx*dx + dy*dy)
    
    if magnitude > max_speed:
        scale = max_speed / magnitude
        dx *= scale
        dy *= scale
    
    return int(dx), int(dy)


# Global instances for convenience
_system_monitor: Optional[SystemMonitor] = None
_cuda_accelerator: Optional[CudaAccelerator] = None


def get_system_monitor() -> SystemMonitor:
    """Get or create global system monitor"""
    global _system_monitor
    if _system_monitor is None:
        _system_monitor = SystemMonitor()
        _system_monitor.start_monitoring()
    return _system_monitor


def get_cuda_accelerator() -> CudaAccelerator:
    """Get or create global CUDA accelerator"""
    global _cuda_accelerator
    if _cuda_accelerator is None:
        _cuda_accelerator = CudaAccelerator()
    return _cuda_accelerator


# Cleanup on exit
import atexit

def _cleanup():
    """Cleanup global resources"""
    global _system_monitor
    if _system_monitor:
        _system_monitor.stop_monitoring()

atexit.register(_cleanup)