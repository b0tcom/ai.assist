"""
Production-Grade DPI-Aware Screen Capture System with Hardware Acceleration
Purpose: High-performance screen capture with automatic DPI scaling correction
"""
import numpy as np
import cv2
import ctypes
from ctypes import wintypes
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import threading
import time
from abc import ABC, abstractmethod
from enum import Enum

try:
    import bettercam
    BETTERCAM_AVAILABLE = True
except ImportError:
    BETTERCAM_AVAILABLE = False

try:
    import win32api
    import win32con
    WIN32_AVAILABLE = True
except ImportError:
    WIN32_AVAILABLE = False
    win32api = None
    win32con = None

import mss
from .logger_util import get_logger
from .config_manager import ConfigManager, ScreenRegion


class CaptureBackend(Enum):
    """Available capture backends"""
    BETTERCAM = "bettercam"
    MSS = "mss"
    DXCAM = "dxcam"
    WIN32 = "win32"


@dataclass
class DisplayInfo:
    """Display information with DPI awareness"""
    desktop_width: int
    desktop_height: int
    actual_width: int
    actual_height: int
    dpi_x: float
    dpi_y: float
    scale_x: float
    scale_y: float
    refresh_rate: int
    
    @property
    def has_scaling(self) -> bool:
        """Check if display has DPI scaling"""
        return self.scale_x != 1.0 or self.scale_y != 1.0


class CaptureBackendBase(ABC):
    """Abstract base class for capture backends"""
    
    @abstractmethod
    def initialize(self, region: ScreenRegion) -> None:
        """Initialize the capture backend"""
        pass
    
    @abstractmethod
    def capture(self) -> Optional[np.ndarray]:
        """Capture a frame"""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup resources"""
        pass


class BettercamBackend(CaptureBackendBase):
    """Bettercam capture backend with hardware acceleration"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.camera = None
        self.region = None
        
    def initialize(self, region: ScreenRegion) -> None:
        """Initialize bettercam with region"""
        try:
            # Convert region to bettercam format
            bc_region = (region.left, region.top, 
                        region.left + region.width, 
                        region.top + region.height)
            
            self.camera = bettercam.create(
                output_idx=0,  # Primary monitor
                output_color="BGR",
                region=bc_region,
                max_buffer_len=2
            )
            
            if self.camera is None:
                raise RuntimeError("Failed to create bettercam instance")
                
            self.region = region
            self.logger.info(f"Bettercam initialized with region: {region.to_dict()}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize bettercam: {e}")
            raise
    
    def capture(self) -> Optional[np.ndarray]:
        """Capture frame using bettercam"""
        if self.camera is None:
            return None
            
        try:
            frame = self.camera.grab()
            
            if frame is None:
                return None
                
            # Ensure frame is contiguous BGR
            if frame.shape[2] == 4:  # BGRA
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
            return np.ascontiguousarray(frame)
            
        except Exception as e:
            self.logger.error(f"Capture error: {e}")
            return None
    
    def cleanup(self) -> None:
        """Cleanup bettercam resources"""
        if self.camera:
            try:
                self.camera.stop()
            except Exception:
                pass
            self.camera = None


class MSSBackend(CaptureBackendBase):
    """MSS capture backend (fallback)"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.sct = None
        self.monitor = None
        
    def initialize(self, region: ScreenRegion) -> None:
        """Initialize MSS with region"""
        self.sct = mss.mss()
        self.monitor = {
            "left": region.left,
            "top": region.top,
            "width": region.width,
            "height": region.height
        }
        self.logger.info(f"MSS initialized with region: {region.to_dict()}")
    
    def capture(self) -> Optional[np.ndarray]:
        """Capture frame using MSS"""
        if self.sct is None or self.monitor is None:
            return None
            
        try:
            img = self.sct.grab(self.monitor)
            frame = np.array(img)
            
            # Convert BGRA to BGR
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
            return np.ascontiguousarray(frame)
            
        except Exception as e:
            self.logger.error(f"MSS capture error: {e}")
            return None
    
    def cleanup(self) -> None:
        """Cleanup MSS resources"""
        if self.sct:
            self.sct.close()
            self.sct = None


class ScreenCapture:
    """
    Production-grade screen capture with DPI awareness and backend selection
    """
    
    def __init__(self, 
                 backend: Optional[CaptureBackend] = None,
                 config_manager: Optional[ConfigManager] = None,
                 enable_metrics: bool = True):
        
        self.logger = get_logger(__name__)
        self.config_manager = config_manager or ConfigManager()
        self.enable_metrics = enable_metrics
        
        # Display information
        self.display_info = self._get_display_info()
        self._log_display_info()
        
        # Select and initialize backend
        self.backend_type = backend or self._select_best_backend()
        self.backend = self._create_backend(self.backend_type)
        
        # Get and validate region
        self.region = self._get_validated_region()
        
        # Performance metrics
        self._metrics_lock = threading.Lock()
        self._capture_times = []
        self._capture_count = 0
        self._last_fps_calc = time.time()
        self._current_fps = 0.0
        
        # Initialize backend
        self.backend.initialize(self.region)
        
        # Warm up
        self._warmup()
    
    def _get_display_info(self) -> DisplayInfo:
        """Get comprehensive display information with DPI awareness"""
        if WIN32_AVAILABLE and win32api is not None and win32con is not None:
            try:
                # Make process DPI aware
                user32 = ctypes.windll.user32
                shcore = ctypes.windll.shcore
                
                # Set DPI awareness
                try:
                    shcore.SetProcessDpiAwareness(2)  # Per-monitor DPI aware
                except Exception:
                    user32.SetProcessDPIAware()
                
                # Get desktop resolution (scaled)
                desktop_width = win32api.GetSystemMetrics(0)
                desktop_height = win32api.GetSystemMetrics(1)
                
                # Get actual display resolution
                dev_mode = win32api.EnumDisplaySettings(None, win32con.ENUM_CURRENT_SETTINGS)
                actual_width = dev_mode.PelsWidth
                actual_height = dev_mode.PelsHeight
                refresh_rate = dev_mode.DisplayFrequency
                
                # Get DPI information
                monitor = user32.MonitorFromPoint(wintypes.POINT(0, 0), 1)
                dpi_x = ctypes.c_uint()
                dpi_y = ctypes.c_uint()
                
                try:
                    shcore.GetDpiForMonitor(monitor, 0, ctypes.byref(dpi_x), ctypes.byref(dpi_y))
                    dpi_x_val = dpi_x.value
                    dpi_y_val = dpi_y.value
                except Exception:
                    # Fallback to system DPI
                    hdc = user32.GetDC(0)
                    dpi_x_val = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)  # LOGPIXELSX
                    dpi_y_val = ctypes.windll.gdi32.GetDeviceCaps(hdc, 90)  # LOGPIXELSY
                    user32.ReleaseDC(0, hdc)
                
                # Calculate scaling factors
                scale_x = dpi_x_val / 96.0
                scale_y = dpi_y_val / 96.0
                
                return DisplayInfo(
                    desktop_width=desktop_width,
                    desktop_height=desktop_height,
                    actual_width=actual_width,
                    actual_height=actual_height,
                    dpi_x=dpi_x_val,
                    dpi_y=dpi_y_val,
                    scale_x=scale_x,
                    scale_y=scale_y,
                    refresh_rate=refresh_rate
                )
                
            except Exception as e:
                self.logger.warning(f"Failed to get Windows display info: {e}")
        
        # Fallback to MSS
        with mss.mss() as sct:
            mon = sct.monitors[1]
            width = mon['width']
            height = mon['height']
            
            return DisplayInfo(
                desktop_width=width,
                desktop_height=height,
                actual_width=width,
                actual_height=height,
                dpi_x=96,
                dpi_y=96,
                scale_x=1.0,
                scale_y=1.0,
                refresh_rate=60
            )
    
    def _log_display_info(self) -> None:
        """Log display information"""
        info = self.display_info
        self.logger.info("="*60)
        self.logger.info("Display Information:")
        self.logger.info(f"  Desktop Resolution: {info.desktop_width}x{info.desktop_height}")
        self.logger.info(f"  Actual Resolution: {info.actual_width}x{info.actual_height}")
        self.logger.info(f"  DPI: {info.dpi_x}x{info.dpi_y}")
        self.logger.info(f"  Scaling: {info.scale_x:.2f}x{info.scale_y:.2f} ({int(info.scale_x*100)}%)")
        self.logger.info(f"  Refresh Rate: {info.refresh_rate}Hz")
        self.logger.info(f"  Has Scaling: {info.has_scaling}")
        self.logger.info("="*60)
    
    def _select_best_backend(self) -> CaptureBackend:
        """Select the best available capture backend"""
        if BETTERCAM_AVAILABLE:
            self.logger.info("Selected backend: Bettercam (hardware accelerated)")
            return CaptureBackend.BETTERCAM
        else:
            self.logger.info("Selected backend: MSS (fallback)")
            return CaptureBackend.MSS
    
    def _create_backend(self, backend_type: CaptureBackend) -> CaptureBackendBase:
        """Create capture backend instance"""
        if backend_type == CaptureBackend.BETTERCAM:
            return BettercamBackend()
        elif backend_type == CaptureBackend.MSS:
            return MSSBackend()
        else:
            raise ValueError(f"Unsupported backend: {backend_type}")
    
    def _get_validated_region(self) -> ScreenRegion:
        """Get and validate capture region with DPI scaling correction"""
        # Get region from config
        region = self.config_manager.get_screen_region()
        
        # Check if using desktop coordinates
        uses_desktop_coords = self.config_manager.get('Application', 'uses_desktop_coordinates', True)
        
        if uses_desktop_coords and self.display_info.has_scaling:
            # Convert desktop coordinates to actual coordinates
            self.logger.info("Converting desktop coordinates to actual display coordinates")
            
            scale_x = self.display_info.scale_x
            scale_y = self.display_info.scale_y
            
            # Scale the region
            actual_region = ScreenRegion(
                left=int(region.left / scale_x),
                top=int(region.top / scale_y),
                width=int(region.width / scale_x),
                height=int(region.height / scale_y)
            )
            
            self.logger.info(f"Desktop region: {region.to_dict()}")
            self.logger.info(f"Actual region: {actual_region.to_dict()}")
            
            region = actual_region
        
        # Validate against actual display bounds
        if not region.validate_against_screen(self.display_info.actual_width, 
                                            self.display_info.actual_height):
            self.logger.warning(f"Region {region.to_dict()} exceeds display bounds")
            
            # Clamp to display bounds
            region = ScreenRegion(
                left=max(0, min(region.left, self.display_info.actual_width - 100)),
                top=max(0, min(region.top, self.display_info.actual_height - 100)),
                width=min(region.width, self.display_info.actual_width - region.left),
                height=min(region.height, self.display_info.actual_height - region.top)
            )
            
            self.logger.info(f"Clamped region: {region.to_dict()}")
        
        return region
    
    def _warmup(self, iterations: int = 5) -> None:
        """Warm up capture pipeline"""
        self.logger.info(f"Warming up capture pipeline ({iterations} iterations)...")
        
        for i in range(iterations):
            frame = self.backend.capture()
            if frame is not None:
                self.logger.debug(f"Warmup {i+1}: {frame.shape}")
            else:
                self.logger.warning(f"Warmup {i+1}: Failed to capture")
    
    def capture(self) -> Optional[np.ndarray]:
        """Capture a frame with performance tracking"""
        start_time = time.perf_counter()
        
        try:
            frame = self.backend.capture()
            
            if frame is None:
                return None
            
            # Track metrics
            if self.enable_metrics:
                capture_time = (time.perf_counter() - start_time) * 1000
                self._update_metrics(capture_time)
            
            return frame
            
        except Exception as e:
            self.logger.error(f"Capture error: {e}", exc_info=True)
            return None
    
    def _update_metrics(self, capture_time_ms: float) -> None:
        """Update performance metrics"""
        with self._metrics_lock:
            self._capture_times.append(capture_time_ms)
            self._capture_count += 1
            
            # Keep only last 100 samples
            if len(self._capture_times) > 100:
                self._capture_times = self._capture_times[-100:]
            
            # Calculate FPS every second
            now = time.time()
            if now - self._last_fps_calc >= 1.0:
                elapsed = now - self._last_fps_calc
                self._current_fps = self._capture_count / elapsed
                self._capture_count = 0
                self._last_fps_calc = now
    
    def get_metrics(self) -> Dict[str, float]:
        """Get capture performance metrics"""
        with self._metrics_lock:
            if not self._capture_times:
                return {
                    'fps': 0.0,
                    'avg_ms': 0.0,
                    'min_ms': 0.0,
                    'max_ms': 0.0
                }
            
            return {
                'fps': self._current_fps,
                'avg_ms': sum(self._capture_times) / len(self._capture_times),
                'min_ms': min(self._capture_times),
                'max_ms': max(self._capture_times)
            }
    
    def get_region(self) -> ScreenRegion:
        """Get current capture region"""
        return self.region
    
    def set_region(self, region: ScreenRegion) -> None:
        """Update capture region"""
        # Validate new region
        if not region.validate_against_screen(self.display_info.actual_width,
                                            self.display_info.actual_height):
            raise ValueError(f"Invalid region: {region.to_dict()}")
        
        self.region = region
        self.backend.cleanup()
        self.backend.initialize(region)
        self.logger.info(f"Capture region updated: {region.to_dict()}")
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        if self.backend:
            self.backend.cleanup()
        self.logger.info("Screen capture cleaned up")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()
        return False


# Factory function for backward compatibility
def get_frame(region: Optional[Dict[str, int]] = None, 
              config_path: str = "configs/config.ini") -> Optional[np.ndarray]:
    """
    Capture a single frame (backward compatibility)
    
    Args:
        region: Optional region dict with keys: left, top, width, height
        config_path: Path to configuration file
        
    Returns:
        Captured frame as numpy array or None
    """
    config_manager = ConfigManager(config_path)
    
    # Override region if provided
    if region:
        screen_region = ScreenRegion(**region)
        config_manager.set('screen_region', '', screen_region.to_dict())
    
    capture = ScreenCapture(config_manager=config_manager)
    
    try:
        return capture.capture()
    finally:
        capture.cleanup()


# Resolution detection functions for backward compatibility
def get_screen_size() -> Tuple[int, int]:
    """Get desktop resolution (scaled)"""
    capture = ScreenCapture()
    info = capture.display_info
    capture.cleanup()
    return info.desktop_width, info.desktop_height


def get_actual_display_resolution() -> Tuple[int, int]:
    """Get actual display resolution (unscaled)"""
    capture = ScreenCapture()
    info = capture.display_info
    capture.cleanup()
    return info.actual_width, info.actual_height


def get_dpi_scale() -> Tuple[float, float]:
    """Get DPI scaling factors"""
    capture = ScreenCapture()
    info = capture.display_info
    capture.cleanup()
    return info.scale_x, info.scale_y