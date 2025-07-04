"""
Production-Grade AI Aim Assist Application Orchestrator
Purpose: Thread-safe application lifecycle management with performance monitoring
"""
import sys
import os
import time
import threading
import queue
import signal
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import multiprocessing as mp
from pathlib import Path
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from collections import deque

import numpy as np
import cv2

# Optional ML imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Production imports - handle both direct execution and module import
try:
    # When run as module
    from .logger_util import get_logger, setup_logging
    from .config_manager import ConfigManager, ScreenRegion
    from .capture import ScreenCapture, DisplayInfo
    from .detect import YOLOv8Detector, Detection, InferenceBackend
    from .input_handler import InputController, SafetyLevel
    from .pygame_overlay import OverlayMode, OverlaySystem
    from .toggle import HotkeyManager
except ImportError:
    # When run directly
    from logger_util import get_logger, setup_logging
    from config_manager import ConfigManager, ScreenRegion
    from capture import ScreenCapture, DisplayInfo
    from detect import YOLOv8Detector, Detection, InferenceBackend
    from input_handler import InputController, SafetyLevel
    from pygame_overlay import OverlayMode, OverlaySystem
    from toggle import HotkeyManager

# Optional GUI - handle both direct execution and module import
try:
    import tkinter as tk
    try:
        # When run as module
        from .gui import CVTargetingGUI as ProductionGUI
    except ImportError:
        # When run directly
        from gui import CVTargetingGUI as ProductionGUI
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False


class ApplicationState(Enum):
    """Application lifecycle states"""
    INITIALIZING = auto()
    READY = auto()
    RUNNING = auto()
    PAUSED = auto()
    ERROR = auto()
    SHUTTING_DOWN = auto()


@dataclass
class SystemMetrics:
    """Comprehensive system performance metrics"""
    capture_fps: float = 0.0
    detection_fps: float = 0.0
    input_fps: float = 0.0
    total_latency_ms: float = 0.0
    cpu_usage: float = 0.0
    gpu_usage: float = 0.0
    memory_usage_mb: float = 0.0
    dropped_frames: int = 0
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/display"""
        return {
            'capture_fps': round(self.capture_fps, 1),
            'detection_fps': round(self.detection_fps, 1),
            'input_fps': round(self.input_fps, 1),
            'total_latency_ms': round(self.total_latency_ms, 2),
            'cpu_usage': round(self.cpu_usage, 1),
            'gpu_usage': round(self.gpu_usage, 1),
            'memory_usage_mb': round(self.memory_usage_mb, 1),
            'dropped_frames': self.dropped_frames,
            'error_count': len(self.errors)
        }


class PerformanceMonitor:
    """Real-time performance monitoring with anomaly detection"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = SystemMetrics()
        self.history: Dict[str, deque] = {
            'capture': deque(maxlen=window_size),
            'detection': deque(maxlen=window_size),
            'input': deque(maxlen=window_size),
            'latency': deque(maxlen=window_size)
        }
        self.anomaly_callbacks: List[Callable[[str, float], None]] = []
        self._lock = threading.Lock()
    
    def update(self, metric_type: str, value: float) -> None:
        """Update metric with anomaly detection"""
        with self._lock:
            if metric_type in self.history:
                self.history[metric_type].append(value)
                
                # Check for anomalies (adjusted thresholds for high-performance systems)
                if len(self.history[metric_type]) >= 10:
                    avg = sum(self.history[metric_type]) / len(self.history[metric_type])
                    # More lenient thresholds for very fast systems
                    if metric_type == 'capture':
                        # For capture times, only alert if >5ms or very extreme variations
                        if value > 5.0 or (avg > 0 and (value > avg * 5 or value < avg * 0.2)):
                            for callback in self.anomaly_callbacks:
                                callback(metric_type, value)
                    else:
                        # Standard thresholds for other metrics
                        if value > avg * 3 or value < avg * 0.3:
                            for callback in self.anomaly_callbacks:
                                callback(metric_type, value)
    
    def get_metrics(self) -> SystemMetrics:
        """Get current metrics"""
        with self._lock:
            if self.history['capture']:
                self.metrics.capture_fps = 1000.0 / (sum(self.history['capture']) / len(self.history['capture']))
            if self.history['detection']:
                self.metrics.detection_fps = 1000.0 / (sum(self.history['detection']) / len(self.history['detection']))
            if self.history['input']:
                self.metrics.input_fps = 1000.0 / (sum(self.history['input']) / len(self.history['input']))
            if self.history['latency']:
                self.metrics.total_latency_ms = sum(self.history['latency']) / len(self.history['latency'])
            
            # Get system resources
            try:
                import psutil
                process = psutil.Process()
                self.metrics.cpu_usage = process.cpu_percent()
                self.metrics.memory_usage_mb = process.memory_info().rss / 1024 / 1024
            except ImportError:
                pass
            
            return self.metrics
    
    def register_anomaly_callback(self, callback: Callable[[str, float], None]) -> None:
        """Register callback for anomaly detection"""
        self.anomaly_callbacks.append(callback)


class FrameProcessor:
    """High-performance frame processing pipeline"""
    
    def __init__(self, 
                 detector: YOLOv8Detector,
                 input_controller: InputController,
                 config_manager: ConfigManager,
                 simulation_mode: bool = False):
        
        self.detector = detector
        self.input_controller = input_controller
        self.config_manager = config_manager
        self.simulation_mode = simulation_mode
        self.logger = get_logger(__name__)
        
        # Processing queues
        self.frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=2)
        self.detection_queue: queue.Queue[List[Detection]] = queue.Queue(maxsize=10)
        
        # Threading
        self.detection_thread: Optional[threading.Thread] = None
        self.targeting_thread: Optional[threading.Thread] = None
        self.running = False
        
        # Performance
        self.frame_count = 0
        self.last_fps_update = time.time()
        
    def start(self) -> None:
        """Start processing threads"""
        self.running = True
        
        self.detection_thread = threading.Thread(
            target=self._detection_worker,
            name="DetectionWorker",
            daemon=True
        )
        self.targeting_thread = threading.Thread(
            target=self._targeting_worker,
            name="TargetingWorker",
            daemon=True
        )
        
        self.detection_thread.start()
        self.targeting_thread.start()
        
        self.logger.info("Frame processor started")
    
    def stop(self) -> None:
        """Stop processing threads"""
        self.running = False
        
        # Clear queues
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        # Wait for threads
        if self.detection_thread:
            self.detection_thread.join(timeout=2.0)
        if self.targeting_thread:
            self.targeting_thread.join(timeout=2.0)
        
        self.logger.info("Frame processor stopped")
    
    def process_frame(self, frame: np.ndarray) -> bool:
        """Add frame to processing pipeline"""
        try:
            self.frame_queue.put_nowait(frame)
            self.frame_count += 1
            return True
        except queue.Full:
            return False
    
    def _detection_worker(self) -> None:
        """Detection processing thread"""
        while self.running:
            try:
                # Get frame
                frame = self.frame_queue.get(timeout=0.01)
                
                # Run detection
                detections = self.detector.detect(frame)
                
                # Queue detections
                if detections:
                    try:
                        self.detection_queue.put_nowait(detections)
                    except queue.Full:
                        # Drop oldest
                        try:
                            self.detection_queue.get_nowait()
                            self.detection_queue.put_nowait(detections)
                        except queue.Empty:
                            pass
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Detection error: {e}")
    
    def _targeting_worker(self) -> None:
        """Targeting logic thread"""
        screen_region = self.config_manager.get_screen_region()
        screen_center = (screen_region.width // 2, screen_region.height // 2)
        
        while self.running:
            try:
                # Get detections
                detections = self.detection_queue.get(timeout=0.01)
                
                # Select best target
                best_target = self.detector.select_best_target(
                    detections, screen_center
                )
                
                if best_target:
                    # Predict position
                    predicted_pos = self.detector.predict(best_target)
                    
                    # Move to target (or simulate if Arduino not available)
                    if self.simulation_mode:
                        # Simulation mode - just log the movement
                        self.logger.debug(f"SIMULATION: Would move to {predicted_pos}")
                    else:
                        self.input_controller.move_to_target(
                            {k: int(v) for k, v in predicted_pos.items()},
                            {'center': best_target.center, 'tracking_id': best_target.tracking_id}
                        )
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Targeting error: {e}")


class CVTargetingSystem:
    """
    Production AI Aim Assist System with comprehensive lifecycle management
    """
    
    def __init__(self, config_path: str = "configs/config.ini"):
        self.logger = get_logger(__name__)
        self.config_path = config_path
        self.state = ApplicationState.INITIALIZING
        
        # Core components
        self.config_manager: Optional[ConfigManager] = None
        self.capture: Optional[ScreenCapture] = None
        self.detector: Optional[YOLOv8Detector] = None
        self.input_controller: Optional[InputController] = None
        self.frame_processor: Optional[FrameProcessor] = None
        self.overlay: Optional[OverlaySystem] = None
        self.hotkey_manager: Optional[HotkeyManager] = None
        self.gui: Optional[ProductionGUI] = None
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        self.metrics_thread: Optional[threading.Thread] = None
        
        # Lifecycle management
        self.main_loop_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()
        
        # Error handling
        self.error_count = 0
        self.max_errors = 50  # Increased from 10 to handle intermittent capture issues
        
        # Signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Operating mode
        self.simulation_mode = False  # Set to True when Arduino is not available
        
        self.logger.info("CVTargetingSystem initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdown()
    
    def initialize(self) -> bool:
        """Initialize all system components"""
        try:
            self.logger.info("Initializing system components...")
            
            # Setup logging
            setup_logging({'level': 'INFO', 'log_dir': 'logs'})
            
            # Load configuration
            self.config_manager = ConfigManager(self.config_path)
            self.config_manager.register_watcher(self._on_config_change)
            
            # Initialize capture
            self.capture = ScreenCapture(config_manager=self.config_manager)
            
            # Reset error count after successful capture initialization
            self.error_count = 0
            
            # Initialize detector
            self.detector = YOLOv8Detector(
                config_manager=None,  # Let detector create its own to avoid circular import
                backend=InferenceBackend.PYTORCH if torch and torch.cuda.is_available() else InferenceBackend.ONNX
            )
            
            # Initialize input controller
            self.input_controller = InputController(
                config_manager=self.config_manager,
                safety_level=SafetyLevel.STANDARD
            )
            
            # Attempt to connect to Arduino with retry logic
            try:
                self.logger.info("Attempting to connect to Arduino...")
                arduino_connected = self.input_controller.connect()
                if arduino_connected:
                    self.simulation_mode = False
                    self.logger.info("Arduino connected successfully")
                else:
                    self.simulation_mode = True
                    self.logger.warning("Arduino not available - continuing in SIMULATION MODE")
                    self.logger.info("To fix Arduino connection:")
                    self.logger.info("  1. Close Arduino IDE or other programs using the port")
                    self.logger.info("  2. Try running as Administrator")
                    self.logger.info("  3. Check the Arduino is properly connected")
            except Exception as arduino_error:
                self.simulation_mode = True
                self.logger.warning(f"Arduino connection failed: {arduino_error}")
                self.logger.warning("Continuing in SIMULATION MODE - all targeting will be logged only")
                if "PermissionError" in str(arduino_error):
                    self.logger.info("Permission Error - Try running as Administrator or close other Arduino programs")
            
            # Initialize frame processor
            self.frame_processor = FrameProcessor(
                self.detector,
                self.input_controller,
                self.config_manager,
                simulation_mode=self.simulation_mode
            )
            
            # Initialize hotkey manager
            self.hotkey_manager = HotkeyManager()
            if self.hotkey_manager is not None:
                self.hotkey_manager.register_callback('toggle_system', self.toggle_system)
                self.hotkey_manager.register_callback('toggle_overlay', self.toggle_overlay)
                self.hotkey_manager.register_callback('reload_config', self.reload_config)
            
            # Register performance anomaly handler
            self.performance_monitor.register_anomaly_callback(self._on_performance_anomaly)
            
            # Set ready state
            self.state = ApplicationState.READY
            self.logger.info("System initialization complete")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}", exc_info=True)
            self.state = ApplicationState.ERROR
            return False
    
    def _on_config_change(self, config: Dict[str, Any]) -> None:
        """Handle configuration changes"""
        self.logger.info("Configuration changed, applying updates...")
        
        # Update capture region
        if self.capture and self.config_manager is not None:
            try:
                new_region = self.config_manager.get_screen_region()
                self.capture.set_region(new_region)
            except Exception as e:
                self.logger.error(f"Failed to update capture region: {e}")
    
    def _on_performance_anomaly(self, metric: str, value: float) -> None:
        """Handle performance anomalies"""
        self.logger.warning(f"Performance anomaly detected - {metric}: {value}")
        
        # Auto-adjust based on anomaly
        if metric == 'latency' and value > 50:  # 50ms latency
            self.logger.info("High latency detected, reducing quality...")
            # Could adjust detection model precision or capture resolution
    
    def run(self) -> None:
        """Main application entry point"""
        # Initialize components
        if not self.initialize():
            self.logger.error("Failed to initialize, exiting...")
            return
        
        # Start performance monitoring
        self.metrics_thread = threading.Thread(
            target=self._metrics_worker,
            name="MetricsWorker",
            daemon=True
        )
        self.metrics_thread.start()
        
        # Launch UI based on preference
        ui_mode = self._get_ui_preference()
        
        if ui_mode == 'gui' and GUI_AVAILABLE:
            self._run_with_gui()
        elif ui_mode == 'overlay':
            self._run_with_overlay()
        else:
            self._run_headless()
    
    def _get_ui_preference(self) -> str:
        """Get UI preference from user"""
        # Check command line arguments first
        if len(sys.argv) > 1:
            mode = sys.argv[1].lower()
            if mode in ['gui', 'overlay', 'headless']:
                return mode
        
        # Check for --ui argument
        for i, arg in enumerate(sys.argv):
            if arg == '--ui' and i + 1 < len(sys.argv):
                mode = sys.argv[i + 1].lower()
                if mode in ['gui', 'overlay', 'headless']:
                    return mode
        
        # Default to GUI if available, otherwise headless
        if GUI_AVAILABLE:
            self.logger.info("No UI mode specified, defaulting to GUI")
            return 'gui'
        else:
            self.logger.info("GUI not available, defaulting to headless mode")
            return 'headless'
    
    def _run_with_gui(self) -> None:
        """Run with GUI interface"""
        self.logger.info("Starting with GUI interface...")
        
        # Create GUI
        if self.config_manager is not None:
            self.gui = ProductionGUI(self.config_manager)
        else:
            self.logger.error("Config manager is not initialized. Cannot start GUI.")
            return
        
        # Start main loop in thread
        self.main_loop_thread = threading.Thread(
            target=self._main_loop,
            name="MainLoop",
            daemon=True
        )
        self.main_loop_thread.start()
        
        # Run GUI (blocking)
        if self.gui is not None:
            self.gui.run()
        
        # GUI closed, shutdown
        self.shutdown()
    
    def _run_with_overlay(self) -> None:
        """Run with overlay interface"""
        self.logger.info("Starting with overlay interface...")
        
        # Create overlay
        self.overlay = OverlaySystem(
            mode=OverlayMode.TRANSPARENT,
            config_manager=self.config_manager
        )
        
        # Start overlay in thread
        overlay_thread = threading.Thread(
            target=self.overlay.run,
            daemon=True
        )
        overlay_thread.start()
        
        # Run main loop
        self._main_loop()
    
    def _run_headless(self) -> None:
        """Run without UI"""
        self.logger.info("Starting in headless mode...")
        
        print("\nSystem running. Press Ctrl+C to stop.")
        if self.hotkey_manager is not None and hasattr(self.hotkey_manager, 'get_bindings'):
            print(f"Hotkeys: {self.hotkey_manager.get_bindings()}")
        else:
            print("Hotkeys: Not available")
        
        # Run main loop
        self._main_loop()
    
    def _main_loop(self) -> None:
        """Main processing loop"""
        self.logger.info("Starting main processing loop...")
        
        # Start components
        if self.frame_processor is not None:
            self.frame_processor.start()
        else:
            self.logger.error("Frame processor is not initialized.")
            return
        
        if self.hotkey_manager is not None:
            self.hotkey_manager.start()
        
        self.state = ApplicationState.RUNNING
        
        # Performance tracking
        frame_times = deque(maxlen=100)
        last_metrics_log = time.time()
        
        # Startup grace period for more lenient error handling
        startup_time = time.time()
        startup_grace_period = 10.0  # 10 seconds of more lenient error handling
        consecutive_errors = 0
        successful_captures = 0
        
        self.logger.info("Main loop started with startup grace period")
        
        while not self.shutdown_event.is_set():
            try:
                if self.state != ApplicationState.RUNNING:
                    time.sleep(0.1)
                    continue
                
                loop_start = time.perf_counter()
                
                # Capture frame
                capture_start = time.perf_counter()
                frame = self.capture.capture() if self.capture is not None else None
                capture_time = (time.perf_counter() - capture_start) * 1000
                
                if frame is None:
                    consecutive_errors += 1
                    self.error_count += 1
                    
                    # During startup grace period, be more lenient
                    current_time = time.time()
                    in_grace_period = (current_time - startup_time) < startup_grace_period
                    
                    if in_grace_period:
                        # During startup, only fail if we have too many consecutive errors
                        if consecutive_errors > 20:  # Allow more consecutive failures during startup
                            self.logger.error(f"Too many consecutive capture errors during startup ({consecutive_errors}), shutting down...")
                            break
                        elif consecutive_errors % 5 == 0:  # Log every 5th error during startup
                            self.logger.warning(f"Capture errors during startup: {consecutive_errors}/20 (system may still be warming up)")
                    else:
                        # After grace period, use normal error handling
                        if self.error_count > self.max_errors:
                            self.logger.error(f"Too many capture errors ({self.error_count}/{self.max_errors}), shutting down...")
                            break
                    continue
                
                # Reset error counts on successful capture
                consecutive_errors = 0
                self.error_count = max(0, self.error_count - 1)  # Gradually reduce error count on success
                successful_captures += 1
                
                # Log startup completion
                if successful_captures == 10:
                    self.logger.info("Capture pipeline stabilized - 10 successful captures completed")
                
                # Process frame
                if self.frame_processor is not None:
                    if not self.frame_processor.process_frame(frame):
                        self.performance_monitor.metrics.dropped_frames += 1
                
                # Update metrics
                self.performance_monitor.update('capture', capture_time)
                
                # Send frame to overlay if active
                if self.overlay:
                    self.overlay.update_frame(frame)
                
                # Track loop time
                loop_time = (time.perf_counter() - loop_start) * 1000
                frame_times.append(loop_time)
                
                # Log metrics periodically
                if time.time() - last_metrics_log > 5.0:
                    metrics = self.performance_monitor.get_metrics()
                    self.logger.info(f"System metrics: {metrics.to_dict()}")
                    last_metrics_log = time.time()
                
                # Frame rate limiting
                target_frame_time = 1.0 / 144  # 144 FPS target
                elapsed = time.perf_counter() - loop_start
                if elapsed < target_frame_time:
                    time.sleep(target_frame_time - elapsed)
                
            except KeyboardInterrupt:
                self.logger.info("Keyboard interrupt received")
                break
            except Exception as e:
                self.logger.error(f"Main loop error: {e}", exc_info=True)
                self.error_count += 1
                
                if self.error_count > self.max_errors:
                    self.logger.error("Too many errors, shutting down...")
                    break
        
        self.logger.info("Main loop ended")
    
    def _metrics_worker(self) -> None:
        """Background metrics collection"""
        while not self.shutdown_event.is_set():
            try:
                # Collect detector metrics
                if self.detector:
                    detector_metrics = self.detector.get_metrics()
                    if 'inference' in detector_metrics:
                        avg_inference = detector_metrics['inference']['avg_us'] / 1000
                        self.performance_monitor.update('detection', avg_inference)
                
                # Collect input metrics
                if self.input_controller:
                    input_metrics = self.input_controller.get_metrics()
                    if 'metrics' in input_metrics:
                        latency = input_metrics['metrics'].get('latency_avg_us', 0) / 1000
                        self.performance_monitor.update('input', latency)
                
                # Collect capture metrics
                if self.capture:
                    capture_metrics = self.capture.get_metrics()
                    if 'avg_ms' in capture_metrics:
                        self.performance_monitor.update('capture', capture_metrics['avg_ms'])
                
            except Exception as e:
                self.logger.error(f"Metrics collection error: {e}")
            
            time.sleep(1.0)
    
    def toggle_system(self) -> None:
        """Toggle system running state"""
        if self.state == ApplicationState.RUNNING:
            self.state = ApplicationState.PAUSED
            self.logger.info("System paused")
        elif self.state == ApplicationState.PAUSED:
            self.state = ApplicationState.RUNNING
            self.logger.info("System resumed")
    
    def toggle_overlay(self) -> None:
        """Toggle overlay visibility"""
        if self.overlay:
            self.overlay.toggle_visibility()
    
    def reload_config(self) -> None:
        """Reload configuration"""
        self.logger.info("Reloading configuration...")
        if self.config_manager is not None:
            self.config_manager.reload()
        else:
            self.logger.error("Config manager is not initialized. Cannot reload.")
    
    def shutdown(self) -> None:
        """Clean shutdown procedure"""
        if self.state == ApplicationState.SHUTTING_DOWN:
            return
        
        self.logger.info("Initiating shutdown sequence...")
        self.state = ApplicationState.SHUTTING_DOWN
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Stop components in order
        if self.frame_processor:
            self.frame_processor.stop()
        
        if self.hotkey_manager:
            self.hotkey_manager.stop()
        
        if self.overlay:
            self.overlay.stop()
        
        if self.input_controller:
            self.input_controller.disconnect()
        
        if self.detector:
            self.detector.cleanup()
        
        if self.capture:
            self.capture.cleanup()
        
        # Wait for threads
        if self.main_loop_thread and self.main_loop_thread.is_alive():
            self.main_loop_thread.join(timeout=5.0)
        
        if self.metrics_thread and self.metrics_thread.is_alive():
            self.metrics_thread.join(timeout=2.0)
        
        self.logger.info("Shutdown complete")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()
        return False
    
    def enable_gui(self) -> bool:
        """Enable GUI interface if available"""
        try:
            if not self.config_manager:
                self.logger.error("Cannot enable GUI: system not initialized")
                return False
                
            if GUI_AVAILABLE:
                self.gui = ProductionGUI(self.config_manager)
                self.logger.info("GUI enabled")
                return True
            else:
                self.logger.warning("GUI not available")
                return False
        except Exception as e:
            self.logger.error(f"Failed to enable GUI: {e}")
            return False
    
    def enable_overlay(self) -> bool:
        """Enable overlay system if available"""
        try:
            if not self.config_manager:
                self.logger.error("Cannot enable overlay: system not initialized")
                return False
                
            try:
                from .pygame_overlay import OverlaySystem, OverlayMode
            except ImportError:
                from pygame_overlay import OverlaySystem, OverlayMode
            screen_region = self.config_manager.get_screen_region()
            self.overlay = OverlaySystem(
                config_manager=self.config_manager
            )
            self.logger.info("Overlay enabled")
            return True
        except ImportError:
            self.logger.warning("Overlay system not available")
            return False
        except Exception as e:
            self.logger.error(f"Failed to enable overlay: {e}")
            return False


def main():
    """Application entry point"""
    # Parse arguments
    parser = argparse.ArgumentParser(description="AI Aim Assist System")
    parser.add_argument('--config', default='configs/config.ini', help='Configuration file path')
    parser.add_argument('--ui', choices=['gui', 'overlay', 'headless'], help='UI mode')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()
    
    # Setup logging
    log_level = 'DEBUG' if args.debug else 'INFO'
    setup_logging({'level': log_level})
    
    # Create and run system
    with CVTargetingSystem(args.config) as system:
        try:
            system.run()
        except Exception as e:
            logger = get_logger(__name__)
            logger.error(f"Fatal error: {e}", exc_info=True)
            sys.exit(1)


if __name__ == "__main__":
    main()
