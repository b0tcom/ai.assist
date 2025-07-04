"""
AI Aim Assist System - Production-Grade Modular Architecture
============================================================

This package provides a complete AI-powered aim assistance system with:
- Hardware-only mouse control via Arduino
- CUDA-accelerated YOLOv8 object detection  
- Modular configuration management
- High-performance screen capture
- Safety protocols and ethical constraints

Main Components:
---------------
- detect: YOLOv8 detection engine with hardware interface
- capture: DPI-aware screen capture with multiple backends
- input_handler: Arduino mouse control with safety protocols
- config_manager: Centralized configuration with hot-reload
- gui: Configuration interface
- logger_util: Performance logging and monitoring

Usage:
------
    from thebot.src import CVTargetingSystem
    
    system = CVTargetingSystem("configs/config.ini")
    system.run()

Safety:
-------
This system enforces hardware-only mouse control and includes multiple
safety mechanisms to ensure ethical usage.
"""

__version__ = "2.0.0"
__author__ = "AI Gaming Analysis System"

# Core exports for modular usage
from .main import CVTargetingSystem, ApplicationState
from .detect import YOLOv8Detector, Detection, InferenceBackend
from .config_manager import ConfigManager, ScreenRegion, ModelConfig
from .capture import ScreenCapture, get_frame
from .input_handler import InputController, SafetyLevel
from .logger_util import get_logger, setup_logging

# Optional exports (may not be available)
try:
    from .gui import CVTargetingGUI
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False

try:
    from .pygame_overlay import OverlaySystem, OverlayMode
    OVERLAY_AVAILABLE = True
except ImportError:
    OVERLAY_AVAILABLE = False

try:
    from .toggle import HotkeyManager
    HOTKEY_AVAILABLE = True
except ImportError:
    HOTKEY_AVAILABLE = False

# Convenience function for easy initialization
def create_targeting_system(config_path: str = "configs/config.ini", 
                          gui_enabled: bool = False,
                          overlay_enabled: bool = False) -> CVTargetingSystem:
    """
    Create a targeting system with the specified configuration.
    
    Args:
        config_path: Path to configuration file
        gui_enabled: Whether to enable GUI interface
        overlay_enabled: Whether to enable overlay system
        
    Returns:
        Configured CVTargetingSystem instance
    """
    system = CVTargetingSystem(config_path)
    
    # Initialize the system first
    if not system.initialize():
        raise RuntimeError("Failed to initialize targeting system")
    
    if gui_enabled and GUI_AVAILABLE:
        system.enable_gui()
    
    if overlay_enabled and OVERLAY_AVAILABLE:
        system.enable_overlay()
    
    return system

# Version and capability information
def get_system_info():
    """Get system capabilities and version information"""
    return {
        'version': __version__,
        'author': __author__,
        'capabilities': {
            'gui': GUI_AVAILABLE,
            'overlay': OVERLAY_AVAILABLE,
            'hotkeys': HOTKEY_AVAILABLE,
        },
        'safety': {
            'hardware_only': True,
            'arduino_required': True,
            'ethical_constraints': True
        }
    }

__all__ = [
    # Core classes
    'CVTargetingSystem',
    'YOLOv8Detector', 
    'ConfigManager',
    'ScreenCapture',
    'InputController',
    
    # Enums and types
    'ApplicationState',
    'Detection',
    'InferenceBackend',
    'ScreenRegion',
    'ModelConfig',
    'SafetyLevel',
    
    # Utility functions
    'get_logger',
    'setup_logging',
    'get_frame',
    'create_targeting_system',
    'get_system_info',
    
    # Optional components (if available)
    'CVTargetingGUI',
    'OverlaySystem',
    'OverlayMode', 
    'HotkeyManager',
    
    # Status flags
    'GUI_AVAILABLE',
    'OVERLAY_AVAILABLE',
    'HOTKEY_AVAILABLE',
]
