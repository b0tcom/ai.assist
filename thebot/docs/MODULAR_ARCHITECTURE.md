# AI Aim Assist System - Modular Architecture

## Overview

This system has been designed with complete modularity in mind, ensuring that each component can be used independently or as part of the complete system. The modular design provides flexibility, maintainability, and testability.

## Architecture

### Core Modules

1. **`detect.py`** - YOLOv8 detection engine with hardware interface
2. **`capture.py`** - DPI-aware screen capture with multiple backends  
3. **`input_handler.py`** - Arduino mouse control with safety protocols
4. **`config_manager.py`** - Centralized configuration management
5. **`main.py`** - System orchestration and lifecycle management
6. **`logger_util.py`** - Performance logging and monitoring

### Optional Modules

7. **`gui.py`** - Configuration interface (optional)
8. **`pygame_overlay.py`** - Overlay system (optional)
9. **`toggle.py`** - Hotkey management (optional)
10. **`utils.py`** - Utility functions and helpers

### Backward Compatibility Modules

11. **`config.py`** - Legacy configuration wrapper
12. **`predict.py`** - Legacy prediction wrapper

## Modular Design Principles

### 1. Dependency Injection

Each module accepts its dependencies through constructor parameters:

```python
# Components can be configured independently
config_manager = ConfigManager("custom_config.ini")
detector = YOLOv8Detector(config_manager=config_manager)
capture = ScreenCapture(config_manager=config_manager)
```

### 2. Interface Segregation

Modules only depend on the interfaces they need:

```python
# InputController only needs config for Arduino settings
input_controller = InputController(
    config_manager=config_manager,
    safety_level=SafetyLevel.STRICT
)
```

### 3. Single Responsibility

Each module has a clear, focused purpose:

- `detect.py`: Object detection and targeting logic
- `capture.py`: Screen capture functionality  
- `input_handler.py`: Hardware input control
- `config_manager.py`: Configuration management

### 4. Relative Imports

All internal imports use relative imports for modularity:

```python
from .logger_util import get_logger
from .config_manager import ConfigManager
```

### 5. Optional Dependencies

Optional features gracefully handle missing dependencies:

```python
try:
    from .gui import CVTargetingGUI
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
```

## Usage Patterns

### Simple Usage

```python
from thebot.src import create_targeting_system

# Create and run with default configuration
system = create_targeting_system("configs/config.ini")
system.run()
```

### Component-Level Usage

```python
from thebot.src import ConfigManager, YOLOv8Detector

# Use individual components
config = ConfigManager("configs/config.ini")
detector = YOLOv8Detector(config_manager=config)

# Test detection
import numpy as np
image = np.zeros((640, 640, 3), dtype=np.uint8)
detections = detector.detect(image)
```

### Advanced Configuration

```python
from thebot.src import CVTargetingSystem, SafetyLevel

system = CVTargetingSystem("configs/config.ini")
system.initialize()

# Configure safety level
system.input_controller.safety_level = SafetyLevel.STRICT

# Enable optional features
system.enable_gui()
system.enable_overlay()

system.run()
```

## Configuration Management

The modular configuration system supports multiple formats and hot-reloading:

```python
# Supports INI, JSON, and YAML
config = ConfigManager("config.ini")    # INI format
config = ConfigManager("config.json")   # JSON format  
config = ConfigManager("config.yaml")   # YAML format

# Get typed configuration objects
screen_region = config.get_screen_region()
model_config = config.get_model_config()
arduino_config = config.get_arduino_config()
```

## Testing Individual Components

Each module can be tested in isolation:

```python
# Test capture system
from thebot.src import ScreenCapture, ConfigManager

config = ConfigManager("test_config.ini")
capture = ScreenCapture(config_manager=config)
frame = capture.get_frame()

# Test detection system
from thebot.src import YOLOv8Detector
detector = YOLOv8Detector(config_manager=config)
detections = detector.detect(frame)
```

## Error Handling

The modular design includes comprehensive error handling at each level:

```python
# System-level error handling
try:
    system = CVTargetingSystem("configs/config.ini")
    if not system.initialize():
        print("Failed to initialize system")
        exit(1)
    system.run()
except Exception as e:
    logger.error(f"System error: {e}")
```

## Performance Monitoring

Each module includes performance monitoring:

```python
# Get performance metrics from any component
detector_metrics = detector.get_metrics()
capture_metrics = capture.get_performance_stats()
system_metrics = system.performance_monitor.get_metrics()
```

## Safety and Ethics

The modular design enforces safety at multiple levels:

1. **Hardware-only mouse control** - No software mouse movement
2. **Safety levels** - Configurable safety constraints
3. **Rate limiting** - Prevents excessive movement commands
4. **Bounds checking** - Validates all movement commands

## Extension Points

The modular design makes it easy to extend functionality:

```python
# Custom detection backend
class CustomDetector(DetectorBackend):
    def load_model(self, model_path: str, config: ModelConfig):
        # Custom implementation
        pass
    
    def infer(self, image: np.ndarray):
        # Custom inference logic
        pass

# Custom input backend  
class CustomInputController(InputBackend):
    def move_to_target(self, target: Dict, metadata: Dict):
        # Custom movement logic
        pass
```

## Best Practices

1. **Always use dependency injection** - Pass dependencies through constructors
2. **Use relative imports** - Ensures modularity
3. **Handle optional dependencies** - Graceful degradation when features unavailable
4. **Implement proper cleanup** - Use context managers or explicit cleanup
5. **Follow single responsibility** - Each module should have one clear purpose
6. **Use typed configuration** - Leverage the typed config objects
7. **Monitor performance** - Use built-in performance monitoring
8. **Respect safety constraints** - Never bypass hardware-only enforcement

## File Structure

```
thebot/
├── __init__.py              # Package exports
├── src/
│   ├── __init__.py          # Core module exports  
│   ├── main.py              # System orchestration
│   ├── detect.py            # Detection engine
│   ├── capture.py           # Screen capture
│   ├── input_handler.py     # Hardware control
│   ├── config_manager.py    # Configuration
│   ├── logger_util.py       # Logging utilities
│   ├── gui.py               # GUI interface (optional)
│   ├── pygame_overlay.py    # Overlay system (optional)
│   ├── toggle.py            # Hotkey management (optional)
│   ├── utils.py             # Utility functions
│   ├── config.py            # Legacy compatibility
│   └── predict.py           # Legacy compatibility
└── examples/
    └── modular_usage.py     # Usage examples
```

This modular architecture ensures that the system is maintainable, testable, and extensible while maintaining the highest standards of safety and performance.
