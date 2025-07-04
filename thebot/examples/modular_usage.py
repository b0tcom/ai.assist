#!/usr/bin/env python3
"""
Modular Usage Examples for AI Aim Assist System
===============================================

This file demonstrates various ways to use the modular components
of the AI aim assist system.
"""

# Example 1: Simple usage with convenience function
def example_simple_usage():
    """Simple usage example using the convenience function"""
    from thebot.src import create_targeting_system
    
    # Create system with default configuration
    system = create_targeting_system(
        config_path="configs/config.ini",
        gui_enabled=True,
        overlay_enabled=True
    )
    
    # Run the system
    system.run()


# Example 2: Manual component initialization for advanced usage
def example_manual_initialization():
    """Advanced usage with manual component initialization"""
    from thebot.src import (
        CVTargetingSystem, 
        ConfigManager, 
        YOLOv8Detector,
        ScreenCapture,
        InputController,
        SafetyLevel
    )
    
    # Initialize configuration
    config_manager = ConfigManager("configs/config.ini")
    
    # Create detector with specific backend
    detector = YOLOv8Detector(config_manager=config_manager)
    
    # Create screen capture
    capture = ScreenCapture(config_manager=config_manager)
    
    # Create input controller with strict safety
    input_controller = InputController(
        config_manager=config_manager,
        safety_level=SafetyLevel.STRICT
    )
    
    # Create main system
    system = CVTargetingSystem("configs/config.ini")
    system.initialize()
    
    # Run system
    system.run()


# Example 3: Testing individual components
def example_component_testing():
    """Example of testing individual components in isolation"""
    from thebot.src import ConfigManager, YOLOv8Detector
    import numpy as np
    
    # Test configuration
    config = ConfigManager("configs/config.ini")
    screen_region = config.get_screen_region()
    print(f"Screen region: {screen_region}")
    
    # Test detector
    detector = YOLOv8Detector(config_manager=config)
    
    # Test with dummy image
    dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
    detections = detector.detect(dummy_image)
    print(f"Detections: {len(detections)}")


# Example 4: Using context managers for proper cleanup
def example_context_manager():
    """Example using context managers for proper resource management"""
    from thebot.src import CVTargetingSystem
    
    # Use context manager for automatic cleanup
    with CVTargetingSystem("configs/config.ini") as system:
        system.initialize()
        system.enable_gui()
        system.run()
    # System automatically cleaned up here


# Example 5: Custom configuration
def example_custom_configuration():
    """Example of creating custom configuration programmatically"""
    from thebot.src import ConfigManager, ScreenRegion, ModelConfig
    
    config = ConfigManager()
    
    # Set custom screen region
    custom_region = ScreenRegion(
        left=100, 
        top=100, 
        width=500, 
        height=500
    )
    
    # Update configuration
    config.set("Application", "capture_region_width", custom_region.width)
    config.set("Application", "capture_region_height", custom_region.height)
    config.save()


# Example 6: Performance monitoring
def example_performance_monitoring():
    """Example of using performance monitoring features"""
    from thebot.src import CVTargetingSystem
    
    system = CVTargetingSystem("configs/config.ini")
    system.initialize()
    
    # Get performance metrics
    metrics = system.performance_monitor.get_metrics()
    print(f"System metrics: {metrics}")
    
    # Register custom performance callback
    def on_performance_anomaly(metric_type: str, value: float):
        print(f"Performance anomaly: {metric_type} = {value}")
    
    system.performance_monitor.register_anomaly_callback(on_performance_anomaly)


if __name__ == "__main__":
    print("AI Aim Assist System - Modular Usage Examples")
    print("=" * 50)
    
    # You can uncomment any of these examples to test them:
    
    # example_simple_usage()
    # example_manual_initialization()
    # example_component_testing()
    # example_context_manager()
    # example_custom_configuration()
    # example_performance_monitoring()
    
    print("Examples available. Uncomment the desired example to run it.")
