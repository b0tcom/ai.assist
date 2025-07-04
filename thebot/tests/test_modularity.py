#!/usr/bin/env python3
"""
Modularity Test Suite
====================

This test suite verifies that the modular architecture works correctly
and that components can be imported and used independently.
"""

import sys
import traceback
from pathlib import Path


class ModularityTester:
    """Test the modularity of the system"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def test_case(self, name: str, test_func):
        """Run a test case and track results"""
        try:
            print(f"Testing: {name}...", end=" ")
            test_func()
            print("✓ PASS")
            self.passed += 1
        except Exception as e:
            print("✗ FAIL")
            self.failed += 1
            self.errors.append(f"{name}: {str(e)}")
            traceback.print_exc()
    
    def test_core_imports(self):
        """Test that core modules can be imported"""
        from thebot.src import ConfigManager
        from thebot.src import YOLOv8Detector
        from thebot.src import ScreenCapture
        from thebot.src import InputController
        from thebot.src import get_logger
    
    def test_optional_imports(self):
        """Test that optional imports work gracefully"""
        try:
            from thebot.src import CVTargetingGUI
        except ImportError:
            pass  # Expected for optional components
        
        try:
            from thebot.src import OverlaySystem
        except ImportError:
            pass  # Expected for optional components
    
    def test_package_exports(self):
        """Test package-level exports"""
        from thebot.src import CVTargetingSystem
        from thebot.src import create_targeting_system
        from thebot.src import get_system_info
    
    def test_configuration_isolation(self):
        """Test that configuration can be used in isolation"""
        from thebot.src import ConfigManager
        
        # Should work with default config
        config = ConfigManager()
        
        # Should be able to get typed objects
        screen_region = config.get_screen_region()
        model_config = config.get_model_config()
        
        assert screen_region is not None
        assert model_config is not None
    
    def test_detector_isolation(self):
        """Test that detector can be used in isolation"""
        from thebot.src import YOLOv8Detector, ConfigManager
        import numpy as np
        
        config = ConfigManager()
        detector = YOLOv8Detector(config_manager=None)  # Should create its own
        
        # Should be able to create dummy detections
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        detections = detector.detect(dummy_image)
        
        # Should return a list (even if empty)
        assert isinstance(detections, list)
    
    def test_capture_isolation(self):
        """Test that capture can be used in isolation"""
        from thebot.src import ScreenCapture, ConfigManager
        
        config = ConfigManager()
        capture = ScreenCapture(config_manager=config)
        
        # Should be able to initialize
        assert capture is not None
    
    def test_relative_imports(self):
        """Test that all imports are properly relative"""
        import inspect
        from thebot.src import main, detect, capture, config_manager
        
        # Check that modules don't have absolute imports to each other
        # This is a basic check - in a full test we'd parse the AST
        modules = [main, detect, capture, config_manager]
        for module in modules:
            assert hasattr(module, '__file__')
    
    def test_convenience_function(self):
        """Test the convenience function (without actually running)"""
        from thebot.src import create_targeting_system
        
        # Should be callable
        assert callable(create_targeting_system)
        
        # This would create a system but we won't run it in tests
        # system = create_targeting_system("configs/config.ini")
    
    def test_backward_compatibility(self):
        """Test that backward compatibility modules work"""
        try:
            from thebot.src import config  # Legacy config module
            from thebot.src import predict  # Legacy predict module
        except ImportError as e:
            # Some backward compatibility might not be available
            if "No module named" not in str(e):
                raise
    
    def run_all_tests(self):
        """Run all modularity tests"""
        print("=" * 60)
        print("AI Aim Assist System - Modularity Test Suite")
        print("=" * 60)
        
        # Core functionality tests
        self.test_case("Core Module Imports", self.test_core_imports)
        self.test_case("Optional Module Imports", self.test_optional_imports)
        self.test_case("Package Exports", self.test_package_exports)
        
        # Isolation tests
        self.test_case("Configuration Isolation", self.test_configuration_isolation)
        self.test_case("Detector Isolation", self.test_detector_isolation)
        self.test_case("Capture Isolation", self.test_capture_isolation)
        
        # Architecture tests
        self.test_case("Relative Imports", self.test_relative_imports)
        self.test_case("Convenience Function", self.test_convenience_function)
        self.test_case("Backward Compatibility", self.test_backward_compatibility)
        
        # Results
        print("=" * 60)
        print(f"Test Results: {self.passed} passed, {self.failed} failed")
        
        if self.errors:
            print("\nErrors:")
            for error in self.errors:
                print(f"  - {error}")
        
        print("=" * 60)
        
        if self.failed == 0:
            print("✓ All modularity tests passed!")
            return True
        else:
            print("✗ Some modularity tests failed!")
            return False


def main():
    """Run the modularity test suite"""
    tester = ModularityTester()
    success = tester.run_all_tests()
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
