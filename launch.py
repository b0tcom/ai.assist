#!/usr/bin/env python3
"""
AI Aim Assist System - Python Launcher
=====================================

This launcher ensures proper module imports and handles different execution contexts.
It can be used as an alternative to the batch file for cross-platform compatibility.
"""

import sys
import os
from pathlib import Path

def setup_python_path():
    """Setup Python path for proper imports"""
    # Get the directory containing this script
    project_root = Path(__file__).parent.absolute()
    
    # Add project root to Python path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Add thebot directory to Python path for module imports
    thebot_path = project_root / "thebot"
    if str(thebot_path) not in sys.path:
        sys.path.insert(0, str(thebot_path))
    
    return project_root

def check_dependencies():
    """Check for critical dependencies"""
    missing_deps = []
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        import cv2
    except ImportError:
        missing_deps.append("opencv-python")
    
    try:
        import serial
    except ImportError:
        missing_deps.append("pyserial")
    
    if missing_deps:
        print("⚠️  Missing dependencies:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\nInstall with: pip install " + " ".join(missing_deps))
        return False
    
    return True

def main():
    """Main launcher function"""
    print("🚀 AI Aim Assist System - Python Launcher")
    print("=" * 50)
    
    # Setup paths
    project_root = setup_python_path()
    os.chdir(project_root)
    
    print(f"📁 Project root: {project_root}")
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    if not check_dependencies():
        print("❌ Dependency check failed!")
        sys.exit(1)
    
    print("✅ Dependencies OK")
    
    # Import and run the main application
    try:
        # Try importing as module first
        try:
            from thebot.src.main import main as app_main
            print("📦 Imported as module")
        except ImportError:
            # Fallback to direct import
            sys.path.insert(0, str(project_root / "thebot" / "src"))
            from main import main as app_main
            print("📦 Imported directly")
        
        print("🎯 Starting AI Aim Assist System...")
        print("=" * 50)
        print()
        
        # Run the application
        app_main()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure you're in the correct directory")
        print("2. Check that thebot/src/main.py exists")
        print("3. Verify all dependencies are installed")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
