
#!/usr/bin/env python3
"""
Main entry point for the AI Aim Assist System.
This file serves as the primary launch point referenced by .replit configuration.
"""

import sys
import os

# Add the thebot/src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'thebot', 'src'))

def main():
    """Main entry point that launches the targeting system."""
    try:
        from main import CVTargetingSystem
        print("🎯 Initializing AI Aim Assist System...")
        system = CVTargetingSystem()
        system.run()
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Please ensure all dependencies are installed:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting system: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
