"""
AI Aim Assist System - Root Package
===================================

This is the root package for the AI aim assist system.
All functionality is contained in the 'src' subpackage.

Quick Start:
-----------
    from thebot.src import create_targeting_system
    
    # Create and run the targeting system
    system = create_targeting_system("configs/config.ini")
    system.run()

For more advanced usage, see the src package documentation.
"""

__version__ = "2.0.0"

# Re-export main functionality from src
from .src import (
    CVTargetingSystem,
    create_targeting_system,
    get_system_info,
    GUI_AVAILABLE,
    OVERLAY_AVAILABLE,
    HOTKEY_AVAILABLE
)

__all__ = [
    'CVTargetingSystem',
    'create_targeting_system', 
    'get_system_info',
    'GUI_AVAILABLE',
    'OVERLAY_AVAILABLE',
    'HOTKEY_AVAILABLE'
]
