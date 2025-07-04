"""
Configuration Backward Compatibility Module
Purpose: Maintains compatibility with legacy code expecting the old Config class
"""
import warnings
from pathlib import Path
from typing import Optional, Dict, Any

# Import the new configuration system
from .config_manager import ConfigManager, ScreenRegion, ModelConfig, AimConfig
from .logger_util import get_logger

# Show deprecation warning
warnings.warn(
    "config.py is deprecated. Please use config_manager.py instead.",
    DeprecationWarning,
    stacklevel=2
)


class Config:
    """
    Legacy Config class wrapper around new ConfigManager
    Provides backward compatibility for existing code
    """
    
    def __init__(self, config_path: Optional[str] = None, cli_args: Optional[Any] = None):
        self.logger = get_logger(__name__)
        
        # Use new ConfigManager internally
        if config_path is None:
            # Check for old paths
            if Path("src/configs/config.ini").exists():
                config_path = "src/configs/config.ini"
            elif Path("configs/config.ini").exists():
                config_path = "configs/config.ini"
            else:
                config_path = "configs/default_config.json"
        
        self._manager = ConfigManager(config_path)
        self._config = self._manager.as_dict()
        
        # Handle CLI overrides
        if cli_args:
            self._override_with_cli(cli_args)
        
        self.logger.info("Legacy Config wrapper initialized")
    
    def _override_with_cli(self, cli_args: Any) -> None:
        """Override with CLI arguments (legacy compatibility)"""
        overrides = {key: value for key, value in vars(cli_args).items() if value is not None}
        
        if overrides:
            self.logger.info(f"Overriding config with CLI arguments: {overrides}")
            
            # Map legacy CLI args to new structure
            cli_to_config_map = {
                'aim_offset': ('aim_settings', 'aim_height'),
                'max_distance': ('aim_settings', 'max_distance'),
                'confidence_threshold': ('model_settings', 'confidence'),
                'model_path': ('model_settings', 'model_path')
            }
            
            for cli_key, value in overrides.items():
                if cli_key in cli_to_config_map:
                    section, key = cli_to_config_map[cli_key]
                    self._manager.set(section, key, value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value (legacy method)"""
        # First check flat config
        if key in self._config:
            return self._config[key]
        
        # Then check nested values
        for section, values in self._config.items():
            if isinstance(values, dict) and key in values:
                return values[key]
        
        return default
    
    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access"""
        value = self.get(key)
        if value is None:
            raise KeyError(f"Configuration key not found: {key}")
        return value
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists"""
        return self.get(key) is not None
    
    def to_dict(self) -> Dict[str, Any]:
        """Get full configuration as dictionary"""
        return self._config
    
    # Add commonly accessed properties for compatibility
    @property
    def screen_region(self) -> Dict[str, int]:
        """Get screen region as dictionary"""
        region = self._manager.get_screen_region()
        return region.to_dict()
    
    @property
    def model_path(self) -> str:
        """Get model path"""
        return self._manager.get_model_config().model_path
    
    @property
    def confidence_threshold(self) -> float:
        """Get confidence threshold"""
        return self._manager.get_model_config().confidence_threshold
    
    @property
    def arduino_port(self) -> str:
        """Get Arduino port"""
        return self._manager.get_arduino_config().port
    
    @property
    def aim_height_offset(self) -> float:
        """Get aim height offset"""
        return self._manager.get_aim_config().aim_height_offset


# Global instance for backward compatibility
CONFIG: Optional[Config] = None


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration (legacy function)
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config instance
    """
    global CONFIG
    CONFIG = Config(config_path)
    return CONFIG


# Auto-load default config if imported
if CONFIG is None:
    try:
        CONFIG = load_config()
    except Exception as e:
        logger = get_logger(__name__)
        logger.warning(f"Failed to auto-load config: {e}")