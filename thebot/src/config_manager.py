"""
Production-Grade Configuration Management with Runtime Validation
Purpose: Centralized configuration with type safety and hot-reload support
"""
import configparser
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union, TypeVar, Generic, List
from dataclasses import dataclass, field
from enum import Enum
import threading
import hashlib
from datetime import datetime

# Fixed import - remove relative import
import logging

def get_logger(name):
    return logging.getLogger(name)

# YAML support (optional)
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None


class ConfigFormat(Enum):
    """Supported configuration formats"""
    INI = "ini"
    JSON = "json"
    YAML = "yaml"


@dataclass
class ScreenRegion:
    """Validated screen region configuration"""
    left: int
    top: int
    width: int
    height: int
    
    def __post_init__(self):
        """Validate region parameters"""
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Invalid region dimensions: {self.width}x{self.height}")
        if self.left < 0 or self.top < 0:
            raise ValueError(f"Invalid region position: ({self.left}, {self.top})")
    
    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary"""
        return {
            'left': self.left,
            'top': self.top,
            'width': self.width,
            'height': self.height
        }
    
    def validate_against_screen(self, screen_width: int, screen_height: int) -> bool:
        """Validate region fits within screen bounds"""
        return (0 <= self.left < screen_width and
                0 <= self.top < screen_height and
                self.left + self.width <= screen_width and
                self.top + self.height <= screen_height)


@dataclass
class ModelConfig:
    """Model configuration with validation"""
    model_path: str
    confidence_threshold: float = 0.4
    device: str = "cuda"
    precision_mode: str = "float32"
    warmup_iterations: int = 10
    target_class_id: int = 0
    target_class_name: str = "player"
    
    def __post_init__(self):
        """Validate model configuration"""
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError(f"Invalid confidence threshold: {self.confidence_threshold}")
        if self.device not in ["cuda", "cpu"]:
            raise ValueError(f"Invalid device: {self.device}")
        if self.precision_mode not in ["float16", "float32", "mixed"]:
            raise ValueError(f"Invalid precision mode: {self.precision_mode}")


@dataclass
class ArduinoConfig:
    """Arduino configuration"""
    port: str = "COM5"
    baudrate: int = 115200
    timeout: float = 0.1
    
    def __post_init__(self):
        """Validate Arduino configuration"""
        if self.baudrate not in [9600, 19200, 38400, 57600, 115200]:
            raise ValueError(f"Invalid baudrate: {self.baudrate}")


@dataclass
class AimConfig:
    """Aiming configuration"""
    sensitivity: float = 1.0
    max_distance: int = 500
    fov_size: int = 280
    aim_height_offset: float = 0.25
    smoothing_factor: float = 0.3
    kalman_transition_cov: float = 0.01
    kalman_observation_cov: float = 0.01
    delay: float = 5e-05
    
    def __post_init__(self):
        """Validate aim configuration"""
        if not 0 <= self.sensitivity <= 10:
            raise ValueError(f"Invalid sensitivity: {self.sensitivity}")
        if self.fov_size <= 0:
            raise ValueError(f"Invalid FOV size: {self.fov_size}")


class ConfigManager:
    """
    Thread-safe configuration manager with hot-reload and validation
    **CRITICAL: Ensures all mouse movement goes to Arduino hardware only**
    """
    
    def __init__(self, config_path: str = "configs/config.ini", auto_reload: bool = True):
        self.logger = get_logger(__name__)
        self.config_path = Path(config_path)
        self._lock = threading.RLock()
        self._config: Dict[str, Any] = {}
        self._file_hash: Optional[str] = None
        self._last_modified: Optional[float] = None
        self._watchers: List = []
        
        # Configuration caches
        self._screen_region: Optional[ScreenRegion] = None
        self._model_config: Optional[ModelConfig] = None
        self._arduino_config: Optional[ArduinoConfig] = None
        self._aim_config: Optional[AimConfig] = None
        
        # **CRITICAL: Hardware mouse control enforcement**
        self.hardware_mouse_only = True
        
        # Load initial configuration
        self._load()
        
        # Start file watcher if enabled
        if auto_reload:
            self._start_file_watcher()
        
        self.logger.info("ConfigManager initialized - HARDWARE MOUSE CONTROL ENFORCED")
    
    def _get_file_hash(self) -> str:
        """Calculate file hash for change detection"""
        if not self.config_path.exists():
            return ""
        
        with open(self.config_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _detect_format(self) -> ConfigFormat:
        """Detect configuration file format"""
        suffix = self.config_path.suffix.lower()
        if suffix == '.ini':
            return ConfigFormat.INI
        elif suffix == '.json':
            return ConfigFormat.JSON
        elif suffix in ['.yaml', '.yml']:
            return ConfigFormat.YAML
        else:
            # Try to detect by content
            try:
                with open(self.config_path, 'r') as f:
                    content = f.read()
                    if content.strip().startswith('{'):
                        return ConfigFormat.JSON
                    elif '[' in content:
                        return ConfigFormat.INI
                    else:
                        return ConfigFormat.YAML
            except Exception:
                return ConfigFormat.INI
    
    def _load(self) -> None:
        """Load configuration from file"""
        with self._lock:
            try:
                if not self.config_path.exists():
                    self.logger.warning(f"Config file not found: {self.config_path}")
                    self._load_defaults()
                    return
                
                format_type = self._detect_format()
                
                if format_type == ConfigFormat.INI:
                    self._load_ini()
                elif format_type == ConfigFormat.JSON:
                    self._load_json()
                elif format_type == ConfigFormat.YAML:
                    self._load_yaml()
                
                # **CRITICAL: Enforce hardware mouse control**
                self._enforce_hardware_mouse_config()
                
                # Update file tracking
                self._file_hash = self._get_file_hash()
                self._last_modified = os.path.getmtime(self.config_path)
                
                # Clear caches
                self._clear_caches()
                
                # Validate configuration
                self._validate_config()
                
                # Notify watchers
                self._notify_watchers()
                
                self.logger.info(f"Configuration loaded from {self.config_path}")
                
            except Exception as e:
                self.logger.error(f"Failed to load configuration: {e}", exc_info=True)
                self._load_defaults()
    
    def _enforce_hardware_mouse_config(self) -> None:
        """Ensure configuration enforces hardware-only mouse control"""
        # Add hardware enforcement flags
        if 'Application' not in self._config:
            self._config['Application'] = {}
        
        self._config['Application']['hardware_mouse_only'] = True
        self._config['Application']['software_mouse_disabled'] = True
        self._config['Application']['arduino_required'] = True
        
        self.logger.info("Configuration enforced: HARDWARE MOUSE CONTROL ONLY")
    
    def _load_ini(self) -> None:
        """Load INI configuration"""
        parser = configparser.ConfigParser()
        parser.read(self.config_path)
        
        # Convert to nested dictionary
        self._config = {}
        for section in parser.sections():
            self._config[section] = {}
            for key, value in parser.items(section):
                # Try to parse values
                try:
                    # Check for boolean
                    if value.lower() in ['true', 'false']:
                        self._config[section][key] = value.lower() == 'true'
                    # Check for list
                    elif ',' in value:
                        self._config[section][key] = [
                            self._parse_value(v.strip()) for v in value.split(',')
                        ]
                    # Check for JSON
                    elif value.startswith('{') or value.startswith('['):
                        self._config[section][key] = json.loads(value)
                    else:
                        self._config[section][key] = self._parse_value(value)
                except Exception:
                    self._config[section][key] = value
    
    def _load_json(self) -> None:
        """Load JSON configuration"""
        with open(self.config_path, 'r') as f:
            self._config = json.load(f)
    
    def _load_yaml(self) -> None:
        """Load YAML configuration"""
        if not YAML_AVAILABLE or yaml is None:
            raise RuntimeError("YAML support not available - install PyYAML")
        
        with open(self.config_path, 'r') as f:
            self._config = yaml.safe_load(f)
    
    def _parse_value(self, value: str) -> Union[int, float, str]:
        """Parse string value to appropriate type"""
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value
    
    def _load_defaults(self) -> None:
        """Load default configuration with hardware mouse enforcement"""
        self._config = {
            'Application': {
                'log_level': 'INFO',
                'uses_desktop_coordinates': True,
                'capture_region_width': 280,
                'capture_region_height': 280,
                'capture_region_x_offset': 0,
                'capture_region_y_offset': 0,
                'target_class_id': 0,
                # **CRITICAL: Hardware mouse enforcement**
                'hardware_mouse_only': True,
                'software_mouse_disabled': True,
                'arduino_required': True
            },
            'model_settings': {
                'model_path': 'thebot/src/models/best.pt',
                'confidence': 0.4,
                'device': 'cuda',
                'precision_mode': 'float32'
            },
            'arduino': {
                'arduino_port': 'COM5',
                'baudrate': 115200
            },
            'aim_settings': {
                'sensitivity': 1.0,
                'max_distance': 500,
                'fov_size': 280,
                'aim_height': 0.25,
                'smoothing_factor': 0.3
            }
        }
        
        self.logger.info("Default configuration loaded with HARDWARE MOUSE ENFORCEMENT")
    
    def _clear_caches(self) -> None:
        """Clear cached configurations"""
        self._screen_region = None
        self._model_config = None
        self._arduino_config = None
        self._aim_config = None
    
    def _validate_config(self) -> None:
        """Validate loaded configuration"""
        try:
            # Validate by attempting to create typed configs
            _ = self.get_screen_region()
            _ = self.get_model_config()
            _ = self.get_arduino_config()
            _ = self.get_aim_config()
            
            # **CRITICAL: Validate hardware mouse enforcement**
            app_config = self._config.get('Application', {})
            if not app_config.get('hardware_mouse_only', False):
                self.logger.warning("Forcing hardware mouse control")
                app_config['hardware_mouse_only'] = True
                
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            raise
    
    def _start_file_watcher(self) -> None:
        """Start background thread to watch for config changes"""
        def watch():
            while True:
                try:
                    if self.config_path.exists():
                        current_hash = self._get_file_hash()
                        if current_hash != self._file_hash:
                            self.logger.info("Configuration file changed, reloading...")
                            self.reload()
                except Exception as e:
                    self.logger.error(f"File watcher error: {e}")
                
                threading.Event().wait(1.0)  # Check every second
        
        watcher_thread = threading.Thread(target=watch, daemon=True)
        watcher_thread.start()
    
    def reload(self) -> None:
        """Reload configuration from file"""
        self._load()
    
    def get(self, section: str, key: Optional[str] = None, default: Any = None) -> Any:
        """Get configuration value with dot notation support"""
        with self._lock:
            # Support dot notation
            if '.' in section and key is None:
                parts = section.split('.')
                value = self._config
                for part in parts:
                    if isinstance(value, dict) and part in value:
                        value = value[part]
                    else:
                        return default
                return value
            
            # Traditional section/key access
            if key is None:
                return self._config.get(section, default)
            
            section_data = self._config.get(section, {})
            if isinstance(section_data, dict):
                return section_data.get(key, default)
            return default
    
    def get_screen_region(self) -> ScreenRegion:
        """Get validated screen region configuration"""
        with self._lock:
            if self._screen_region is None:
                app_config = self._config.get('Application', {})
                
                # Check for direct region specification
                if 'screen_region' in self._config:
                    region = self._config['screen_region']
                    self._screen_region = ScreenRegion(
                        left=int(region.get('left', 0)),
                        top=int(region.get('top', 0)),
                        width=int(region.get('width', 280)),
                        height=int(region.get('height', 280))
                    )
                else:
                    # Build from individual settings
                    self._screen_region = ScreenRegion(
                        left=int(app_config.get('capture_region_x_offset', 0)),
                        top=int(app_config.get('capture_region_y_offset', 0)),
                        width=int(app_config.get('capture_region_width', 280)),
                        height=int(app_config.get('capture_region_height', 280))
                    )
            
            return self._screen_region
    
    def get_model_config(self) -> ModelConfig:
        """Get validated model configuration"""
        with self._lock:
            if self._model_config is None:
                model_settings = self._config.get('model_settings', {})
                app_config = self._config.get('Application', {})
                
                self._model_config = ModelConfig(
                    model_path=model_settings.get('model_path', 'thebot/src/models/best.pt'),
                    confidence_threshold=float(model_settings.get('confidence', 0.4)),
                    device=model_settings.get('device', 'cuda'),
                    precision_mode=model_settings.get('precision_mode', 'float32'),
                    warmup_iterations=int(model_settings.get('warmup_iterations', 10)),
                    target_class_id=int(app_config.get('target_class_id', 0)),
                    target_class_name=app_config.get('target_class_name', 'player')
                )
            
            return self._model_config
    
    def get_arduino_config(self) -> ArduinoConfig:
        """Get validated Arduino configuration"""
        with self._lock:
            if self._arduino_config is None:
                arduino_config = self._config.get('arduino', {})
                
                self._arduino_config = ArduinoConfig(
                    port=arduino_config.get('arduino_port', 'COM5'),
                    baudrate=int(arduino_config.get('baudrate', 115200)),
                    timeout=float(arduino_config.get('timeout', 0.1))
                )
            
            return self._arduino_config
    
    def get_aim_config(self) -> AimConfig:
        """Get validated aim configuration"""
        with self._lock:
            if self._aim_config is None:
                aim_settings = self._config.get('aim_settings', {})
                
                self._aim_config = AimConfig(
                    sensitivity=float(aim_settings.get('sensitivity', 1.0)),
                    max_distance=int(aim_settings.get('max_distance', 500)),
                    fov_size=int(aim_settings.get('fov_size', 280)),
                    aim_height_offset=float(aim_settings.get('aim_height', 0.25)),
                    smoothing_factor=float(aim_settings.get('smoothing_factor', 0.3)),
                    kalman_transition_cov=float(aim_settings.get('kalman_transition_cov', 0.01)),
                    kalman_observation_cov=float(aim_settings.get('kalman_observation_cov', 0.01)),
                    delay=float(aim_settings.get('delay', 5e-05))
                )
            
            return self._aim_config
    
    def set(self, section: str, key: str, value: Any) -> None:
        """Set configuration value"""
        with self._lock:
            if section not in self._config:
                self._config[section] = {}
            self._config[section][key] = value
            self._clear_caches()
    
    def save(self, path: Optional[str] = None) -> None:
        """Save configuration to file"""
        save_path = Path(path) if path else self.config_path
        
        with self._lock:
            try:
                format_type = self._detect_format()
                
                if format_type == ConfigFormat.INI:
                    self._save_ini(save_path)
                elif format_type == ConfigFormat.JSON:
                    self._save_json(save_path)
                elif format_type == ConfigFormat.YAML:
                    self._save_yaml(save_path)
                
                self.logger.info(f"Configuration saved to {save_path}")
                
            except Exception as e:
                self.logger.error(f"Failed to save configuration: {e}", exc_info=True)
                raise
    
    def _save_ini(self, path: Path) -> None:
        """Save configuration as INI"""
        parser = configparser.ConfigParser()
        
        for section, values in self._config.items():
            parser[section] = {}
            for key, value in values.items():
                if isinstance(value, (list, dict)):
                    parser[section][key] = json.dumps(value)
                else:
                    parser[section][key] = str(value)
        
        with open(path, 'w') as f:
            parser.write(f)
    
    def _save_json(self, path: Path) -> None:
        """Save configuration as JSON"""
        with open(path, 'w') as f:
            json.dump(self._config, f, indent=2)
    
    def _save_yaml(self, path: Path) -> None:
        """Save configuration as YAML"""
        if not YAML_AVAILABLE or yaml is None:
            raise RuntimeError("YAML support not available - install PyYAML")
        
        with open(path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False)
    
    def register_watcher(self, callback) -> None:
        """Register callback for configuration changes"""
        with self._lock:
            self._watchers.append(callback)
    
    def _notify_watchers(self) -> None:
        """Notify all registered watchers of configuration change"""
        for watcher in self._watchers:
            try:
                watcher(self._config)
            except Exception as e:
                self.logger.error(f"Watcher notification error: {e}")
    
    def as_dict(self) -> Dict[str, Any]:
        """Get full configuration as dictionary"""
        with self._lock:
            return dict(self._config)
    
    def validate_screen_region(self, screen_width: int, screen_height: int) -> bool:
        """Validate screen region against display bounds"""
        region = self.get_screen_region()
        return region.validate_against_screen(screen_width, screen_height)
    
    def is_hardware_mouse_only(self) -> bool:
        """Check if hardware mouse control is enforced"""
        app_config = self._config.get('Application', {})
        return app_config.get('hardware_mouse_only', False)