import configparser
import json
import os
from logger_util import Logger

class ConfigManager:
    """
    Dedicated configuration manager for loading, validating, and providing access to config values.
    Supports INI and JSON config files. Handles region calculation and validation.
    """
    def __init__(self, config_path="configs/config.ini"):
        self.logger = Logger(__name__)
        self.config_path = config_path
        self.config = None
        self.region = None
        self._load()

    def _load(self):
        if self.config_path.endswith(".ini"):
            self.config = configparser.ConfigParser()
            if not os.path.exists(self.config_path):
                self.logger.warning(f"INI config not found at {self.config_path}, falling back to default_config.json")
                self.config_path = "configs/default_config.json"
                self._load_json()
            else:
                self.config.read(self.config_path)
        elif self.config_path.endswith(".json"):
            self._load_json()
        else:
            raise ValueError(f"Unsupported config file type: {self.config_path}")

    def _load_json(self):
        with open(self.config_path, "r") as f:
            self.config = json.load(f)

    def get(self, section, key=None, default=None):
        if isinstance(self.config, configparser.ConfigParser):
            if key is None:
                return dict(self.config[section]) if section in self.config else default
            return self.config[section].get(key, default) if section in self.config else default
        elif isinstance(self.config, dict):
            if key is None:
                return self.config.get(section, default)
            return self.config.get(section, {}).get(key, default)
        return default

    def get_screen_region(self):
        """
        Extracts and returns the screen region as a dict: {left, top, width, height}
        Handles both INI and JSON config formats.
        """
        if isinstance(self.config, configparser.ConfigParser):
            app = self.config["Application"] if "Application" in self.config else {}
            left = int(app.get("capture_region_x_offset", 0))
            top = int(app.get("capture_region_y_offset", 0))
            width = int(app.get("capture_region_width", 0))
            height = int(app.get("capture_region_height", 0))
        elif isinstance(self.config, dict):
            region = self.config.get("screen_region")
            if region:
                left = int(region.get("left", 0))
                top = int(region.get("top", 0))
                width = int(region.get("width", 0))
                height = int(region.get("height", 0))
            else:
                left = top = 0
                width = height = 280
        else:
            left = top = 0
            width = height = 280
        self.region = {"left": left, "top": top, "width": width, "height": height}
        return self.region

    def validate_region(self, screen_width, screen_height):
        """
        Validates that the configured region fits within the given screen dimensions.
        Returns True if valid, False otherwise.
        """
        region = self.get_screen_region()
        valid = (0 <= region["left"] < screen_width and
                 0 <= region["top"] < screen_height and
                 region["left"] + region["width"] <= screen_width and
                 region["top"] + region["height"] <= screen_height)
        if not valid:
            self.logger.error(f"Configured region {region} exceeds display bounds {screen_width}x{screen_height}")
        return valid

    def reload(self):
        self._load()

    def as_dict(self):
        # Always return a plain dict with all values as native Python types
        if isinstance(self.config, configparser.ConfigParser):
            def parse_section(section):
                d = {}
                if self.config is not None and section in self.config:
                    for k, v in self.config[section].items():
                        # Try to convert to int or float if possible
                        try:
                            if "," in v:
                                # List of ints or floats
                                vals = v.split(",")
                                d[k] = [int(x) if x.strip().isdigit() else x for x in vals]
                            elif v.isdigit():
                                d[k] = int(v)
                            else:
                                try:
                                    d[k] = float(v)
                                except Exception:
                                    d[k] = v
                        except Exception:
                            d[k] = v
                return d
            return {s: parse_section(s) for s in self.config.sections()} if self.config is not None else {}
        elif isinstance(self.config, dict):
            return dict(self.config)
        return {}

    def __enter__(self):
        # Context manager entry: reload config for freshness
        self.reload()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # No persistent resources to clean up, but could flush or log
        self.logger.info("ConfigManager context exited.")
        return False  # Do not suppress exceptions
