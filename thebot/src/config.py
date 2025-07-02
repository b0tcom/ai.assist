"""
Configuration Management Module
Purpose: Loads, merges, and provides access to application settings.
"""
import json
import argparse
import os
import yaml  # Add this import
from logger_util import Logger

class Config:
    """Handles application configuration from a file and CLI arguments."""

    def __init__(self, config_path=None, cli_args=None):
        self.logger = Logger(__name__)
        if config_path is None:
            # Prefer ini if present, else fallback to json
            if os.path.exists("configs/config.ini"):
                config_path = "configs/config.ini"
            else:
                config_path = "configs/default_config.json"
        self._config = self._load_from_file(config_path)

        # Load data.yaml for real class name and id
        data_yaml_path = "thebot/src/models/data.yaml"
        if os.path.exists(data_yaml_path):
            with open(data_yaml_path, "r") as f:
                data = yaml.safe_load(f)
                # NOTE: This is the models trained targeting info for detections:
                # names: ['fn - v1 2023-11-26 7-24am']
                # target_class_id: 0
                # classes: ['fn - v1 2023-11-26 7-24am']
                real_name = None
                if "names" in data and isinstance(data["names"], list) and data["names"]:
                    real_name = data["names"][0]
                elif "classes" in data and isinstance(data["classes"], list) and data["classes"]:
                    real_name = data["classes"][0]
                class_id = data.get("target_class_id", 0)
                self._config["target_class_id"] = class_id
                self._config["target_class_real_name"] = real_name
                self._config["target_class_name"] = "player"  # always use "player" internally
        else:
            self.logger.warning(f"data.yaml not found at {data_yaml_path}")

        # Remove toggle_key from config if present
        if "toggle_key" in self._config:
            del self._config["toggle_key"]

        if cli_args:
            self._override_with_cli(cli_args)

    def _load_from_file(self, config_path):
        """Loads a JSON configuration file."""
        try:
            with open(config_path, 'r') as f:
                self.logger.info(f"Loading configuration from {config_path}")
                return json.load(f)
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found at {config_path}. Exiting.")
            raise
        except json.JSONDecodeError:
            self.logger.error(f"Invalid JSON in configuration file: {config_path}. Exiting.")
            raise

    def _override_with_cli(self, cli_args):
        """Overrides file-based settings with any provided CLI arguments."""
        overrides = {key: value for key, value in vars(cli_args).items() if value is not None}
        if overrides:
            self.logger.info(f"Overriding config with CLI arguments: {overrides}")
            cli_to_config_map = {
                'aim_offset': 'aim_height_offset',
                'max_distance': 'max_tracking_distance',
                'confidence_threshold': 'confidence_threshold',
                'model_path': 'model_path'
            }
            for cli_key, value in overrides.items():
                if cli_key in cli_to_config_map:
                    config_key = cli_to_config_map[cli_key]
                    # Enforce model paths if model_path is being set
                    if config_key == 'model_path':
                        self._config[config_key] = "thebot/src/models/best.onnx"
                        self._config['fallback_model_path'] = "thebot/src/models/yolo/yolov8n.pt"
                    else:
                        self._config[config_key] = value

    def get(self, key, default=None):
        """Retrieves a configuration value by key."""
        return self._config.get(key, default)

    def __getitem__(self, key):
        return self._config[key]

    def __contains__(self, key):
        return key in self._config

    def to_dict(self):
        return dict(self._config)