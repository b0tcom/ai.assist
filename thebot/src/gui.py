import json
import os
from datetime import datetime
from typing import Optional, Callable
from pathlib import Path
import configparser
import tkinter as tk
from tkinter import ttk, messagebox

# Import modular configuration
try:
    from .config_manager import ConfigManager
    from .logger_util import get_logger
    MODULAR_CONFIG_AVAILABLE = True
except ImportError:
    MODULAR_CONFIG_AVAILABLE = False

def load_config(config_path: str = "configs/config.ini"):
    """Load configuration with flexible path handling"""
    if MODULAR_CONFIG_AVAILABLE:
        return ConfigManager(config_path)
    else:
        # Fallback to legacy configparser
        cp = configparser.ConfigParser()
        # Try multiple possible paths
        possible_paths = [
            config_path,
            f"src/{config_path}",
            f"thebot/src/{config_path}",
            Path(__file__).parent.parent / config_path
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                cp.read(path)
                return cp
        
        # If no config found, create empty one
        return configparser.ConfigParser()

def save_config(cp, config_path: str = "configs/config.ini"):
    """Save configuration with flexible path handling"""
    # Try multiple possible paths
    possible_paths = [
        config_path,
        f"src/{config_path}",
        f"thebot/src/{config_path}",
        Path(__file__).parent.parent / config_path
    ]
    
    # Find existing file or use first path
    save_path = config_path
    for path in possible_paths:
        if os.path.exists(path):
            save_path = path
            break
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, "w") as f:
        cp.write(f)

def update_field(cp, section, key, value):
    """Update configuration field with support for both ConfigManager and ConfigParser"""
    if MODULAR_CONFIG_AVAILABLE and hasattr(cp, 'set'):
        # Using modular ConfigManager
        cp.set(section, key, value)
    else:
        # Using legacy ConfigParser
        if section not in cp:
            cp.add_section(section)
        cp[section][key] = str(value)
        save_config(cp)

class CVTargetingGUI:
    def __init__(self, config=None):
        self.root = tk.Tk()
        self.root.title("AI Aim Assist Configuration")
        self.config = config or load_config()
        self.entries = {}
        self.sections = [
            ("Application", [
                ("log_level", "Log Level"),
                ("uses_desktop_coordinates", "Use Desktop Coordinates"),
                ("capture_region_width", "Capture Region Width"),
                ("capture_region_height", "Capture Region Height"),
                ("capture_region_x_offset", "Capture Region X Offset"),
                ("capture_region_y_offset", "Capture Region Y Offset"),
                ("target_class_id", "Target Class ID"),
                ("input_method", "Input Method"),
                ("aim_activation_key", "Aim Activation Key"),
            ]),
            ("Detection", [
                ("confidence_threshold", "Confidence Threshold"),
                ("iou_threshold", "IOU Threshold"),
            ]),
            ("Targeting", [
                ("aim_height_offset", "Aim Height Offset"),
                ("max_tracking_distance", "Max Tracking Distance"),
                ("smoothing_factor", "Smoothing Factor"),
                ("prediction_factor", "Prediction Factor"),
            ]),
            ("AntiRecoil", [
                ("enable_anti_recoil", "Enable Anti-Recoil"),
                ("recoil_pattern_x", "Recoil Pattern X"),
                ("recoil_pattern_y", "Recoil Pattern Y"),
                ("recoil_shots_to_compensate", "Recoil Shots To Compensate"),
                ("recoil_compensation_delay", "Recoil Compensation Delay"),
            ]),
            ("model_settings", [
                ("model_path", "Model Path"),
                ("confidence", "Model Confidence"),
                ("device", "Device"),
                ("fallback_model_path", "Fallback Model Path"),
                ("precision_mode", "Precision Mode"),
                ("warmup_iterations", "Warmup Iterations"),
                ("target_priority", "Target Priority"),
                ("detection_mode", "Detection Mode"),
                ("min_player_size", "Min Player Size"),
                ("max_player_size", "Max Player Size"),
                ("ethical_mode", "Ethical Mode"),
            ]),
            ("arduino", [
                ("arduino_port", "Arduino Port"),
            ]),
            ("aim_settings", [
                ("sensitivity", "Sensitivity"),
                ("max_distance", "Max Distance"),
                ("fov_size", "FOV Size"),
                ("shooting_height_ratios", "Shooting Height Ratios"),
                ("altura_tiro", "Altura Tiro"),
                ("target_areas", "Target Areas"),
                ("smoothing_factor", "Smoothing Factor"),
                ("kalman_transition_cov", "Kalman Transition Covariance"),
                ("kalman_observation_cov", "Kalman Observation Covariance"),
                ("delay", "Delay"),
                ("aim_height", "Aim Height"),
            ]),
            ("anti_recoil", [
                ("enabled", "Anti-Recoil Enabled"),
                ("vertical_strength", "Vertical Strength"),
                ("horizontal_strength", "Horizontal Strength"),
                ("pattern_enabled", "Pattern Enabled"),
            ]),
            ("rapid_fire", [
                ("enabled", "Rapid Fire Enabled"),
                ("fire_rate", "Fire Rate"),
            ]),
            ("mode_settings", [
                ("right_click", "Right Click Settings"),
                ("left_click", "Left Click Settings"),
            ]),
            ("key_bindings", [
                ("activation_key", "Activation Key"),
                ("deactivation_key", "Deactivation Key"),
            ]),
            ("hip_mode_enabled", [
                ("enabled", "Hip Mode Enabled"),
            ]),
        ]
        self._build_gui()

    def _get_config_value(self, section, key):
        """Get configuration value with support for both ConfigManager and ConfigParser"""
        try:
            if MODULAR_CONFIG_AVAILABLE and hasattr(self.config, 'get') and hasattr(self.config, '_config'):
                # Using modular ConfigManager
                return self.config.get(section, key)
            else:
                # Using legacy ConfigParser
                try:
                    return self.config.get(section, key)
                except (configparser.NoSectionError, configparser.NoOptionError):
                    return ""
        except Exception:
            return ""

    def _build_gui(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True)
        for section, fields in self.sections:
            frame = ttk.Frame(notebook)
            notebook.add(frame, text=section)
            row = 0
            for key, label in fields:
                tk.Label(frame, text=label).grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
                entry = tk.Entry(frame, width=40)
                entry.grid(row=row, column=1, padx=5, pady=2)
                entry.insert(0, self._get_config_value(section, key))
                self.entries[(section, key)] = entry
                row += 1
        save_btn = tk.Button(self.root, text="Save", command=self.save)
        save_btn.pack(pady=10)

    def save(self):
        for (section, key), entry in self.entries.items():
            value = entry.get()
            update_field(self.config, section, key, value)
        messagebox.showinfo("Saved", "Configuration saved successfully.")

    def run(self):
        self.root.mainloop()

    def enable_pygame_overlay(self):
        # Placeholder for overlay integration
        pass
