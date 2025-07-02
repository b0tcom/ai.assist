import json
import os
from datetime import datetime
from typing import Optional, Callable
import configparser
import tkinter as tk

def load_config():
    cp = configparser.ConfigParser()
    cp.read("src/configs/config.ini")
    return cp

def save_config(cp):
    with open("src/configs/config.ini", "w") as f:
        cp.write(f)

def update_field(cp, section, key, value):
    if section not in cp:
        cp.add_section(section)
    cp[section][key] = str(value)
    save_config(cp)

class CVTargetingGUI:
    def __init__(self, config=None):
        self.root = tk.Tk()
        self.root.title("AI Aim Assist Configuration")
        self.config = config or {}
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
        row = 0
        for section, fields in self.sections:
            tk.Label(self.root, text=section, font=("Arial", 12, "bold")).grid(row=row, column=0, sticky="w", pady=(10,0))
            row += 1
            for key, label in fields:
                tk.Label(self.root, text=label).grid(row=row, column=0, sticky="w")
                val = self._get_config_value(section, key)
                entry = tk.Entry(self.root, width=40)
                entry.insert(0, str(val) if val is not None else "")
                entry.grid(row=row, column=1, padx=5, pady=2)
                self.entries[(section, key)] = entry
                row += 1
        tk.Button(self.root, text="Save", command=self.save).grid(row=row, column=0, pady=20)
        tk.Button(self.root, text="Close", command=self.root.quit).grid(row=row, column=1, pady=20)

    def _get_config_value(self, section, key):
        # Try to get from config dict, fallback to blank
        if section in self.config and key in self.config[section]:
            return self.config[section][key]
        return ""

    def save(self):
        # Save all entries back to the config dict and file
        for (section, key), entry in self.entries.items():
            val = entry.get()
            if section not in self.config:
                self.config[section] = {}
            self.config[section][key] = val
        # Save to INI file
        cp = configparser.ConfigParser()
        for section in self.config:
            cp[section] = {k: str(v) for k, v in self.config[section].items()}
        with open("src/configs/config.ini", "w") as f:
            cp.write(f)

    def run(self):
        self.root.mainloop()
        if hasattr(self, 'enable_overlay') and self.enable_overlay:
            try:
                from pygame_overlay import PygameOverlay
                region = self.config.get('screen_region', {'width': 960, 'height': 720})
                fov = self.config.get('fov_size', 280)
                overlay = PygameOverlay(width=int(region.get('width', 960)), height=int(region.get('height', 720)), fov=int(fov))
                overlay.run()
            except Exception as e:
                print(f"Failed to launch Pygame overlay: {e}")

    def enable_pygame_overlay(self):
        self.enable_overlay = True
