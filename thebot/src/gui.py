import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import os
from datetime import datetime
from typing import Optional, Callable
import configparser

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
    def __init__(self, root, config=None):
        self.root = root
        self.root.title("CV Targeting Control Panel")
        self.root.geometry("1100x850")
        self.root.configure(bg='#1a1a1a')

        # Theme colors
        self.bg_color = '#1a1a1a'
        self.fg_color = '#ffffff'
        self.red_color = '#ff0000'
        self.dark_red = '#8b0000'
        self.grey_color = '#333333'

        # Settings variables
        self.settings = {
            'confidence_threshold': tk.DoubleVar(value=0.5),
            'iou_threshold': tk.DoubleVar(value=0.45),
            'sensitivity': tk.DoubleVar(value=1.0),
            'max_distance': tk.IntVar(value=500),
            'fov_size': tk.DoubleVar(value=280.0),
            'aim_height': tk.DoubleVar(value=0.25),
            'show_fov_overlay': tk.BooleanVar(value=True),
            'show_eliuth_vision': tk.BooleanVar(value=False),
            'kalman_transition_cov': tk.DoubleVar(value=0.01),
            'kalman_observation_cov': tk.DoubleVar(value=0.01),
            'enable_system': tk.BooleanVar(value=False),
            'anti_recoil_enabled': tk.BooleanVar(value=True),
            'anti_recoil_vertical_strength': tk.DoubleVar(value=0.5),
            'anti_recoil_horizontal_strength': tk.DoubleVar(value=0.2),
            'anti_recoil_pattern_enabled': tk.BooleanVar(value=False),
            'rapid_fire_enabled': tk.BooleanVar(value=False),
            'rapid_fire_rate': tk.DoubleVar(value=0.1),
            'hip_mode_enabled': tk.BooleanVar(value=True),
            'activation_key': tk.StringVar(value="f1"),
            'deactivation_key': tk.StringVar(value="f2"),
            'right_click_button': tk.StringVar(value="0x02"),
            'right_click_sensitivity': tk.DoubleVar(value=1.0),
            'left_click_button': tk.StringVar(value="0x01"),
            'left_click_sensitivity': tk.DoubleVar(value=0.5),
            'shooting_height_head': tk.DoubleVar(value=0.15),
            'shooting_height_neck': tk.DoubleVar(value=0.25),
            'shooting_height_chest': tk.DoubleVar(value=0.35),
            'altura_tiro': tk.DoubleVar(value=1.5),
            'delay': tk.DoubleVar(value=5e-05),
        }
        self.arduino_status = tk.StringVar(value="Unknown")
        self.show_capture_window = tk.BooleanVar(value=False)
        self.show_popups = tk.BooleanVar(value=False)
        self.on_capture_window_toggle: Optional[Callable[[bool], None]] = None

        # Display tab settings
        self.settings.update({
            'show_overlay': tk.BooleanVar(value=True),
            'show_crosshair': tk.BooleanVar(value=True),
            'show_fps': tk.BooleanVar(value=True),
            'show_popups': self.show_popups,
        })

        # Load config if provided
        if config:
            self.init_from_config(config)

        self.current_profile = "default"
        self.profiles_dir = "profiles"
        os.makedirs(self.profiles_dir, exist_ok=True)

        # --- Ensure logs tab (and self.log_text) is created before any log() call ---
        self._build_gui_pre_logs()
        self._build_gui_post_logs()

    def _build_gui_pre_logs(self):
        # Main frame and notebook
        main_frame = tk.Frame(self.root, bg=self.bg_color)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        title_label = tk.Label(main_frame, text="CV TARGETING SYSTEM",
                               font=('Arial', 26, 'bold'),
                               fg=self.red_color, bg=self.bg_color)
        title_label.pack(pady=(0, 20))

        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        style = ttk.Style()
        style.theme_use('default')
        style.configure('TNotebook', background=self.bg_color, borderwidth=0)
        style.configure('TNotebook.Tab', background=self.grey_color,
                        foreground=self.fg_color, padding=[20, 10])
        style.map('TNotebook.Tab', background=[('selected', self.dark_red)])

        # Create logs tab first so self.log_text exists before any log() call
        self.create_logs_tab()

    def _build_gui_post_logs(self):
        # Add the rest of the tabs after logs
        self.create_general_tab()
        self.create_display_tab()
        self.create_hotkeys_tab()

        # Status bar
        status_frame = tk.Frame(self.root, bg=self.bg_color)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        tk.Label(status_frame, text="Arduino Status:", fg=self.fg_color, bg=self.bg_color).pack(side=tk.LEFT, padx=10)
        tk.Label(status_frame, textvariable=self.arduino_status, fg='#00ff00', bg=self.bg_color).pack(side=tk.LEFT)

    def init_from_config(self, config):
        for key, var in self.settings.items():
            if key in config:
                try:
                    var.set(config[key])
                except Exception:
                    pass

    def create_slider(self, parent, label, variable, from_val, to_val, resolution):
        frame = tk.Frame(parent, bg=self.bg_color)
        frame.pack(fill=tk.X, pady=5)
        label_widget = tk.Label(frame, text=label, fg=self.fg_color,
                                bg=self.bg_color, width=22, anchor='w')
        label_widget.pack(side=tk.LEFT)
        slider = tk.Scale(frame, from_=from_val, to=to_val,
                          resolution=resolution, orient=tk.HORIZONTAL,
                          variable=variable, bg=self.grey_color,
                          fg=self.fg_color, highlightthickness=0,
                          troughcolor='#555555', activebackground=self.red_color,
                          length=320)
        slider.pack(side=tk.LEFT, padx=10)
        value_label = tk.Label(frame, textvariable=variable,
                               fg=self.red_color, bg=self.bg_color, width=10)
        value_label.pack(side=tk.LEFT)

    def create_general_tab(self):
        general_frame = tk.Frame(self.notebook, bg=self.bg_color)
        self.notebook.add(general_frame, text="General")

        # Profile section
        profile_frame = tk.Frame(general_frame, bg=self.bg_color)
        profile_frame.pack(fill=tk.X, padx=20, pady=10)
        tk.Label(profile_frame, text="Select Profile:",
                 fg=self.fg_color, bg=self.bg_color).pack(side=tk.LEFT, padx=5)
        self.profile_var = tk.StringVar(value="default")
        self.profile_menu = ttk.Combobox(profile_frame, textvariable=self.profile_var,
                                         values=["default"], state="readonly", width=15)
        self.profile_menu.pack(side=tk.LEFT, padx=5)
        button_style = {'bg': self.dark_red, 'fg': self.fg_color,
                        'activebackground': self.red_color, 'bd': 0, 'padx': 15}
        tk.Button(profile_frame, text="Load", command=self.load_selected_profile,
                  **button_style).pack(side=tk.LEFT, padx=5)
        tk.Button(profile_frame, text="Save", command=self.save_current_profile,
                  **button_style).pack(side=tk.LEFT, padx=5)
        tk.Button(profile_frame, text="Import", command=self.import_profile,
                  **button_style).pack(side=tk.LEFT, padx=5)
        tk.Button(profile_frame, text="Export", command=self.export_profile,
                  **button_style).pack(side=tk.LEFT, padx=5)

        # Settings section
        settings_frame = tk.Frame(general_frame, bg=self.bg_color)
        settings_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        # Sliders for all main settings
        self.create_slider(settings_frame, "Confidence Threshold:",
                           self.settings['confidence_threshold'], 0, 1, 0.01)
        self.create_slider(settings_frame, "IOU Threshold:",
                           self.settings['iou_threshold'], 0, 1, 0.01)
        self.create_slider(settings_frame, "Sensitivity:",
                           self.settings['sensitivity'], 0, 1, 0.01)
        self.create_slider(settings_frame, "Max Distance:",
                           self.settings['max_distance'], 0, 1000, 1)
        self.create_slider(settings_frame, "FOV Size:",
                           self.settings['fov_size'], 50, 500, 1)
        self.create_slider(settings_frame, "Aim Height:",
                           self.settings['aim_height'], 0, 1, 0.01)
        self.create_slider(settings_frame, "Delay (s):",
                           self.settings['delay'], 0, 0.01, 0.00001)
        self.create_slider(settings_frame, "Kalman Transition Cov:",
                           self.settings['kalman_transition_cov'], 0, 1, 0.001)
        self.create_slider(settings_frame, "Kalman Observation Cov:",
                           self.settings['kalman_observation_cov'], 0, 1, 0.001)
        self.create_slider(settings_frame, "Shooting Height Head:",
                           self.settings['shooting_height_head'], 0, 1, 0.01)
        self.create_slider(settings_frame, "Shooting Height Neck:",
                           self.settings['shooting_height_neck'], 0, 1, 0.01)
        self.create_slider(settings_frame, "Shooting Height Chest:",
                           self.settings['shooting_height_chest'], 0, 1, 0.01)
        self.create_slider(settings_frame, "Altura Tiro:",
                           self.settings['altura_tiro'], 0, 3, 0.01)

        # Enable/Disable button
        self.toggle_button = tk.Button(general_frame, text="ENABLE SYSTEM",
                                       command=self.toggle_system,
                                       bg=self.dark_red, fg=self.fg_color,
                                       font=('Arial', 14, 'bold'),
                                       activebackground=self.red_color,
                                       bd=0, pady=10)
        self.toggle_button.pack(fill=tk.X, padx=20, pady=20)

        # Checkboxes for overlays and modes
        tk.Checkbutton(settings_frame, text="Show FOV Overlay", variable=self.settings['show_fov_overlay'],
                       fg=self.fg_color, bg=self.bg_color, selectcolor=self.bg_color).pack(anchor='w', pady=2)
        tk.Checkbutton(settings_frame, text="Show Eliuth Vision", variable=self.settings['show_eliuth_vision'],
                       fg=self.fg_color, bg=self.bg_color, selectcolor=self.bg_color).pack(anchor='w', pady=2)
        tk.Checkbutton(settings_frame, text="Anti-Recoil Enabled", variable=self.settings['anti_recoil_enabled'],
                       fg=self.fg_color, bg=self.bg_color, selectcolor=self.bg_color).pack(anchor='w', pady=2)
        tk.Checkbutton(settings_frame, text="Anti-Recoil Pattern", variable=self.settings['anti_recoil_pattern_enabled'],
                       fg=self.fg_color, bg=self.bg_color, selectcolor=self.bg_color).pack(anchor='w', pady=2)
        tk.Checkbutton(settings_frame, text="Rapid Fire Enabled", variable=self.settings['rapid_fire_enabled'],
                       fg=self.fg_color, bg=self.bg_color, selectcolor=self.bg_color).pack(anchor='w', pady=2)
        tk.Checkbutton(settings_frame, text="Hip Mode Enabled", variable=self.settings['hip_mode_enabled'],
                       fg=self.fg_color, bg=self.bg_color, selectcolor=self.bg_color).pack(anchor='w', pady=2)

        # Add checkbox for capture window
        self.capture_checkbox = tk.Checkbutton(
            settings_frame,
            text="Show Capture Window (Live Detection Preview)",
            variable=self.show_capture_window,
            fg=self.fg_color, bg=self.bg_color, selectcolor=self.bg_color,
            command=self._on_capture_window_toggle
        )
        self.capture_checkbox.pack(anchor='w', pady=4)
        self._update_capture_window_state()

    def create_display_tab(self):
        display_frame = tk.Frame(self.notebook, bg=self.bg_color)
        self.notebook.add(display_frame, text="Display")

        options_frame = tk.Frame(display_frame, bg=self.bg_color)
        options_frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)

        tk.Label(options_frame, text="Display Settings",
                 font=('Arial', 16, 'bold'),
                 fg=self.red_color, bg=self.bg_color).pack(pady=10)

        tk.Checkbutton(
            options_frame,
            text="Enable Pop-up Displays (Overlay, Crosshair, etc)",
            variable=self.show_popups,
            fg=self.fg_color, bg=self.bg_color, selectcolor=self.bg_color,
            command=self._on_popups_toggle
        ).pack(anchor='w', pady=5)

        self.overlay_checkbox = tk.Checkbutton(
            options_frame, text="Show Overlay",
            variable=self.settings['show_overlay'],
            fg=self.fg_color, bg=self.bg_color,
            selectcolor=self.bg_color
        )
        self.overlay_checkbox.pack(anchor='w', pady=5)

        self.crosshair_checkbox = tk.Checkbutton(
            options_frame, text="Show Crosshair",
            variable=self.settings['show_crosshair'],
            fg=self.fg_color, bg=self.bg_color,
            selectcolor=self.bg_color
        )
        self.crosshair_checkbox.pack(anchor='w', pady=5)

        self.fps_checkbox = tk.Checkbutton(
            options_frame, text="Show FPS Counter",
            variable=self.settings['show_fps'],
            fg=self.fg_color, bg=self.bg_color,
            selectcolor=self.bg_color
        )
        self.fps_checkbox.pack(anchor='w', pady=5)

        self.info_label = tk.Label(
            options_frame,
            text="Note: Overlay and popup displays are currently DISABLED for safety",
            fg='#888888', bg=self.bg_color
        )
        self.info_label.pack(pady=20)
        self._update_popups_state()

    def create_logs_tab(self):
        logs_frame = tk.Frame(self.notebook, bg=self.bg_color)
        self.notebook.add(logs_frame, text="Logs")
        self.log_text = tk.Text(logs_frame, bg='#0a0a0a', fg='#00ff00',
                                insertbackground='#00ff00', font=('Consolas', 10))
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.log("System initialized")
        self.log("GUI started successfully")

    def create_hotkeys_tab(self):
        hotkeys_frame = tk.Frame(self.notebook, bg=self.bg_color)
        self.notebook.add(hotkeys_frame, text="Hotkeys")
        tk.Label(hotkeys_frame, text="Hotkey Configuration",
                 font=('Arial', 16, 'bold'),
                 fg=self.red_color, bg=self.bg_color).pack(pady=20)
        hotkeys = [
            ("Toggle System", "F1"),
            ("Reload Config", "F5"),
            ("Panic Stop", "F9"),
            ("Increase Sensitivity", "Page Up"),
            ("Decrease Sensitivity", "Page Down"),
        ]
        for name, key in hotkeys:
            frame = tk.Frame(hotkeys_frame, bg=self.bg_color)
            frame.pack(pady=5)
            tk.Label(frame, text=f"{name}:", fg=self.fg_color,
                     bg=self.bg_color, width=22, anchor='w').pack(side=tk.LEFT)
            tk.Label(frame, text=key, fg=self.red_color,
                     bg=self.bg_color, font=('Arial', 10, 'bold')).pack(side=tk.LEFT)

    def _on_popups_toggle(self):
        self._update_popups_state()

    def _update_popups_state(self):
        state = tk.NORMAL if self.show_popups.get() else tk.DISABLED
        self.overlay_checkbox.config(state=state)
        self.crosshair_checkbox.config(state=state)
        self.fps_checkbox.config(state=state)
        if self.show_popups.get():
            self.info_label.config(
                text="Note: Overlay and popup displays are ENABLED. Use with caution."
            )
        else:
            self.info_label.config(
                text="Note: Overlay and popup displays are currently DISABLED for safety"
            )

    def toggle_system(self):
        self.settings['enable_system'].set(not self.settings['enable_system'].get())
        if self.settings['enable_system'].get():
            self.toggle_button.config(text="DISABLE SYSTEM", bg='#00ff00')
            self.log("System ENABLED")
        else:
            self.toggle_button.config(text="ENABLE SYSTEM", bg=self.dark_red)
            self.log("System DISABLED")

    def load_profile(self, profile_name):
        profile_path = os.path.join(self.profiles_dir, f"{profile_name}.json")
        if os.path.exists(profile_path):
            try:
                with open(profile_path, 'r') as f:
                    data = json.load(f)
                for key, var in self.settings.items():
                    if key in data:
                        var.set(data[key])
                self.log(f"Loaded profile: {profile_name}")
            except Exception as e:
                self.log(f"Error loading profile: {e}")
        else:
            self.save_profile(profile_name)

    def save_profile(self, profile_name):
        profile_path = os.path.join(self.profiles_dir, f"{profile_name}.json")
        data = {key: var.get() for key, var in self.settings.items()}
        with open(profile_path, 'w') as f:
            json.dump(data, f, indent=2)
        self.log(f"Saved profile: {profile_name}")
        self.update_profile_list()

    def load_selected_profile(self):
        self.load_profile(self.profile_var.get())

    def save_current_profile(self):
        self.save_profile(self.profile_var.get())

    def import_profile(self):
        filename = filedialog.askopenfilename(
            title="Import Profile",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            import shutil
            profile_name = os.path.splitext(os.path.basename(filename))[0]
            shutil.copy(filename, os.path.join(self.profiles_dir, f"{profile_name}.json"))
            self.update_profile_list()
            self.profile_var.set(profile_name)
            self.load_profile(profile_name)

    def export_profile(self):
        filename = filedialog.asksaveasfilename(
            title="Export Profile",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            self.save_profile(os.path.splitext(os.path.basename(filename))[0])

    def update_profile_list(self):
        profiles = [f[:-5] for f in os.listdir(self.profiles_dir) if f.endswith('.json')]
        self.profile_menu['values'] = profiles

    def log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        if "Arduino connected" in message:
            self.arduino_status.set("Connected")
        elif "Arduino connection failed" in message or "could not be opened" in message:
            self.arduino_status.set("Not Connected")

    def get_settings(self):
        return {key: var.get() for key, var in self.settings.items()}

    def _on_capture_window_toggle(self):
        self._update_capture_window_state()
        if self.on_capture_window_toggle:
            self.on_capture_window_toggle(self.show_capture_window.get())

    def _update_capture_window_state(self):
        state = tk.NORMAL if self.show_popups.get() else tk.DISABLED
        self.capture_checkbox.config(state=state)
        if not self.show_popups.get():
            self.log("Pop-up displays are disabled; capture window cannot be shown.")

    # CVTargetingGUI is imported and used in main.py as intended.
    # It is initialized with config and provides the GUI interface.
