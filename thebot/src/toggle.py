"""
Runtime Functional Toggle Module
Purpose: Manages the on/off state of the system via hotkeys and mouse events.
- Supports aimbot toggle (F1) and right mouse button for AI targeting.
- Ensures CUDA/NVIDIA GPU is used for detection if available.

Supports configurable hotkeys for aimbot and AI targeting.
Ensures thread-safe operation and proper resource cleanup.
"""
try:
    import keyboard
except ImportError:
    keyboard = None
try:
    import torch
except ImportError:
    torch = None
import threading
import time
from utils import Logger

class ToggleManager:
    """
    Listens for hotkeys and mouse events to toggle system states.
    - F1 toggles aimbot on/off.
    - Right mouse button toggles AI targeting while held.
    - Ensures CUDA is used if available.

    - Configurable hotkey toggles aimbot on/off (default: F1).
    - Configurable key toggles AI targeting while held (default: right mouse).
    - Implements thread-safe state management.
    """


    def __init__(self, on_aimbot_toggle=None, on_targeting_toggle=None, 
                 aimbot_key='f1', targeting_key='right'):
        self.logger = Logger(__name__)
        self._state_lock = threading.Lock()
        self.aimbot_enabled = False
        self.targeting_enabled = False
        self.on_aimbot_toggle = on_aimbot_toggle
        self.on_targeting_toggle = on_targeting_toggle
        self.aimbot_key = aimbot_key
        self.targeting_key = targeting_key
        self._thread = None
        self._running = False


        # CUDA check moved to initialization for clarity
        self.cuda_available = torch and torch.cuda.is_available()
        if self.cuda_available:
            self.logger.info("CUDA is available. Using NVIDIA GPU for detection.")
        else:
            self.logger.warning("CUDA is NOT available. Detection will run on CPU.")

    def _listen(self):
        if not keyboard:
            self.logger.error("keyboard module not available! ToggleManager cannot run.")
            return






        try:
            keyboard.add_hotkey(self.aimbot_key, self._toggle_aimbot)
            keyboard.add_hotkey(self.targeting_key, self._toggle_targeting_down, suppress=False, trigger_on_release=False)
            keyboard.add_hotkey(self.targeting_key, self._toggle_targeting_up, suppress=False, trigger_on_release=True)
        except Exception as e:
            self.logger.error(f"Failed to register hotkeys: {e}")
            return

        while self._running:
            time.sleep(0.05)



        try:
            keyboard.remove_hotkey(self.aimbot_key)
            keyboard.remove_hotkey(self.targeting_key)
        except Exception as e:
            self.logger.warning(f"Failed to remove hotkeys: {e}")

    def _toggle_aimbot(self):





        with self._state_lock:
            self.aimbot_enabled = not self.aimbot_enabled
            if self.on_aimbot_toggle:
                self.on_aimbot_toggle(self.aimbot_enabled)
            state = "ENABLED" if self.aimbot_enabled else "DISABLED"
            self.logger.info(f"Aimbot {state}")

    def _toggle_targeting_down(self):





        with self._state_lock:
            if not self.targeting_enabled:
                self.targeting_enabled = True
                if self.on_targeting_toggle:
                    self.on_targeting_toggle(True)
                self.logger.info("AI Targeting ENABLED (right mouse down)")

    def _toggle_targeting_up(self):





        with self._state_lock:
            if self.targeting_enabled:
                self.targeting_enabled = False
                if self.on_targeting_toggle:
                    self.on_targeting_toggle(False)
                self.logger.info("AI Targeting DISABLED (right mouse up)")

    def start(self):
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._listen, daemon=True)
            self._thread.start()

            self.logger.info(f"ToggleManager started. {self.aimbot_key} toggles aimbot. Hold {self.targeting_key} for targeting.")

    def stop(self):
        if self._running:
            self._running = False
            if self._thread:
                self._thread.join()
            self.logger.info("ToggleManager stopped.")

    def is_aimbot_enabled(self):

        with self._state_lock:
            return self.aimbot_enabled

    def is_targeting_enabled(self):

        with self._state_lock:
            return self.targeting_enabled

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

# Example usage:
# def on_aimbot(state): print("Aimbot:", state)
# def on_targeting(state): print("Targeting:", state)
# tm = ToggleManager(on_aimbot, on_targeting)
# tm.start()