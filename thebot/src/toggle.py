
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


class HotkeyCircuitBreaker:
    """Circuit breaker pattern for hotkey registration failures."""
    
    def __init__(self, failure_threshold=3, recovery_timeout=10):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.last_failure_time = None
        self.recovery_timeout = recovery_timeout
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def can_attempt(self):
        """Check if we can attempt hotkey registration."""
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def record_success(self):
        """Record successful hotkey registration."""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def record_failure(self):
        """Record failed hotkey registration."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


class StateManager:
    """Thread-safe state management for toggle states."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._aimbot_enabled = False
        self._targeting_enabled = False
        self._callbacks = {
            'aimbot': [],
            'targeting': []
        }
    
    def register_callback(self, event_type, callback):
        """Register a callback for state changes."""
        if event_type in self._callbacks:
            self._callbacks[event_type].append(callback)
    
    def set_aimbot_enabled(self, enabled):
        """Set aimbot state and notify callbacks."""
        with self._lock:
            old_state = self._aimbot_enabled
            self._aimbot_enabled = enabled
            if old_state != enabled:
                self._notify_callbacks('aimbot', enabled)
    
    def set_targeting_enabled(self, enabled):
        """Set targeting state and notify callbacks."""
        with self._lock:
            old_state = self._targeting_enabled
            self._targeting_enabled = enabled
            if old_state != enabled:
                self._notify_callbacks('targeting', enabled)
    
    def is_aimbot_enabled(self):
        """Get current aimbot state."""
        with self._lock:
            return self._aimbot_enabled
    
    def is_targeting_enabled(self):
        """Get current targeting state."""
        with self._lock:
            return self._targeting_enabled
    
    def _notify_callbacks(self, event_type, state):
        """Notify all registered callbacks."""
        for callback in self._callbacks[event_type]:
            try:
                callback(state)
            except Exception as e:
                # Log error but don't let callback failures break the system
                Logger(__name__).error(f"Callback error for {event_type}: {e}")


class HotkeyHandler:
    """Handles hotkey registration and management."""
    
    def __init__(self, state_manager, logger, aimbot_key='f1', targeting_key='right'):
        self.state_manager = state_manager
        self.logger = logger
        self.aimbot_key = aimbot_key
        self.targeting_key = targeting_key
        self.circuit_breaker = HotkeyCircuitBreaker()
        self._registered_hotkeys = []
    
    def register_hotkeys(self):
        """Register hotkeys with circuit breaker protection."""
        if not keyboard:
            self.logger.error("keyboard module not available! Cannot register hotkeys.")
            return False
        
        if not self.circuit_breaker.can_attempt():
            self.logger.warning("Circuit breaker open - skipping hotkey registration")
            return False
        
        try:
            # Register aimbot toggle
            keyboard.add_hotkey(self.aimbot_key, self._toggle_aimbot)
            self._registered_hotkeys.append(self.aimbot_key)
            
            # Register targeting down/up
            keyboard.add_hotkey(self.targeting_key,
                              self._toggle_targeting_down,
                              suppress=False,
                              trigger_on_release=False)
            keyboard.add_hotkey(self.targeting_key,
                              self._toggle_targeting_up,
                              suppress=False,
                              trigger_on_release=True)
            self._registered_hotkeys.extend([self.targeting_key, self.targeting_key])
            
            self.circuit_breaker.record_success()
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register hotkeys: {e}")
            self.circuit_breaker.record_failure()
            return False
    
    def unregister_hotkeys(self):
        """Unregister all hotkeys safely."""
        if not keyboard:
            return
        
        for hotkey in self._registered_hotkeys:
            try:
                keyboard.remove_hotkey(hotkey)
            except Exception as e:
                self.logger.warning(f"Failed to remove hotkey {hotkey}: {e}")
        
        self._registered_hotkeys.clear()
    
    def _toggle_aimbot(self):
        """Toggle aimbot state."""
        current_state = self.state_manager.is_aimbot_enabled()
        new_state = not current_state
        self.state_manager.set_aimbot_enabled(new_state)
        state_text = "ENABLED" if new_state else "DISABLED"
        self.logger.info(f"Aimbot {state_text}")
    
    def _toggle_targeting_down(self):
        """Enable targeting on key down."""
        if not self.state_manager.is_targeting_enabled():
            self.state_manager.set_targeting_enabled(True)
            self.logger.info("AI Targeting ENABLED (right mouse down)")
    
    def _toggle_targeting_up(self):
        """Disable targeting on key up."""
        if self.state_manager.is_targeting_enabled():
            self.state_manager.set_targeting_enabled(False)
            self.logger.info("AI Targeting DISABLED (right mouse up)")


class ToggleManager:
    """
    Main toggle manager with improved resource management and error handling.
    - Configurable hotkey toggles aimbot on/off (default: F1).
    - Configurable key toggles AI targeting while held (default: right mouse).
    - Implements thread-safe state management and proper cleanup.
    """

    def __init__(self,
                 on_aimbot_toggle=None,
                 on_targeting_toggle=None,
                 aimbot_key='f1',
                 targeting_key='right'):
        self.logger = Logger(__name__)
        self.state_manager = StateManager()
        self.hotkey_handler = HotkeyHandler(
            self.state_manager, self.logger, aimbot_key, targeting_key
        )
        self._thread = None
        self._running = False
        self._shutdown_event = threading.Event()

        # Register legacy callbacks if provided
        if on_aimbot_toggle:
            self.state_manager.register_callback('aimbot', on_aimbot_toggle)
        if on_targeting_toggle:
            self.state_manager.register_callback('targeting', on_targeting_toggle)

        # CUDA check moved to initialization for clarity
        self.cuda_available = torch and torch.cuda.is_available()
        if self.cuda_available:
            self.logger.info("CUDA is available. Using NVIDIA GPU for detection.")
        else:
            self.logger.warning("CUDA is NOT available. Detection will run on CPU.")

    def _listen(self):
        """Main listening loop with improved error handling."""
        if not self.hotkey_handler.register_hotkeys():
            self.logger.error("Failed to register hotkeys - ToggleManager cannot function")
            return

        self.logger.info(
            f"ToggleManager started. {self.hotkey_handler.aimbot_key} toggles aimbot. "
            f"Hold {self.hotkey_handler.targeting_key} for targeting."
        )

        # Main loop with shutdown event
        while self._running and not self._shutdown_event.is_set():
            try:
                self._shutdown_event.wait(timeout=0.1)
            except Exception as e:
                self.logger.error(f"Error in toggle listen loop: {e}")
                time.sleep(0.1)

        # Cleanup hotkeys
        self.hotkey_handler.unregister_hotkeys()

    def start(self):
        """Start the toggle manager."""
        if self._running:
            self.logger.warning("ToggleManager already running")
            return

        self._running = True
        self._shutdown_event.clear()
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the toggle manager with proper cleanup."""
        if not self._running:
            return

        self._running = False
        self._shutdown_event.set()
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                self.logger.warning("Toggle thread did not stop gracefully")

        self.logger.info("ToggleManager stopped.")

    def is_aimbot_enabled(self):
        """Get current aimbot state."""
        return self.state_manager.is_aimbot_enabled()

    def is_targeting_enabled(self):
        """Get current targeting state."""
        return self.state_manager.is_targeting_enabled()

    def register_aimbot_callback(self, callback):
        """Register callback for aimbot state changes."""
        self.state_manager.register_callback('aimbot', callback)

    def register_targeting_callback(self, callback):
        """Register callback for targeting state changes."""
        self.state_manager.register_callback('targeting', callback)

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.stop()


# Example usage:
# def on_aimbot(state): print("Aimbot:", state)
# def on_targeting(state): print("Targeting:", state)
# 
# with ToggleManager(on_aimbot, on_targeting) as tm:
#     # Use the toggle manager
#     pass
