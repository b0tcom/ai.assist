"""
Production-Grade Hotkey Management System with Thread-Safe Event Handling
Purpose: Reliable hotkey detection with customizable bindings and event propagation
"""
import threading
import time
from typing import Dict, List, Callable, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum, auto
import json
from pathlib import Path
from collections import defaultdict
import queue

try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False
    keyboard = None

try:
    import pynput
    from pynput import keyboard as pynput_keyboard
    from pynput import mouse as pynput_mouse
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False

# Handle both direct execution and module import
try:
    from .logger_util import get_logger
except ImportError:
    from logger_util import get_logger


class InputBackend(Enum):
    """Available input backends"""
    KEYBOARD = auto()    # keyboard library
    PYNPUT = auto()      # pynput library
    WIN32 = auto()       # Win32 API
    MOCK = auto()        # Testing


class SystemState(Enum):
    """System operational states"""
    DISABLED = auto()
    ENABLED = auto()
    TARGETING = auto()
    MENU = auto()


@dataclass
class HotkeyBinding:
    """Hotkey binding configuration"""
    key: str
    modifiers: Set[str] = field(default_factory=set)
    action: str = ""
    description: str = ""
    enabled: bool = True
    hold_to_activate: bool = False
    cooldown_ms: int = 0
    last_triggered: float = 0.0
    
    def matches(self, key: str, current_modifiers: Set[str]) -> bool:
        """Check if key combination matches this binding"""
        return self.key == key and self.modifiers == current_modifiers
    
    def can_trigger(self) -> bool:
        """Check if hotkey can be triggered (cooldown)"""
        if self.cooldown_ms == 0:
            return True
        
        elapsed = (time.perf_counter() - self.last_triggered) * 1000
        return elapsed >= self.cooldown_ms
    
    def trigger(self) -> None:
        """Mark hotkey as triggered"""
        self.last_triggered = time.perf_counter()
    
    def to_string(self) -> str:
        """Get human-readable hotkey string"""
        parts = list(self.modifiers) + [self.key]
        return "+".join(parts).upper()


@dataclass
class InputEvent:
    """Input event data"""
    event_type: str  # 'key_down', 'key_up', 'mouse_down', 'mouse_up'
    key: str
    modifiers: Set[str]
    timestamp: float
    backend: InputBackend


class HotkeyProfile:
    """Collection of hotkey bindings"""
    
    def __init__(self, name: str = "default"):
        self.name = name
        self.bindings: Dict[str, HotkeyBinding] = {}
        self._lock = threading.RLock()
    
    def add_binding(self, binding: HotkeyBinding) -> None:
        """Add hotkey binding"""
        with self._lock:
            self.bindings[binding.action] = binding
    
    def remove_binding(self, action: str) -> None:
        """Remove hotkey binding"""
        with self._lock:
            self.bindings.pop(action, None)
    
    def get_binding(self, action: str) -> Optional[HotkeyBinding]:
        """Get binding for action"""
        with self._lock:
            return self.bindings.get(action)
    
    def find_matching(self, key: str, modifiers: Set[str]) -> Optional[HotkeyBinding]:
        """Find binding matching key combination"""
        with self._lock:
            for binding in self.bindings.values():
                if binding.enabled and binding.matches(key, modifiers):
                    return binding
            return None
    
    def save(self, path: Path) -> None:
        """Save profile to file"""
        with self._lock:
            data = {
                'name': self.name,
                'bindings': {
                    action: {
                        'key': binding.key,
                        'modifiers': list(binding.modifiers),
                        'description': binding.description,
                        'enabled': binding.enabled,
                        'hold_to_activate': binding.hold_to_activate,
                        'cooldown_ms': binding.cooldown_ms
                    }
                    for action, binding in self.bindings.items()
                }
            }
            
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'HotkeyProfile':
        """Load profile from file"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        profile = cls(data['name'])
        
        for action, binding_data in data['bindings'].items():
            binding = HotkeyBinding(
                key=binding_data['key'],
                modifiers=set(binding_data['modifiers']),
                action=action,
                description=binding_data.get('description', ''),
                enabled=binding_data.get('enabled', True),
                hold_to_activate=binding_data.get('hold_to_activate', False),
                cooldown_ms=binding_data.get('cooldown_ms', 0)
            )
            profile.add_binding(binding)
        
        return profile


class InputListener:
    """Abstract input listener interface"""
    
    def start(self) -> None:
        """Start listening for input"""
        raise NotImplementedError
    
    def stop(self) -> None:
        """Stop listening for input"""
        raise NotImplementedError
    
    def is_running(self) -> bool:
        """Check if listener is running"""
        raise NotImplementedError


class KeyboardLibListener(InputListener):
    """Listener using keyboard library"""
    
    def __init__(self, event_queue: queue.Queue):
        self.event_queue = event_queue
        self.logger = get_logger(__name__)
        self._running = False
        self._pressed_keys: Set[str] = set()
    
    def start(self) -> None:
        """Start keyboard hooks"""
        if not KEYBOARD_AVAILABLE or keyboard is None:
            raise RuntimeError("keyboard library not available")
        
        self._running = True
        
        # Hook all keys
        keyboard.hook(self._on_key_event)
        
        self.logger.info("KeyboardLib listener started")
    
    def stop(self) -> None:
        """Stop keyboard hooks"""
        if self._running and keyboard is not None:
            keyboard.unhook_all()
            self._running = False
            self.logger.info("KeyboardLib listener stopped")
    
    def is_running(self) -> bool:
        """Check if running"""
        return self._running
    
    def _on_key_event(self, event: Any) -> None:
        """Handle keyboard event"""
        try:
            # Get modifiers
            modifiers = set()
            if keyboard is not None and keyboard.is_pressed('ctrl'):
                modifiers.add('ctrl')
            if keyboard is not None and keyboard.is_pressed('alt'):
                modifiers.add('alt')
            if keyboard is not None and keyboard.is_pressed('shift'):
                modifiers.add('shift')
            
            # Create event
            input_event = InputEvent(
                event_type='key_down' if event.event_type == 'down' else 'key_up',
                key=event.name.lower(),
                modifiers=modifiers,
                timestamp=time.perf_counter(),
                backend=InputBackend.KEYBOARD
            )
            
            # Queue event
            try:
                self.event_queue.put_nowait(input_event)
            except queue.Full:
                pass
            
        except Exception as e:
            self.logger.error(f"Error handling key event: {e}")


class PynputListener(InputListener):
    """Listener using pynput library"""
    
    def __init__(self, event_queue: queue.Queue):
        self.event_queue = event_queue
        self.logger = get_logger(__name__)
        self._keyboard_listener = None
        self._mouse_listener = None
        self._modifiers: Set[str] = set()
    
    def start(self) -> None:
        """Start pynput listeners"""
        if not PYNPUT_AVAILABLE:
            raise RuntimeError("pynput library not available")
        
        # Keyboard listener
        self._keyboard_listener = pynput_keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release
        )
        
        # Mouse listener
        self._mouse_listener = pynput_mouse.Listener(
            on_click=self._on_mouse_click
        )
        
        self._keyboard_listener.start()
        self._mouse_listener.start()
        
        self.logger.info("Pynput listener started")
    
    def stop(self) -> None:
        """Stop pynput listeners"""
        if self._keyboard_listener:
            self._keyboard_listener.stop()
        if self._mouse_listener:
            self._mouse_listener.stop()
        
        self.logger.info("Pynput listener stopped")
    
    def is_running(self) -> bool:
        """Check if running"""
        return (
            bool(self._keyboard_listener and getattr(self._keyboard_listener, "running", False)) and
            bool(self._mouse_listener and getattr(self._mouse_listener, "running", False))
        )
    
    def _on_key_press(self, key) -> None:
        """Handle key press"""
        try:
            # Update modifiers
            if hasattr(key, 'name'):
                key_name = key.name.lower()
                if 'ctrl' in key_name:
                    self._modifiers.add('ctrl')
                elif 'alt' in key_name:
                    self._modifiers.add('alt')
                elif 'shift' in key_name:
                    self._modifiers.add('shift')
            else:
                key_name = str(key).strip("'").lower()
            
            # Create event
            event = InputEvent(
                event_type='key_down',
                key=key_name,
                modifiers=self._modifiers.copy(),
                timestamp=time.perf_counter(),
                backend=InputBackend.PYNPUT
            )
            
            self.event_queue.put_nowait(event)
            
        except Exception as e:
            self.logger.error(f"Error in key press handler: {e}")
    
    def _on_key_release(self, key) -> None:
        """Handle key release"""
        try:
            # Update modifiers
            if hasattr(key, 'name'):
                key_name = key.name.lower()
                if 'ctrl' in key_name:
                    self._modifiers.discard('ctrl')
                elif 'alt' in key_name:
                    self._modifiers.discard('alt')
                elif 'shift' in key_name:
                    self._modifiers.discard('shift')
            else:
                key_name = str(key).strip("'").lower()
            
            # Create event
            event = InputEvent(
                event_type='key_up',
                key=key_name,
                modifiers=self._modifiers.copy(),
                timestamp=time.perf_counter(),
                backend=InputBackend.PYNPUT
            )
            
            self.event_queue.put_nowait(event)
            
        except Exception as e:
            self.logger.error(f"Error in key release handler: {e}")
    
    def _on_mouse_click(self, x, y, button, pressed) -> None:
        """Handle mouse click"""
        try:
            button_name = f"mouse_{button.name}"
            
            event = InputEvent(
                event_type='mouse_down' if pressed else 'mouse_up',
                key=button_name,
                modifiers=self._modifiers.copy(),
                timestamp=time.perf_counter(),
                backend=InputBackend.PYNPUT
            )
            
            self.event_queue.put_nowait(event)
            
        except Exception as e:
            self.logger.error(f"Error in mouse click handler: {e}")


class HotkeyManager:
    """
    Production hotkey management system with multiple backend support
    """
    
    def __init__(self, backend: Optional[InputBackend] = None):
        self.logger = get_logger(__name__)
        self.backend = backend or self._select_backend()
        
        # Event system
        self.event_queue: queue.Queue[InputEvent] = queue.Queue(maxsize=1000)
        self.callbacks: Dict[str, List[Callable[[], None]]] = defaultdict(list)
        
        # Profiles
        self.profiles: Dict[str, HotkeyProfile] = {}
        self.active_profile: Optional[HotkeyProfile] = None
        
        # State tracking
        self.system_state = SystemState.DISABLED
        self.held_keys: Dict[str, float] = {}  # Track held keys for hold-to-activate
        
        # Threading
        self.listener: Optional[InputListener] = None
        self.event_thread: Optional[threading.Thread] = None
        self._running = False
        
        # Load default profile
        self._load_default_profile()
        
        self.logger.info(f"HotkeyManager initialized with backend: {self.backend.name}")
    
    def _select_backend(self) -> InputBackend:
        """Select best available backend"""
        if KEYBOARD_AVAILABLE:
            return InputBackend.KEYBOARD
        elif PYNPUT_AVAILABLE:
            return InputBackend.PYNPUT
        else:
            self.logger.warning("No input libraries available, using mock backend")
            return InputBackend.MOCK
    
    def _load_default_profile(self) -> None:
        """Load default hotkey profile"""
        profile = HotkeyProfile("default")
        
        # Default bindings
        default_bindings = [
            HotkeyBinding(key='f1', action='toggle_system', 
                         description='Toggle system on/off'),
            HotkeyBinding(key='f2', action='toggle_overlay', 
                         description='Toggle overlay display'),
            HotkeyBinding(key='f5', action='reload_config', 
                         description='Reload configuration'),
            HotkeyBinding(key='mouse_right', action='aim_hold', 
                         description='Hold to aim', hold_to_activate=True),
            HotkeyBinding(key='escape', modifiers={'ctrl'}, 
                         action='emergency_stop', description='Emergency stop'),
        ]
        
        for binding in default_bindings:
            profile.add_binding(binding)
        
        self.profiles['default'] = profile
        self.active_profile = profile
    
    def start(self) -> None:
        """Start hotkey monitoring"""
        if self._running:
            return
        
        self._running = True
        
        # Create listener
        if self.backend == InputBackend.KEYBOARD:
            self.listener = KeyboardLibListener(self.event_queue)
        elif self.backend == InputBackend.PYNPUT:
            self.listener = PynputListener(self.event_queue)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
        
        # Start listener
        self.listener.start()
        
        # Start event processing thread
        self.event_thread = threading.Thread(
            target=self._event_processor,
            name="HotkeyEventProcessor",
            daemon=True
        )
        self.event_thread.start()
        
        self.system_state = SystemState.ENABLED
        self.logger.info("HotkeyManager started")
    
    def stop(self) -> None:
        """Stop hotkey monitoring"""
        if not self._running:
            return
        
        self._running = False
        self.system_state = SystemState.DISABLED
        
        # Stop listener
        if self.listener:
            self.listener.stop()
        
        # Wait for event thread
        if self.event_thread:
            self.event_thread.join(timeout=2.0)
        
        self.logger.info("HotkeyManager stopped")
    
    def _event_processor(self) -> None:
        """Process input events"""
        while self._running:
            try:
                # Get event with timeout
                event = self.event_queue.get(timeout=0.1)
                
                # Process event
                self._process_event(event)
                
            except queue.Empty:
                # Check for held keys timeout
                self._check_held_keys()
                
            except Exception as e:
                self.logger.error(f"Event processing error: {e}")
    
    def _process_event(self, event: InputEvent) -> None:
        """Process single input event"""
        if not self.active_profile:
            return
        
        # Find matching binding
        binding = self.active_profile.find_matching(event.key, event.modifiers)
        
        if not binding:
            return
        
        # Check cooldown
        if not binding.can_trigger():
            return
        
        # Handle based on binding type
        if binding.hold_to_activate:
            if event.event_type in ['key_down', 'mouse_down']:
                # Start holding
                self.held_keys[binding.action] = event.timestamp
                self._trigger_action(binding.action, holding=True)
            elif event.event_type in ['key_up', 'mouse_up']:
                # Stop holding
                if binding.action in self.held_keys:
                    del self.held_keys[binding.action]
                    self._trigger_action(binding.action, holding=False)
        else:
            # Normal trigger on key down
            if event.event_type in ['key_down', 'mouse_down']:
                binding.trigger()
                self._trigger_action(binding.action)
    
    def _check_held_keys(self) -> None:
        """Check for held key timeouts"""
        current_time = time.perf_counter()
        timeout_actions = []
        
        for action, start_time in self.held_keys.items():
            if current_time - start_time > 10.0:  # 10 second timeout
                timeout_actions.append(action)
        
        # Release timed out keys
        for action in timeout_actions:
            del self.held_keys[action]
            self._trigger_action(action, holding=False)
    
    def _trigger_action(self, action: str, holding: Optional[bool] = None) -> None:
        """Trigger action callbacks"""
        callbacks = self.callbacks.get(action, [])
        
       
    def register_callback(self, action: str, callback: Callable) -> None:
        """Register action callback"""
        self.callbacks[action].append(callback)
        self.logger.debug(f"Registered callback for action: {action}")
    
    def unregister_callback(self, action: str, callback: Callable) -> None:
        """Unregister action callback"""
        if action in self.callbacks:
            try:
                self.callbacks[action].remove(callback)
            except ValueError:
                pass
    
    def set_profile(self, profile_name: str) -> bool:
        """Switch to different profile"""
        if profile_name in self.profiles:
            self.active_profile = self.profiles[profile_name]
            self.logger.info(f"Switched to profile: {profile_name}")
            return True
        return False
    
    def create_profile(self, name: str) -> HotkeyProfile:
        """Create new profile"""
        profile = HotkeyProfile(name)
        self.profiles[name] = profile
        return profile
    
    def save_profiles(self, directory: Path) -> None:
        """Save all profiles"""
        directory.mkdir(exist_ok=True)
        
        for name, profile in self.profiles.items():
            profile.save(directory / f"{name}.json")
    
    def load_profiles(self, directory: Path) -> None:
        """Load profiles from directory"""
        if not directory.exists():
            return
        
        for profile_file in directory.glob("*.json"):
            try:
                profile = HotkeyProfile.load(profile_file)
                self.profiles[profile.name] = profile
                self.logger.info(f"Loaded profile: {profile.name}")
            except Exception as e:
                self.logger.error(f"Failed to load profile {profile_file}: {e}")
    
    def get_bindings(self) -> Dict[str, str]:
        """Get current hotkey bindings"""
        if not self.active_profile:
            return {}
        
        return {
            binding.action: binding.to_string()
            for binding in self.active_profile.bindings.values()
            if binding.enabled
        }
    
    def rebind(self, action: str, key: str, modifiers: Optional[Set[str]] = None) -> bool:
        """Rebind action to new key"""
        if not self.active_profile:
            return False
        
        binding = self.active_profile.get_binding(action)
        if not binding:
            return False
        
        # Check for conflicts
        conflict = self.active_profile.find_matching(key, modifiers or set())
        if conflict and conflict.action != action:
            self.logger.warning(f"Key combination already bound to: {conflict.action}")
            return False
        
        # Update binding
        binding.key = key
        binding.modifiers = modifiers or set()
        
        self.logger.info(f"Rebound {action} to {binding.to_string()}")
        return True
    
    def get_state(self) -> SystemState:
        """Get current system state"""
        return self.system_state
    
    def set_state(self, state: SystemState) -> None:
        """Set system state"""
        self.system_state = state
        self.logger.info(f"System state changed to: {state.name}")