"""
Production-Grade Arduino Mouse Control System with Safety Protocols
Purpose: Deterministic hardware control with microsecond precision and safety enforcement
"""
import serial
import serial.tools.list_ports
import time
import threading
import queue
from typing import Optional, Dict, Any, List, Tuple, Protocol, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
import struct
try:
    import crc16
    CRC16_AVAILABLE = True
except ImportError:
    CRC16_AVAILABLE = False
    crc16 = None
from collections import deque
import statistics

try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False
    keyboard = None

# Handle both direct execution and module import
try:
    from .logger_util import get_logger
    from .config_manager import ConfigManager, ArduinoConfig, AimConfig
except ImportError:
    from logger_util import get_logger
    from config_manager import ConfigManager, ArduinoConfig, AimConfig


class CommandType(Enum):
    """Arduino command types with protocol versioning"""
    MOVE = 0x01
    CLICK = 0x02
    RECOIL = 0x03
    STATUS = 0x04
    CALIBRATE = 0x05
    SAFETY_OVERRIDE = 0x06
    FIRMWARE_INFO = 0x07


class SafetyLevel(Enum):
    """Safety enforcement levels"""
    OFF = 0
    BASIC = 1      # Basic bounds checking
    STANDARD = 2   # Rate limiting + bounds
    STRICT = 3     # All safety features
    
    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented
    
    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented
    
    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented
    
    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


@dataclass
class MovementCommand:
    """Validated movement command with safety checks"""
    dx: int
    dy: int
    timestamp_us: float
    priority: int = 0
    
    def __post_init__(self):
        """Validate movement parameters"""
        # Clamp to safe ranges
        MAX_MOVEMENT = 500  # pixels
        self.dx = max(-MAX_MOVEMENT, min(self.dx, MAX_MOVEMENT))
        self.dy = max(-MAX_MOVEMENT, min(self.dy, MAX_MOVEMENT))
    
    def to_bytes(self) -> bytes:
        """Serialize to wire format with CRC"""
        # Format: [CMD:1][DX:2][DY:2][TS:4][CRC:2]
        data = struct.pack('<BhhI', 
                          CommandType.MOVE.value,
                          self.dx, self.dy,
                          int(self.timestamp_us))
        
        # Add CRC16 if available
        if CRC16_AVAILABLE and crc16:
            crc = crc16.crc16xmodem(data)
            return data + struct.pack('<H', crc)
        else:
            # Simple checksum if CRC16 not available
            checksum = sum(data) & 0xFFFF
            return data + struct.pack('<H', checksum)


@dataclass
class PerformanceMetrics:
    """Detailed performance tracking"""
    command_latency_us: deque = field(default_factory=lambda: deque(maxlen=100))
    roundtrip_time_us: deque = field(default_factory=lambda: deque(maxlen=100))
    commands_per_second: float = 0.0
    dropped_commands: int = 0
    safety_violations: int = 0
    
    def add_latency(self, latency_us: float) -> None:
        """Add command latency measurement"""
        self.command_latency_us.append(latency_us)
    
    def add_roundtrip(self, rtt_us: float) -> None:
        """Add roundtrip time measurement"""
        self.roundtrip_time_us.append(rtt_us)
    
    def get_stats(self) -> Dict[str, float]:
        """Get statistical summary"""
        stats = {}
        
        if self.command_latency_us:
            stats['latency_avg_us'] = statistics.mean(self.command_latency_us)
            stats['latency_p99_us'] = sorted(self.command_latency_us)[int(len(self.command_latency_us) * 0.99)]
        
        if self.roundtrip_time_us:
            stats['rtt_avg_us'] = statistics.mean(self.roundtrip_time_us)
            stats['rtt_p99_us'] = sorted(self.roundtrip_time_us)[int(len(self.roundtrip_time_us) * 0.99)]
        
        stats['commands_per_second'] = self.commands_per_second
        stats['dropped_commands'] = self.dropped_commands
        stats['safety_violations'] = self.safety_violations
        
        return stats


class InputBackend(ABC):
    """Abstract input backend interface"""
    
    @property
    @abstractmethod
    def connected(self) -> bool:
        """Connection status"""
        pass
    
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection"""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Close connection"""
        pass
    
    @abstractmethod
    def send_movement(self, dx: int, dy: int) -> bool:
        """Send movement command"""
        pass
    
    @abstractmethod
    def send_click(self, button: int, state: bool) -> bool:
        """Send click command"""
        pass
    
    @abstractmethod
    def get_status(self) -> Optional[Dict[str, Any]]:
        """Get device status"""
        pass


class ArduinoBackend(InputBackend):
    """Arduino serial communication backend with protocol validation"""
    
    def __init__(self, config: ArduinoConfig, safety_level: SafetyLevel):
        self.logger = get_logger(__name__)
        self.config = config
        self.safety_level = safety_level
        self.serial_port: Optional[serial.Serial] = None
        self._connected = False
        
        # Command queue for async processing
        self.command_queue: queue.Queue[MovementCommand] = queue.Queue(maxsize=100)
        self.response_queue: queue.Queue[bytes] = queue.Queue()
        
        # Worker threads
        self.tx_thread: Optional[threading.Thread] = None
        self.rx_thread: Optional[threading.Thread] = None
        self.running = False
        
        # Performance tracking
        self.metrics = PerformanceMetrics()
        self.last_command_time = 0
        self.command_count = 0
        
        # Safety enforcement
        self.rate_limiter = RateLimiter(max_rate=1000)  # 1000 commands/sec
        self.movement_validator = MovementValidator()
    
    @property
    def connected(self) -> bool:
        """Connection status"""
        return self._connected
    
    def connect(self) -> bool:
        """Establish Arduino connection with handshake"""
        try:
            # Find Arduino port
            arduino_port = self._find_arduino_port()
            if not arduino_port:
                self.logger.error("No Arduino found")
                return False
            
            # Open serial connection
            self.serial_port = serial.Serial(
                port=arduino_port,
                baudrate=self.config.baudrate,
                timeout=self.config.timeout,
                write_timeout=self.config.timeout,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            
            # Wait for Arduino reset
            time.sleep(2.0)
            
            # Perform handshake
            if not self._handshake():
                self.logger.error("Handshake failed")
                self.serial_port.close()
                return False
            
            # Start worker threads
            self.running = True
            self.tx_thread = threading.Thread(target=self._tx_worker, daemon=True)
            self.rx_thread = threading.Thread(target=self._rx_worker, daemon=True)
            self.tx_thread.start()
            self.rx_thread.start()
            
            self._connected = True
            self.logger.info(f"Arduino connected on {arduino_port}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            return False
    
    def _find_arduino_port(self) -> Optional[str]:
        """Auto-detect Arduino port"""
        # First try configured port
        if self.config.port and self._is_arduino_port(self.config.port):
            return self.config.port
        
        # Auto-detect
        ports = serial.tools.list_ports.comports()
        for port in ports:
            if 'Arduino' in port.description or 'CH340' in port.description:
                self.logger.info(f"Found Arduino: {port.device} - {port.description}")
                return port.device
        
        return None
    
    def _is_arduino_port(self, port: str) -> bool:
        """Check if port has Arduino"""
        try:
            with serial.Serial(port, 9600, timeout=0.1) as test_port:
                return True
        except Exception:
            return False
    
    def _handshake(self) -> bool:
        """Perform protocol handshake"""
        if not self.serial_port:
            return False
            
        # Send firmware info request
        cmd = struct.pack('<B', CommandType.FIRMWARE_INFO.value)
        self.serial_port.write(cmd)
        
        # Wait for response
        response = self.serial_port.read(32)
        if len(response) < 8:
            return False
        
        # Parse firmware info
        try:
            version = struct.unpack('<HH', response[:4])
            self.logger.info(f"Arduino firmware v{version[0]}.{version[1]}")
            return True
        except Exception:
            return False
    
    def _tx_worker(self) -> None:
        """Transmit worker thread"""
        while self.running:
            try:
                # Get command with timeout
                cmd = self.command_queue.get(timeout=0.001)
                
                # Apply safety checks
                if self.safety_level != SafetyLevel.OFF:
                    if not self._validate_command(cmd):
                        self.metrics.safety_violations += 1
                        continue
                
                # Send command
                start_time = time.perf_counter()
                if self.serial_port is not None:
                    self.serial_port.write(cmd.to_bytes())
                else:
                    self.logger.error("Serial port is not initialized.")
                    continue
                
                # Track metrics
                latency = (time.perf_counter() - start_time) * 1e6
                self.metrics.add_latency(latency)
                
                # Update command rate
                self.command_count += 1
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"TX error: {e}")
    
    def _rx_worker(self) -> None:
        """Receive worker thread"""
        buffer = bytearray()
        
        while self.running:
            try:
                # Read available data
                if self.serial_port and self.serial_port.in_waiting:
                    data = self.serial_port.read(self.serial_port.in_waiting)
                    buffer.extend(data)
                    
                    # Process complete responses
                    while len(buffer) >= 4:
                        # Simple response format: [CMD:1][STATUS:1][DATA:2]
                        response = buffer[:4]
                        buffer = buffer[4:]
                        
                        self.response_queue.put(response)
                
                else:
                    time.sleep(0.001)
                    
            except Exception as e:
                self.logger.error(f"RX error: {e}")
    
    def _validate_command(self, cmd: MovementCommand) -> bool:
        """Apply safety validation"""
        # Rate limiting
        if self.safety_level >= SafetyLevel.STANDARD:
            if not self.rate_limiter.check():
                return False
        
        # Movement validation
        if self.safety_level >= SafetyLevel.BASIC:
            if not self.movement_validator.validate(cmd.dx, cmd.dy):
                return False
        
        return True
    
    def send_movement(self, dx: int, dy: int) -> bool:
        """Queue movement command"""
        if not self.connected:
            return False
        
        try:
            cmd = MovementCommand(
                dx=dx, 
                dy=dy,
                timestamp_us=time.perf_counter() * 1e6
            )
            
            self.command_queue.put_nowait(cmd)
            return True
            
        except queue.Full:
            self.metrics.dropped_commands += 1
            return False
    
    def send_click(self, button: int, state: bool) -> bool:
        """Send click command"""
        if not self.connected or not self.serial_port:
            return False
        
        try:
            cmd = struct.pack('<BBB', 
                            CommandType.CLICK.value,
                            button,
                            1 if state else 0)
            
            self.serial_port.write(cmd)
            return True
            
        except Exception as e:
            self.logger.error(f"Click command failed: {e}")
            return False
    
    def get_status(self) -> Optional[Dict[str, Any]]:
        """Get Arduino status"""
        if not self.connected or not self.serial_port:
            return None
        
        try:
            # Send status request
            cmd = struct.pack('<B', CommandType.STATUS.value)
            self.serial_port.write(cmd)
            
            # Wait for response
            response = self.response_queue.get(timeout=0.1)
            
            # Parse status
            if len(response) >= 4:
                status = {
                    'connected': True,
                    'queue_size': self.command_queue.qsize(),
                    'metrics': self.metrics.get_stats()
                }
                return status
                
        except Exception:
            pass
        
        return None
    
    def disconnect(self) -> None:
        """Disconnect Arduino"""
        self.running = False
        
        if self.tx_thread:
            self.tx_thread.join(timeout=1.0)
        if self.rx_thread:
            self.rx_thread.join(timeout=1.0)
        
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
        
        self._connected = False
        self.logger.info("Arduino disconnected")


class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(self, max_rate: int):
        self.max_rate = max_rate
        self.tokens = max_rate
        self.last_update = time.perf_counter()
        self.lock = threading.Lock()
    
    def check(self) -> bool:
        """Check if action is allowed"""
        with self.lock:
            now = time.perf_counter()
            elapsed = now - self.last_update
            
            # Refill tokens
            self.tokens = min(self.max_rate, self.tokens + elapsed * self.max_rate)
            self.last_update = now
            
            # Check if token available
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            
            return False


class MovementValidator:
    """Movement command validation with pattern detection"""
    
    def __init__(self):
        self.history = deque(maxlen=10)
        self.total_movement = [0, 0]
    
    def validate(self, dx: int, dy: int) -> bool:
        """Validate movement command"""
        # Check single movement magnitude
        magnitude = (dx*dx + dy*dy) ** 0.5
        if magnitude > 500:  # Max 500 pixel movement
            return False
        
        # Track cumulative movement
        self.total_movement[0] += dx
        self.total_movement[1] += dy
        
        # Reset if too much cumulative movement
        if abs(self.total_movement[0]) > 2000 or abs(self.total_movement[1]) > 2000:
            self.total_movement = [0, 0]
        
        # Record history
        self.history.append((dx, dy, time.perf_counter()))
        
        return True


class InputController:
    """
    Production input controller with backend abstraction and safety protocols
    """
    
    def __init__(self, 
                 config_manager: Optional[ConfigManager] = None,
                 safety_level: SafetyLevel = SafetyLevel.STANDARD,
                 enable_anti_recoil: bool = True):
        
        self.logger = get_logger(__name__)
        self.config_manager = config_manager or ConfigManager()
        self.arduino_config = self.config_manager.get_arduino_config()
        self.aim_config = self.config_manager.get_aim_config()
        self.safety_level = safety_level
        
        # Backend
        self.backend: Optional[InputBackend] = None
        
        # Anti-recoil system
        self.enable_anti_recoil = enable_anti_recoil
        self.recoil_compensator = RecoilCompensator() if enable_anti_recoil else None
        
        # Performance tracking
        self.movement_history = deque(maxlen=100)
        self.last_target_update = 0
        
        # Safety callbacks
        self.safety_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        
        self.logger.info(f"Input controller initialized with safety level: {safety_level.name}")
    
    def connect(self) -> bool:
        """Connect to input device"""
        try:
            # Create Arduino backend
            self.backend = ArduinoBackend(self.arduino_config, self.safety_level)
            
            # Connect
            if not self.backend.connect():
                self.logger.error("Failed to connect to Arduino")
                return False
            
            # Register safety monitoring
            if self.safety_level != SafetyLevel.OFF:
                self._start_safety_monitor()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            return False
    
    def _start_safety_monitor(self) -> None:
        """Start safety monitoring thread"""
        def monitor():
            while self.backend and self.backend.connected:
                status = self.backend.get_status()
                if status:
                    # Check for violations
                    metrics = status.get('metrics', {})
                    if metrics.get('safety_violations', 0) > 10:
                        self.logger.warning("High safety violation rate detected")
                        
                        # Notify callbacks
                        for callback in self.safety_callbacks:
                            callback({'type': 'safety_violation', 'metrics': metrics})
                
                time.sleep(1.0)
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
    
    def move_to_target(self, 
                      predicted_pos: Dict[str, int], 
                      target_info: Dict[str, Any]) -> bool:
        """
        Move to predicted target position with sub-pixel precision
        
        Args:
            predicted_pos: Predicted position with 'x' and 'y' keys
            target_info: Target information including center position
            
        Returns:
            Success status
        """
        if not self.backend:
            return False
        
        try:
            # Calculate precise movement delta
            current_center = target_info.get('center', (0, 0))
            dx = predicted_pos['x'] - current_center[0]
            dy = predicted_pos['y'] - current_center[1]
            
            # Apply sensitivity scaling
            dx = int(dx * self.aim_config.sensitivity)
            dy = int(dy * self.aim_config.sensitivity)
            
            # Apply anti-recoil if enabled
            if self.recoil_compensator and self._is_firing():
                recoil_dx, recoil_dy = self.recoil_compensator.get_compensation()
                dx += recoil_dx
                dy += recoil_dy
            
            # Send movement
            success = self.backend.send_movement(dx, dy)
            
            # Track movement
            if success:
                self.movement_history.append({
                    'dx': dx,
                    'dy': dy,
                    'timestamp': time.perf_counter(),
                    'target_id': target_info.get('tracking_id', 0)
                })
            
            return success
            
        except Exception as e:
            self.logger.error(f"Movement error: {e}")
            return False
    
    def _is_firing(self) -> bool:
        """Check if weapon is firing"""
        if not KEYBOARD_AVAILABLE or not keyboard:
            return False
        
        try:
            # Check common fire keys
            return (keyboard.is_pressed('left') or 
                    keyboard.is_pressed('mouse left') or
                    keyboard.is_pressed('mouse1'))
        except Exception:
            return False
    
    def handle_recoil(self) -> None:
        """Apply anti-recoil compensation"""
        if not self.recoil_compensator or not self._is_firing():
            return
        
        # Get compensation values
        dx, dy = self.recoil_compensator.get_compensation()
        
        if (dx != 0 or dy != 0) and self.backend:
            self.backend.send_movement(dx, dy)
    
    def register_safety_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register safety event callback"""
        self.safety_callbacks.append(callback)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics"""
        if not self.backend:
            return {}
        
        status = self.backend.get_status()
        if not status:
            return {}
        
        # Add movement statistics
        if self.movement_history:
            recent_movements = list(self.movement_history)[-10:]
            avg_dx = sum(m['dx'] for m in recent_movements) / len(recent_movements)
            avg_dy = sum(m['dy'] for m in recent_movements) / len(recent_movements)
            
            status['movement'] = {
                'avg_dx': avg_dx,
                'avg_dy': avg_dy,
                'count': len(self.movement_history)
            }
        
        return status
    
    def is_connected(self) -> bool:
        """Check connection status"""
        return self.backend is not None and self.backend.connected
    
    def disconnect(self) -> None:
        """Disconnect from device"""
        if self.backend:
            self.backend.disconnect()
            self.backend = None
        
        self.logger.info("Input controller disconnected")


class RecoilCompensator:
    """Advanced recoil compensation with pattern learning"""
    
    def __init__(self):
        self.patterns = {
            'default': [(0, 5), (0, 4), (1, 3), (-1, 3), (0, 2)],
            'ak47': [(0, 7), (1, 6), (-1, 5), (2, 4), (-2, 3)],
            'smg': [(0, 3), (0, 3), (1, 2), (-1, 2), (0, 2)]
        }
        self.current_pattern = 'default'
        self.shot_index = 0
        self.last_shot_time = 0
    
    def get_compensation(self) -> Tuple[int, int]:
        """Get recoil compensation for current shot"""
        now = time.perf_counter()
        
        # Reset if too much time passed
        if now - self.last_shot_time > 0.2:  # 200ms
            self.shot_index = 0
        
        self.last_shot_time = now
        
        # Get pattern
        pattern = self.patterns.get(self.current_pattern, self.patterns['default'])
        
        if self.shot_index < len(pattern):
            dx, dy = pattern[self.shot_index]
            self.shot_index += 1
            return dx, -dy  # Negative Y for upward compensation
        
        return 0, 0
    
    def set_pattern(self, pattern_name: str) -> None:
        """Set recoil pattern"""
        if pattern_name in self.patterns:
            self.current_pattern = pattern_name
            self.shot_index = 0