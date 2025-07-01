"""
Input Handler Module
Purpose: Manages mouse input via Arduino and anti-recoil logic.
"""
try:
    import serial
except ImportError:
    serial = None
import time
try:
    import keyboard # type: ignore
except ImportError:
    keyboard = None
from utils import Logger

class InputController:
    """Handles communication with an Arduino for mouse control."""

    def __init__(self, config):
        self.logger = Logger(__name__)
        self.config = config
        self.port = self.config.get('arduino_port', 'COM5')
        self.baudrate = self.config.get('arduino_baudrate', 115200)
        self.ser = None
        
        # Anti-recoil settings
        self.recoil_strength = self.config.get('recoil_strength')

    def connect(self):
        """Establishes a serial connection with the Arduino."""
        if not serial:
            self.logger.error("pyserial not installed!")
            return
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.1)
            time.sleep(2) # Wait for connection to establish
            self.logger.info(f"Successfully connected to Arduino on {self.port}.")
        except Exception as e:
            self.logger.error(f"Failed to connect to Arduino on {self.port}. "
                              "Ensure it's connected and the correct port is in the config. "
                              f"Error: {e}")
            self.ser = None

    def disconnect(self):
        """Closes the serial connection."""
        if self.ser and self.ser.is_open:
            self.ser.close()
            self.logger.info("Disconnected from Arduino.")

    def move_to_target(self, predicted_pos, target_info):
        """Sends a command to the Arduino to move the mouse."""
        if not self.ser or not self.ser.is_open:
            return
            
        dx = predicted_pos['x'] - target_info['center'][0]
        dy = predicted_pos['y'] - target_info['center'][1]
        
        # Format: "m,dx,dy\n" e.g., "m,10,-5\n"
        command = f"m,{int(dx)},{int(dy)}\n"
        try:
            self.ser.write(command.encode('utf-8'))
        except Exception as e:
            self.logger.error(f"Error writing to Arduino: {e}")

    def handle_recoil(self):
        """Applies anti-recoil if the left mouse button is held down."""
        if not self.ser or not self.ser.is_open or not keyboard:
            return
        
        # Check if user is firing (e.g., holding left mouse button)
        if keyboard.is_pressed('left mouse button'):
             # Command format: "r,strength\n" e.g., "r,5\n"
            command = f"r,{self.recoil_strength}\n"
            try:
                self.ser.write(command.encode('utf-8'))
            except Exception as e:
                self.logger.error(f"Error sending recoil command: {e}")

    def is_connected(self):
        """Returns True if Arduino is connected."""
        return self.ser is not None and self.ser.is_open

    def learn_recoil_pattern(self, gameplay_data):
        """Stub: Learn and adapt recoil compensation from gameplay patterns (to be implemented)."""
        # Example: Fit a model to gameplay_data and update self.recoil_strength or pattern
        pass

    def update_recoil_model(self, new_model):
        """Replace or update the internal recoil compensation model."""
        self.recoil_model = new_model

# InputController is imported and used in main.py as intended.
# It is initialized with config and used for Arduino mouse control.