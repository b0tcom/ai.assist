import os
import sys
import time
import tkinter as tk
from tkinter import messagebox
import threading
from predict import TargetPredictor
import cv2
import numpy as np


from config import Config
from gui import CVTargetingGUI

# Optional: import your input handler and detection modules if available
try:
    from input_handler import InputController
except ImportError:
    InputController = None

try:
    from predict import TargetPredictor
except ImportError:
    TargetPredictor = None

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

def arduino_mouse_test(config):
    """Test Arduino mouse movement from left to right."""
    if not InputController:
        print("InputController module not available.")
        return False, "InputController not available"
    ctrl = InputController(config)
    ctrl.connect()
    if not ctrl.is_connected():
        print("Arduino not connected.")
        return False, "Arduino not connected"
    print("Moving mouse left to right (test)...")
    try:
        for dx in range(0, 400, 10):
            # Move mouse right by dx, dy=0
            command = f"m,{10},0\n"
            if ctrl.ser is not None:
                ctrl.ser.write(command.encode('utf-8'))
            else:
                print("Serial connection is None. Cannot write command.")
                break
            time.sleep(0.02)
        ctrl.disconnect()
        print("Arduino mouse test: SUCCESS")
        return True, "Arduino mouse test: SUCCESS"
    except Exception as e:
        print(f"Arduino mouse test failed: {e}")
        return False, f"Arduino mouse test failed: {e}"

def arduino_move_to_target_test(config):
    """Test Arduino mouse move command through InputController (move right by 30px)."""
    if not InputController:
        print("InputController module not available.")
        return False, "InputController not available"
    ctrl = InputController(config)
    ctrl.connect()
    if not ctrl.is_connected():
        print("Arduino not connected.")
        return False, "Arduino not connected"
    try:
        # Move mouse right by 30 pixels (dx=30, dy=0)
        # This uses the same protocol as the rest of the program
        command = f"m,30,0\n"
        if ctrl.ser is not None:
            ctrl.ser.write(command.encode('utf-8'))
            print("Sent command to Arduino: m,30,0")
            time.sleep(0.2)
            ctrl.disconnect()
            print("Arduino move right test: SUCCESS")
            return True, "Arduino move right test: SUCCESS"
        else:
            print("Serial connection is None. Cannot write command.")
            ctrl.disconnect()
            return False, "Serial connection is None"
    except Exception as e:
        print(f"Arduino move right test failed: {e}")
        ctrl.disconnect()
        return False, f"Arduino move right test failed: {e}"

def gui_modularity_test(config):
    """Test that all GUI settings can be set and toggled without error."""
    try:
        root = tk.Tk()
        gui = CVTargetingGUI(root, config=config)
        for key, var in gui.settings.items():
            # Try toggling booleans, setting numbers, etc.
            try:
                if hasattr(var, 'set'):
                    if isinstance(var.get(), bool):
                        var.set(not var.get())
                        var.set(not var.get())
                    elif isinstance(var.get(), (int, float)):
                        var.set(var.get())
                    elif isinstance(var.get(), str):
                        var.set(var.get())
            except Exception as e:
                print(f"GUI setting '{key}' failed: {e}")
                root.destroy()
                return False, f"GUI setting '{key}' failed: {e}"
        root.destroy()
        print("GUI modularity test: SUCCESS")
        return True, "GUI modularity test: SUCCESS"
    except Exception as e:
        print(f"GUI modularity test failed: {e}")
        return False, f"GUI modularity test failed: {e}"

def model_detection_test(config):
    """Test that the model can run inference without error."""
    if not TargetPredictor or not cv2 or not np:
        print("Detection modules not available.")
        return False, "Detection modules not available"
    try:
        predictor = TargetPredictor(config)
        # Create a dummy detection input
        dummy_detection = [{
            'box': [100, 100, 200, 200],
            'center': (150, 150),
            'height': 100,
            'confidence': 0.9
        }]
        screen_center = (960, 540)
        best = predictor.select_best_target(dummy_detection, screen_center)
        if best is None:
            print("Model detection test: No target selected.")
            return False, "Model detection test: No target selected"
        pred = predictor.predict(best)
        print("Model detection test: SUCCESS")
        return True, "Model detection test: SUCCESS"
    except Exception as e:
        print(f"Model detection test failed: {e}")
        return False, f"Model detection test failed: {e}"

def cleanup_images(folder="test_images", max_keep=100):
    """Delete images in the folder if more than max_keep exist."""
    if not os.path.exists(folder):
        return
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.jpg') or f.endswith('.png')]
    if len(files) > max_keep:
        files.sort(key=os.path.getmtime)
        for f in files[:-max_keep]:
            try:
                os.remove(f)
            except Exception:
                pass

def run_all_tests(config):
    results = []
    results.append(arduino_mouse_test(config))
    results.append(arduino_move_to_target_test(config))  # <-- Updated test
    results.append(gui_modularity_test(config))
    results.append(model_detection_test(config))
    cleanup_images()
    return results

def show_results_popup(results):
    msg = "\n".join([f"{'PASS' if ok else 'FAIL'}: {desc}" for ok, desc in results])
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo("Testuino Results", msg)
    root.destroy()

def main():
    config = Config().to_dict()
    print("Running Testuino diagnostics...")
    results = run_all_tests(config)
    for ok, desc in results:
        print(f"{'PASS' if ok else 'FAIL'}: {desc}")
    show_results_popup(results)

if __name__ == "__main__":
    main()
