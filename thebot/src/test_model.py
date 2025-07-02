import cv2
from ultralytics import YOLO

# Test with a sample image
model = YOLO("thebot/src/models/best.pt")

# Create a test image or load one
test_img = cv2.imread("test_image.jpg")  # Replace with actual test image
if test_img is None:
    # Create a dummy image for testing
    import numpy as np
    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

print(f"Test image shape: {test_img.shape}")

results = model(test_img, verbose=True)
print(f"Results: {results}")

if results[0].boxes is not None:
    print(f"Found {len(results[0].boxes)} detections")
    for box in results[0].boxes:
        print(f"Class: {box.cls.item()}, Conf: {box.conf.item():.3f}")
else:
    print("No detections found")