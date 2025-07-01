"""
YOLOv8 Training Script (PT Export Only, Auto-Download + Custom Dataset Path)
"""

from ultralytics import YOLO
import torch
import os

def train():
    print(f"CUDA available for training: {torch.cuda.is_available()}")

    # Use local model if it exists, else download from ultralytics
    model_path = 'thebot/src/models/yolo/yolov8m.pt'
    if not os.path.exists(model_path):
        print("[INFO] yolov8m.pt not found locally. Downloading...")
        model = YOLO('yolov8m.pt')  # auto-downloads and caches
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
    else:
        print(f"[INFO] Using local model: {model_path}")
        model = YOLO(model_path)

    # Train the model using the correct dataset path
    dataset_path = 'thebot/scripts/dataset/data.yaml'
    model.train(
        data=dataset_path,
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        rect=True,
        name='fn_model'
    )

    # Export only best.pt
    export_path = 'thebot/src/models/best.pt'
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    model.export(format='pt', opset=12)
    print(f"[✔] Model saved as {export_path}")

if __name__ == '__main__':
    train()
