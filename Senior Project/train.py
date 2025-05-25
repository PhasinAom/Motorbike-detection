from ultralytics import YOLO
import torch

# Define dataset path - update this to your dataset location
DATASET_PATH = "/Users/phasin/dataset/data.yaml"

def get_device():
    if torch.backends.mps.is_available():
        return "mps"  # Use Metal Performance Shaders if available
    return "cpu"      # Fallback to CPU

def train_model():
    # Load a pretrained YOLOv8n model
    model = YOLO('yolov8n.pt')
    
    # Get appropriate device for training
    device = get_device()
    print(f"Training on device: {device}")
    
    # Train the model
    results = model.train(
        data=DATASET_PATH,      # Path to data.yaml
        epochs=100,             # Number of epochs
        imgsz=640,             # Image size
        batch=8,               # Reduced batch size for CPU/MPS
        device=device,         # Use detected device
        workers=4,             # Reduced workers for CPU/MPS
        patience=50,           # Early stopping patience
        project='runs/train',  # Save results to project/name
        name='helmet_exp',     # Name of the experiment
        exist_ok=True         # Overwrite existing experiment
    )

if __name__ == "__main__":
    train_model()