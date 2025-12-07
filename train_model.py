# main.py

from glob import glob
import os
import torch
from model_utils import get_data_loaders, SimpleCNN, train_model, export_model_onnx

print("Begin model training...")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using Device: {DEVICE}")
if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    print(f"[INFO] cuDNN benchmark enabled: True")
DATA_DIR = glob("*train_test_split*")[0]  # Automatically find the split dataset directory
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
STEP_SIZE = 10

# Load data
train_loader, test_loader, num_classes = get_data_loaders(DATA_DIR, batch_size=BATCH_SIZE)

# Initialize and train model
model = SimpleCNN(num_classes=num_classes).to(DEVICE)
train_model(model, train_loader, test_loader, device=DEVICE, epochs=EPOCHS, lr=LR, step_size = STEP_SIZE)

# Export to ONNX
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
export_model_onnx(model, output_path=os.path.join(output_dir, "model.onnx"))