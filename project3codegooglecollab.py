!pip install ultralytics torch torchvision matplotlib opencv-python

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

base_path = '/content/drive/My Drive/data'

# Training and validation data paths
train_data = f'{base_path}/train'
valid_data = f'{base_path}/valid'

# Path to the YAML file
yaml_path = f'{base_path}/data.yaml'

# ===== Verify Dataset =====
# Confirm the dataset files are accessible
import os
print("Training Images:", os.listdir(f'{train_data}/images'))
print("Training Labels:", os.listdir(f'{train_data}/labels'))
print("Validation Images:", os.listdir(f'{valid_data}/images'))
print("Validation Labels:", os.listdir(f'{valid_data}/labels'))
print("YAML File Path:", yaml_path)

# ===== Import YOLOv8 =====
from ultralytics import YOLO

# ===== Train YOLOv8 =====
# Load YOLOv8 Nano Model
model = YOLO("yolov8n.pt")  # YOLOv8 Nano weights for lightweight training

# Start Training
model.train(
    data=yaml_path,        # Path to data.yaml
    epochs=100,            # Number of epochs
    batch=8,               # Adjust batch size as needed
    imgsz=900,             # Image size
    name="pcb_detection",  # Name of the run
    device=0               # Use GPU (Colab GPU runtime)
)

# ===== Evaluate Model =====
# Load the best model after training
best_model_path = "runs/detect/pcb_detection/weights/best.pt"
model = YOLO(best_model_path)

# Test on validation set
results = model.val()

# ===== Save Model =====
# Save trained weights to Google Drive
# Setting locale to UTF-8 to avoid NotImplementedError
import locale
def getpreferredencoding(do_setlocale = True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding

!cp {best_model_path} /content/drive/My\ Drive/data
print("Best model saved to Google Drive!")