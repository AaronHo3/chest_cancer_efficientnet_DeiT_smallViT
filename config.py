"""
Configuration for chest cancer CT classification.
Adjust paths after downloading the dataset (e.g. DATA_ROOT = path from kagglehub).
"""
import os

# Dataset: from kagglehub download (override with env DATASET_PATH)
DATA_ROOT = os.environ.get(
    "DATASET_PATH",
    os.path.expanduser("~/.cache/kagglehub/datasets/mohamedhanyyy/chest-ctscan-images/versions/1"),
)

# 4 classes: 3 NSCLC types + normal (folder names may vary; normalize in dataset to these labels)
CLASS_NAMES = [
    "adenocarcinoma",
    "large_cell_carcinoma",
    "squamous_cell_carcinoma",
    "normal",
]
NUM_CLASSES = len(CLASS_NAMES)

# Image size (EfficientNet-B2, DeiT-Tiny use 224)
IMG_SIZE = 224

# Model: "efficientnet_b2" | "deit_tiny" (small ViT, MacBook-friendly)
MODEL = os.environ.get("MODEL", "deit_tiny")

# Training
BATCH_SIZE = 32
NUM_WORKERS = 4
LR = 1e-4
WEIGHT_DECAY = 1e-2
EPOCHS = 30
EARLY_STOP_PATIENCE = 5

# Output
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"
