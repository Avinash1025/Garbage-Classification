import os

# Dataset paths
RAW_DATA_DIR = os.path.join("data", "raw")
PROCESSED_DATA_DIR = os.path.join("data", "processed")

# Image and training config
IMAGE_SIZE = (224, 224)  # EfficientNetB0 default
BATCH_SIZE = 32
SEED = 42
