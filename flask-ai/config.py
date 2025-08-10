# config.py - Stateless Version
"""
Stateless Configuration file for Unified Defect Detection System
No file operations, only in-memory configuration
"""

import os
from pathlib import Path

# Base paths (for model loading only, no outputs)
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

# Model paths (only for loading models)
ANOMALIB_MODEL_PATH = MODELS_DIR / "patchcore.pt"
HRNET_MODEL_PATH = MODELS_DIR / "defect_segmentation_model.pth"

# Detection thresholds (can be updated in-memory)
ANOMALY_THRESHOLD = 0.7
DEFECT_CONFIDENCE_THRESHOLD = 0.85

# Device configuration
DEVICE = 'cuda'  # or 'cpu'

# Enhanced defect detection classes
SPECIFIC_DEFECT_CLASSES = {
    0: "background",
    1: "damaged",
    2: "missing_component", 
    3: "open",
    4: "scratch",
    5: "stained"
}

DEFECT_COLORS = {
    0: (64, 64, 64),     # Dark gray - background
    1: (255, 0, 0),      # Red - damaged
    2: (255, 255, 0),    # Yellow - missing component
    3: (255, 0, 255),    # Magenta - open
    4: (0, 255, 255),    # Cyan - scratch
    5: (128, 0, 128),    # Purple - stained
}

# Image preprocessing settings
IMAGE_SIZE = (512, 512)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Minimum detection thresholds
MIN_DEFECT_PIXELS = 50
MIN_DEFECT_PERCENTAGE = 0.005  # 0.5%
MIN_BBOX_AREA = 100

# Create only models directory (for loading models)
os.makedirs(MODELS_DIR, exist_ok=True)

# Stateless mode - no output directories created
STATELESS_MODE = True

# Debug: Print actual paths for verification
if __name__ == "__main__":
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"MODELS_DIR: {MODELS_DIR}")
    print(f"ANOMALIB_MODEL_PATH: {ANOMALIB_MODEL_PATH}")
    print(f"HRNET_MODEL_PATH: {HRNET_MODEL_PATH}")
    print(f"STATELESS_MODE: {STATELESS_MODE}")
    print(f"Model files exist:")
    print(f"  Anomalib: {ANOMALIB_MODEL_PATH.exists()}")
    print(f"  HRNet: {HRNET_MODEL_PATH.exists()}")