# config.py - Enhanced with OpenAI Support (TESTING MODE)
"""
Configuration file for Unified Defect Detection System with OpenAI integration
TESTING MODE: Lower thresholds for better sensitivity
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

# Model paths
ANOMALIB_MODEL_PATH = MODELS_DIR / "patchcore.pt"
HRNET_MODEL_PATH = MODELS_DIR / "defect_segmentation_model.pth"

# Detection thresholds - LOWERED FOR TESTING
ANOMALY_THRESHOLD = 0.3  # Lowered from 0.7 - more sensitive
DEFECT_CONFIDENCE_THRESHOLD = 0.6  # Lowered from 0.85 - less strict

# Device configuration
DEVICE = 'cuda'

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
    0: (64, 64, 64),
    1: (255, 0, 0),
    2: (255, 255, 0),
    3: (255, 0, 255),
    4: (0, 255, 255),
    5: (128, 0, 128),
}

# Image preprocessing settings
IMAGE_SIZE = (512, 512)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Minimum detection thresholds - LOWERED FOR TESTING
MIN_DEFECT_PIXELS = 25  # Lowered from 50
MIN_DEFECT_PERCENTAGE = 0.002  # Lowered from 0.005
MIN_BBOX_AREA = 50  # Lowered from 100

# OpenAI Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL = "gpt-4-turbo"
OPENAI_MAX_TOKENS = 1000
OPENAI_TEMPERATURE = 0.1

# Create directories
os.makedirs(MODELS_DIR, exist_ok=True)

# Stateless mode
STATELESS_MODE = True

print("TESTING MODE: Using lowered thresholds for better sensitivity")
print(f"   ANOMALY_THRESHOLD: {ANOMALY_THRESHOLD}")
print(f"   DEFECT_CONFIDENCE_THRESHOLD: {DEFECT_CONFIDENCE_THRESHOLD}")
