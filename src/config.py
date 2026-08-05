"""
Configuration file for FruitVision AI
"""

from pathlib import Path

# ==========================
# Project Paths
# ==========================

ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = ROOT / "dataset"

TRAIN_DIR = DATASET_DIR / "Training"

TEST_DIR = DATASET_DIR / "Test"

MODEL_DIR = ROOT / "models"

MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "fruit_classifier.keras"

# ==========================
# Image Settings
# ==========================

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 32

CHANNELS = 3

# ==========================
# Training
# ==========================

EPOCHS = 10

LEARNING_RATE = 0.0001

VALIDATION_SPLIT = 0.2

SEED = 42

# ==========================
# Class Labels
# ==========================

CLASS_NAMES = None
