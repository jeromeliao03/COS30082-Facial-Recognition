"""Central config. All tunables live here so the notebook stays clean."""

from pathlib import Path

# Paths assume Colab + Drive layout. Override DRIVE_ROOT if stored elsewhere.
DRIVE_ROOT = Path("/content/drive/MyDrive/MLGroup/emotion")
DATA_DIR = DRIVE_ROOT / "data" / "fer2013"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
MODELS_DIR = DRIVE_ROOT / "models"
REPORTS_DIR = DRIVE_ROOT / "reports"

# Class order must match the alphabetical order flow_from_directory produces.
CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NUM_CLASSES = len(CLASSES)

# MobileNetV2 was pretrained on 224x224 RGB images at the largest variant.
IMG_SIZE = 224
BATCH_SIZE = 64

# Two-phase training: head only first, then unfreeze backbone tail.
EPOCHS_PHASE1 = 8
EPOCHS_PHASE2 = 17
LR_PHASE1 = 1e-3
LR_PHASE2 = 1e-5
UNFREEZE_LAST_N_LAYERS = 30
DROPOUT = 0.3

# Mild label smoothing to handle FER-2013's noisy labels.
LABEL_SMOOTHING = 0.05

VAL_SPLIT = 0.10
SEED = 42
