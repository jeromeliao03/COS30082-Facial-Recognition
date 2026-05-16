"""Central config. Everything tunable lives here so the notebook stays clean
and Sani's ResNet18 run can mirror the exact same hyperparameters."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths. Defaults assume Colab + Drive layout:
#   /content/drive/MyDrive/MLGroup/emotion/
#       data/fer2013/train/<class>/*.jpg
#       data/fer2013/test/<class>/*.jpg
#       models/
# Override DRIVE_ROOT from the notebook if you stored things elsewhere.
# ---------------------------------------------------------------------------
DRIVE_ROOT = Path("/content/drive/MyDrive/MLGroup/emotion")
DATA_DIR = DRIVE_ROOT / "data" / "fer2013"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
MODELS_DIR = DRIVE_ROOT / "models"
REPORTS_DIR = DRIVE_ROOT / "reports"

# ---------------------------------------------------------------------------
# Classes. Order is fixed and must match the ImageFolder sort order
# (alphabetical) so label indices stay consistent across train/eval/predict.
# ---------------------------------------------------------------------------
CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NUM_CLASSES = len(CLASSES)

# ---------------------------------------------------------------------------
# Model / training hyperparameters.
# Defaults chosen for FER-2013 + MobileNetV2 on a Colab T4 (~15 GB VRAM).
# ---------------------------------------------------------------------------
IMG_SIZE = 224          # MobileNetV2 ImageNet-pretrained expects 224x224.
BATCH_SIZE = 64
NUM_WORKERS = 2         # Colab gives 2 vCPUs on free tier; bump to 4 on Pro.

EPOCHS = 25
LR_HEAD = 1e-3          # New classifier head learns faster.
LR_BACKBONE = 1e-4      # Last block fine-tunes more slowly.
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05  # Helps with FER-2013's noisy labels.

# Fine-tuning strategy: freeze everything, unfreeze only the last N
# inverted-residual blocks of MobileNetV2's `features` sequence. There are
# 19 entries (features[0..18]); 4 keeps params reasonable and accuracy good.
UNFREEZE_LAST_N_BLOCKS = 4

VAL_SPLIT = 0.10        # Carve 10% of train as validation.
SEED = 42

# ImageNet normalisation stats - pretrained weights expect these exact values.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
