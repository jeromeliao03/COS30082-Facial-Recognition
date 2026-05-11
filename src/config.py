"""
config.py  (TensorFlow / Keras baseline — Hasi branch)
------------------------------------------------------
Mirrors the PyTorch teammate's src/config.py so the two baselines
can be compared apples-to-apples on the same dataset, same hyperparameters,
same evaluation protocol — only the framework changes.

Designed to run on Google Colab. Typical Colab setup:

    # 1) Clone the repo on your branch
    !git clone -b hasi https://github.com/<user>/COS30082-Facial-Recognition.git
    %cd COS30082-Facial-Recognition

    # 2) Mount Drive (preferred — dataset stays put between sessions)
    from google.colab import drive
    drive.mount('/content/drive')
    # Then set DATA_ROOT below to your Drive path.

    # 3) Install the optional ResNet18 backbone (only if you switch BACKBONE_NAME)
    # !pip install image-classifiers

    # 4) Run training
    %cd src
    !python train.py
"""

import os
import tensorflow as tf


# ---------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------
# Adjust DATA_ROOT to wherever you placed the dataset.
# Common Colab choices:
#   "/content/data"                                      (uploaded + unzipped)
#   "/content/drive/MyDrive/COS30082/data"               (Drive mount)
#
# The notebook can override this without editing the file by setting the
# DATA_ROOT environment variable BEFORE importing config:
#     os.environ["DATA_ROOT"] = "/content/drive/MyDrive/..."
DATA_ROOT = os.environ.get("DATA_ROOT", "/content/data")

TRAIN_DIR          = os.path.join(DATA_ROOT, "classification_data", "train_data")
VAL_DIR            = os.path.join(DATA_ROOT, "classification_data", "val_data")
TEST_DIR           = os.path.join(DATA_ROOT, "classification_data", "test_data")
VERIFICATION_DIR   = os.path.join(DATA_ROOT, "verification_data")
VERIFICATION_PAIRS = os.path.join(DATA_ROOT, "verification_pairs_val.txt")

# Output dirs (mirror the PyTorch project layout)
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "output")


# ---------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------
GPU_DEVICES = tf.config.list_physical_devices("GPU")
DEVICE = "GPU" if GPU_DEVICES else "CPU"

# Allow GPU memory to grow on demand — avoids TF grabbing the full T4 up front,
# which matters in Colab where other notebooks may share the device.
for _gpu in GPU_DEVICES:
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError:
        pass


# ---------------------------------------------------------------
# Image settings (kept identical to PyTorch baseline for fair comparison)
# ---------------------------------------------------------------
IMAGE_SIZE    = (224, 224)
EMBEDDING_DIM = 512

# Normalisation: pixel values mapped to [-1, 1] — same as PyTorch's
# transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]).
# We deliberately do NOT use tf.keras.applications.resnet.preprocess_input
# (which does Caffe-style BGR mean subtraction) — keeping normalisation
# identical between branches matters more than matching the backbone's
# pretraining recipe.
NORMALISE_MEAN = 127.5
NORMALISE_STD  = 127.5


# ---------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------
# tf.keras.applications ships ResNet50/101/152 (V1+V2) but NOT ResNet18.
#
# Options:
#   "ResNet50"  — Keras built-in, ImageNet weights, recommended default.
#                 For a strictly fair comparison the PyTorch teammate would
#                 also switch to ResNet50.
#   "ResNet18"  — requires `pip install image-classifiers` (community port).
#                 Matches the PyTorch baseline exactly.
BACKBONE_NAME = "ResNet50"


# ---------------------------------------------------------------
# Classification training hyperparameters
# ---------------------------------------------------------------
CLF_EPOCHS          = 30
CLF_BATCH_SIZE      = 64
CLF_LEARNING_RATE   = 1e-4
CLF_WEIGHT_DECAY    = 1e-5
CLF_UNFREEZE_LAYERS = 2   # number of trailing backbone layers to unfreeze.
                          # ResNet50 in Keras has ~175 layers — last 2 unfreezes
                          # only the final conv block tail. Increase (e.g. 20–40)
                          # if you find the model under-fits.


# ---------------------------------------------------------------
# Metric learning hyperparameters (kept here for parity / future use)
# ---------------------------------------------------------------
TRIPLET_EPOCHS        = 20
TRIPLET_BATCH_SIZE    = 64
TRIPLET_LEARNING_RATE = 1e-4
TRIPLET_WEIGHT_DECAY  = 1e-5
TRIPLET_MARGIN        = 1.0
TRIPLET_PER_EPOCH     = 1000


# ---------------------------------------------------------------
# Misc
# ---------------------------------------------------------------
SEED     = 42
AUTOTUNE = tf.data.AUTOTUNE
