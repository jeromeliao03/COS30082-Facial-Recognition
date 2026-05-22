"""Local training entry point for the emotion module.

Run from the repo root:
    python emotion/run_local.py

Reads the FER-2013 dataset from ~/fer2013 and writes all outputs to
~/emotion_outputs. Override either with environment variables:
    EMOTION_DATA_DIR=/some/path EMOTION_OUTPUT_DIR=/some/path python emotion/run_local.py
"""

import os
import sys
import json
from pathlib import Path

# Make `from emotion.src...` work regardless of where this script is run from.
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from emotion.src import config as C

# Resolve paths from env vars or fall back to ~/fer2013 and ~/emotion_outputs.
HOME = Path.home()
DATA_ROOT = Path(os.environ.get("EMOTION_DATA_DIR", HOME / "fer2013"))
OUTPUT_ROOT = Path(os.environ.get("EMOTION_OUTPUT_DIR", HOME / "emotion_outputs"))

C.DATA_DIR    = DATA_ROOT
C.TRAIN_DIR   = DATA_ROOT / "train"
C.TEST_DIR    = DATA_ROOT / "test"
C.MODELS_DIR  = OUTPUT_ROOT / "models"
C.REPORTS_DIR = OUTPUT_ROOT / "reports"
C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Sanity-check that the dataset is where expected.
if not (C.TRAIN_DIR / "happy").exists():
    sys.exit(
        f"FER-2013 not found at {C.DATA_DIR}.\n"
        f"Expected structure:\n"
        f"  {C.TRAIN_DIR}/<7 class folders>\n"
        f"  {C.TEST_DIR}/<7 class folders>\n"
        f"Download with:\n"
        f"  kaggle datasets download -d msambare/fer2013 -p {C.DATA_DIR} --unzip"
    )

import tensorflow as tf
print("TensorFlow:", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
print("GPUs visible:", gpus if gpus else "NONE - training will run on CPU and be very slow")

from emotion.src.data import build_dataloaders
from emotion.src.model import build_mobilenetv2, count_trainable
from emotion.src.train import train
from emotion.src.evaluate import (
    collect_predictions, metrics_report,
    plot_confusion_matrix, plot_training_curves,
)

# ---- Data -----------------------------------------------------------------
train_data, val_data, test_data = build_dataloaders()
print(f"train samples: {train_data.samples}  "
      f"val samples: {val_data.samples}  "
      f"test samples: {test_data.samples}")

# ---- Model ----------------------------------------------------------------
model, base_model = build_mobilenetv2()
trainable, total = count_trainable(model)
print(f"Phase 1 trainable params: {trainable:,} / {total:,} "
      f"({100*trainable/total:.2f}%)")

# ---- Train ----------------------------------------------------------------
history = train(model, base_model, train_data, val_data, tag="mobilenetv2")

plot_training_curves(
    history,
    phase1_epochs=C.EPOCHS_PHASE1,
    save_path=C.REPORTS_DIR / "training_curves.png",
)

# ---- Evaluate -------------------------------------------------------------
best = tf.keras.models.load_model(C.MODELS_DIR / "mobilenetv2_best.keras")
y_true, y_pred, _ = collect_predictions(best, test_data)
report = metrics_report(y_true, y_pred)
print("\nTest accuracy   :", round(report["accuracy"],   4))
print("Test macro-F1   :", round(report["macro_f1"],   4))
print("Test weighted-F1:", round(report["weighted_f1"], 4))
print()
print(report["classification_report"])

plot_confusion_matrix(
    report["confusion_matrix"],
    save_path=C.REPORTS_DIR / "confusion_matrix.png",
    normalize=True,
)

with open(C.REPORTS_DIR / "test_metrics.json", "w") as f:
    json.dump(
        {k: v for k, v in report.items() if k != "classification_report"},
        f, indent=2,
    )

print(f"\nAll outputs written to {OUTPUT_ROOT}")
print("Files to send back:")
print(f"  {C.MODELS_DIR / 'mobilenetv2_best.keras'}")
print(f"  {C.MODELS_DIR / 'mobilenetv2_history.json'}")
print(f"  {C.REPORTS_DIR / 'training_curves.png'}")
print(f"  {C.REPORTS_DIR / 'confusion_matrix.png'}")
print(f"  {C.REPORTS_DIR / 'test_metrics.json'}")
