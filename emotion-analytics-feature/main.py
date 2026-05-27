import sys
import os
import csv
import time
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

sys.path.insert(0, PROJECT_ROOT)

from logger import EmotionLogger
from emotion.src.predict import EmotionPredictor

# -----------------------------
# INIT LOGGER
# -----------------------------
logger = EmotionLogger("emotion_log.csv")

# -----------------------------
# INIT MODEL (IMPORTANT)
# -----------------------------
predictor = EmotionPredictor(
    weights_path="../models/your_model.h5",  # change if needed
    input_is_bgr=True
)

# -----------------------------
# TEMP FAKE FACE (until webcam added)
# -----------------------------
def get_fake_face():
    import numpy as np
    return np.zeros((224, 224, 3), dtype=np.uint8)

# -----------------------------
# MAIN LOOP
# -----------------------------
emotions = ["happy", "sad", "neutral", "surprise"]

while True:

    # TEMP: replace with real webcam later
    face_crop = get_fake_face()

    # REAL MODEL PREDICTION
    result = predictor.predict(face_crop)
    label = result["label"]

    # LOG YOUR FEATURE
    logger.log(label)

    time.sleep(1)