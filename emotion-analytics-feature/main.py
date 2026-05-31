import sys
import os
import csv
import cv2
import time
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

sys.path.insert(0, PROJECT_ROOT)

from logger import EmotionLogger
from emotion.src.predict import EmotionPredictor


from logger import EmotionLogger

logger = EmotionLogger()

# -----------------------------
# INIT MODEL (IMPORTANT)
# -----------------------------
predictor = EmotionPredictor(
    weights_path="emotion/models/mobilenetv2_best.keras",
    input_is_bgr=True
)


cap = cv2.VideoCapture(0)

# -----------------------------
# MAIN LOOP
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    face_crop = cv2.resize(frame, (224, 224))

    result = predictor.predict(face_crop)
    label = result["label"]

    logger.log(label)

    print("Logged:", label)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()