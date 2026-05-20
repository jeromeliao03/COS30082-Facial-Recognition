"""
Facial Detector module
"""

import cv2
import numpy as np
import keras
import yaml

with open("config.yaml") as f:
    config = yaml.safe_load(f)

detector = cv2.CascadeClassifier(config['models']['haar_cascade'])
preprocess = keras.applications.mobilenet_v2.preprocess_input
IMG_SIZE = tuple(config["recognition"]["img_size"])

def detect_faces(frame):
    # current model (cascade) requires grey
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    boxes = detector.detectMultiScale(gray, minSize=(60,60))

    faces = []

    for (x,y,w,h) in boxes:
        crop = frame[y:y+h, x:x+w]
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = cv2.resize(crop, IMG_SIZE).astype(np.float32)
        crop = preprocess(crop)
        faces.append({"box": (x,y,w,h), "crop": crop})

    return faces
