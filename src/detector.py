"""
Facial Detector module: find faces in a raw BGR frame using Haar Cascade,
return cropped, preprocessed face regions ready for model input.
"""
import cv2
import numpy as np
import keras
import yaml

with open("config.yaml") as f:
    config = yaml.safe_load(f)

detector = cv2.CascadeClassifier(config['models']['haar_cascade'])
IMG_SIZE = tuple(config["recognition"]["img_size"])

def detect_faces(frame):
    """Detect faces in BGR frame, returns box and crop dicts"""
    
    # current model (cascade) requires grey
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    boxes = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60))

    faces = []

    for (x,y,w,h) in boxes:
        crop = frame[y:y+h, x:x+w]
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = cv2.resize(crop, IMG_SIZE).astype(np.float32)
        faces.append({"box": (x,y,w,h), "raw_crop": crop})

    return faces
