"""
Facial Detector module: find faces in a raw BGR frame using YuNet,
return cropped, preprocessed face regions ready for model input.
"""
import cv2
import numpy as np
import yaml

with open("config.yaml") as f:
    config = yaml.safe_load(f)

detector = cv2.FaceDetectorYN.create(
    config["models"]["yunet"], "", (300, 300), score_threshold=0.6
)

IMG_SIZE = tuple(config["recognition"]["img_size"])

def detect_faces(frame):
    """Detect faces in BGR frame, returns box and crop dicts"""
    # Not size agnostic like haar cascade, explcitly call
    h, w = frame.shape[:2]
    detector.setInputSize((w,h))
    _, faces = detector.detect(frame)

    if faces is None:
        return []

    result = []
    for face in faces:
        # take first 4 from box coord, keypoints packed by YuNet
        x,y,fw,fh = [int(v) for v in face[:4]]
        # clamp
        x,y = max(0, x), max(0, y)
        crop = frame[y:y+fh, x:x+fw]
        if crop.size == 0:
            continue
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = cv2.resize(crop, IMG_SIZE).astype(np.float32)
        result.append({"box": (x,y,fw,fh), "raw_crop": crop})

    return result