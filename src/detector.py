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
SPOOF_PAD = 0.4
MIN_FACE_SIZE = config["detection"]["min_face_size"]

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
        # skip small detections — usually background false positives
        if fw < MIN_FACE_SIZE or fh < MIN_FACE_SIZE:
            continue
        # clamp
        x,y = max(0, x), max(0, y)
        crop = frame[y:y+fh, x:x+fw]
        if crop.size == 0:
            continue

        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = cv2.resize(crop, IMG_SIZE).astype(np.float32)

        # Add padding to any input for spoofing model
        px = int(fw * SPOOF_PAD)
        py = int(fh * SPOOF_PAD)
        sx1 = max(0, x - px);  sy1 = max(0, y - py)
        sx2 = min(w, x + fw + px); sy2 = min(h, y + fh + py)
        spoof_region = frame[sy1:sy2, sx1:sx2]
        spoof_crop = cv2.cvtColor(spoof_region, cv2.COLOR_BGR2RGB)
        spoof_crop = cv2.resize(spoof_crop, IMG_SIZE).astype(np.float32)

        result.append({"box": (x,y,fw,fh), "raw_crop": crop, "spoof_crop": spoof_crop})

    return result