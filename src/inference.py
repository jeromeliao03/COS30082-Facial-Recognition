"""
Inference module: runs the model pipeline on detected face.
"""

import numpy as np
import keras
import yaml

from registry import search

with open ("config.yaml", "r") as f:
    config = yaml.safe_load(f)


THRESHOLD = config["recognition"]["threshold"]
EMOTION_LABELS = config["emotions"]["labels"]

_base_model_metric = keras.models.load_model(config["model"]["face_recognition_metric"])
_embedding_model = keras.Model(
    inputs=_base_model_metric.input,
    outputs=_base_model_metric.get_layer("embedding").output
)

_spoof_model = keras.models.load_model(config["model"]["anti_spoofing"])
_emotion_model = keras.models.load_model(config["model"]["emotion"])

def run_inference(crop):
    """
    Run inference pipeline on single preprocessed face crop
    Gating for actual model usage to reduce compute
    FOR NOW, gated at if identity exists

    TO:DO: Bake further conditionals to discriminate against using all models on all frames, 
    Example: If a face is determined to not be spoofed, 
        the track of that face going forward will not need subsequent detections
    """
    batch = np.expand_dums(crop, axis=0)

    # 1. Entry is face recognition to establish known identity
    embedding = _embedding_model.predict(batch, verbose=0)[0]
    match = search(embedding)
    
    if match is None:
        return{"identity": None}
    
    # 2. anti-spoofing, run if identity is matched
    spoof_score = _spoof_model.predict(batch, verbose=0)[0]
    liveness = bool(spoof_score > 0.5)

    # 3. emotion, run if identity is matched
    emotion_prob = _emotion_model.predict(batch, verbose=0)[0]
    emotion = EMOTION_LABELS[int(np.argmax(emotion_prob))]

    return {
        "identity": match['name'],
        "distance": float(match['distance']),
        "liveness": liveness,
        "emotion": emotion,
    }

