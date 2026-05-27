"""
Inference module: runs the model pipeline on detected face.
"""

import numpy as np
import keras
import yaml
import tensorflow as tf

from src.registry import search
from src import attendance_log

with open ("config.yaml", "r") as f:
    config = yaml.safe_load(f)


THRESHOLD = config["recognition"]["threshold"]
EMOTION_LABELS = config["emotions"]["labels"]

_active = config["recognition"]["active_model"]
_base_model = keras.models.load_model(config["models"][f"face_recognition_{_active}"])

_embedding_model = keras.Model(
    inputs=_base_model.input,
    outputs=_base_model.get_layer("embedding").output
)

_spoof_model = keras.models.load_model(config["models"]["anti_spoofing"])
_emotion_model = keras.models.load_model(config["models"]["emotion"])

_preprocess = keras.applications.mobilenet_v2.preprocess_input

# Trace once, skip Python overhead on subsequent calls
@tf.function
def _run_embedding(batch):
    return _embedding_model(batch, training=False)

@tf.function
def _run_spoof(batch):
    return _spoof_model(batch, training=False)

@tf.function
def _run_emotion(batch):
    return _emotion_model(batch, training=False)

# Warm up all three models so XLA compiles before first real frame
_dummy = np.zeros((1, 128, 128, 3), dtype="float32")
_run_embedding(_dummy)
_run_spoof(_dummy)
_run_emotion(_dummy)

_liveness_cache = {}

def run_inference(crop):
    """
    Run inference pipeline on single preprocessed face crop
    Gating for actual model usage to reduce compute
    FOR NOW, gated at if identity exists

    TO:DO: Bake further conditionals to discriminate against using all models on all frames, 
    Example: If a face is determined to not be spoofed, 
        the track of that face going forward will not need subsequent detections
    """
    batch = np.expand_dims(_preprocess(crop.astype("float32")), axis=0)

    # 1. Entry is face recognition to establish known identity
    embedding = _run_embedding(batch).numpy()[0]
    match = search(embedding)
    
    if match is None:
        return{"identity": None}
    
    name = match["name"]

     # ckecking system lockout status before running models
    if attendance_log.is_locked():
        return { "identity": None, "locked": True, "message": "System locked due to multiple spoofing attempts. Please wait." }

    # 2. anti-spoofing, run if identity is matched
    if name not in _liveness_cache:
        spoof_score = _run_spoof(batch).numpy()[0]
        _liveness_cache[name] = bool(spoof_score[0] > 0.5)

    liveness = _liveness_cache[name]

    # 3. emotion, run if identity is matched
    emotion_prob = _run_emotion(batch).numpy()[0]
    emotion = EMOTION_LABELS[int(np.argmax(emotion_prob))]

    # innovative feature — log attendance 
    attendance_log.process(match['name'], liveness, emotion)

    return {
        "identity": match['name'],
        "distance": float(match['distance']),
        "liveness": liveness,
        "emotion": emotion,
    }

