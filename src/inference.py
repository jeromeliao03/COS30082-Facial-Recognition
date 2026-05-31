"""
Inference module: runs the model pipeline on detected face.
"""

import numpy as np
import keras
import yaml
import tensorflow as tf

from src.registry import search
from src import attendance_log
from src import greeter
from src import tracker

with open ("config.yaml", "r") as f:
    config = yaml.safe_load(f)


THRESHOLD = config["recognition"]["threshold"]
SPOOF_THRESHOLD = config["recognition"]["spoof_threshold"]
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
_attendance_fired = set()


def evict(name):
    """Remove all cached state for a deleted identity."""
    _liveness_cache.pop(name, None)
    _attendance_fired.discard(name)
    attendance_log.evict(name)
    tracker.clear_all()

def run_inference(crop, spoof_crop, slot=0):
    """
    Run inference pipeline on single preprocessed face crop, gated by temporal aggregate
    """
    batch = np.expand_dims(_preprocess(crop.astype("float32")), axis=0)

    # 1. face recognition, runs every inference frame to cast a tracker vote
    embedding = _run_embedding(batch).numpy()[0]
    match = search(embedding)

    name = match["name"] if match is not None else None

    # accumulate votes (including None for no-match) so the window stays full
    # and sporadic matches on an unregistered face can't reach confirmation
    tracker.observe(slot, name)
    tracking = tracker.query(slot)

    # Confirmed unknown — stop processing, do not attempt identity pipeline
    if tracking["state"] == "unknown":
        return {"identity": None}

    if match is None or tracking["state"] != "confirmed":
        return {
            "identity": name,
            "tracking": "collecting",
            "progress": tracking["progress"],
        }

    # use the tracker's consensus name, not the current frame's recognition result 4 display
    name = tracking["name"]

    # 2. lockout check — only matters once identity is confirmed
    if attendance_log.is_locked():
        return {"identity": None, "locked": True, "message": "System locked due to multiple spoofing attempts. Please wait."}

    # 3. anti-spoofing — runs once per appearance (before attendance fires)
    # True is cached permanently, False is never cached so a bad-angle check can be retried
    # Force 15 good frames
    if name not in _liveness_cache and name not in _attendance_fired:
        spoof_batch = np.expand_dims(spoof_crop / 255.0, axis=0)
        spoof_score = _run_spoof(spoof_batch).numpy()[0]
        liveness = bool(spoof_score[0] >= SPOOF_THRESHOLD)
        print(f"[slot {slot}] CONFIRMED {name} | spoof_score={spoof_score[0]:.4f} liveness={liveness}")
        if liveness:
            _liveness_cache[name] = True

    liveness = _liveness_cache.get(name, False)

    # 4. emotion
    emotion_prob = _run_emotion(batch).numpy()[0]
    emotion = EMOTION_LABELS[int(np.argmax(emotion_prob))]
    # innovative feature — log attendance
    # attendance fires once per confirmed name per appearance
    if name not in _attendance_fired:
        attendance_log.process(name, liveness, emotion)
        _attendance_fired.add(name)

    # innovation feature — confidence-weighted emotion aggregation for greeting
    # Only feed live, recognised faces; spoofs don't earn personalised greetings.
    if liveness:
        greeter.observe(name, emotion_prob)

    return {
        "identity": name,
        "distance": float(match['distance']),
        "liveness": liveness,
        "emotion": emotion,
    }