"""Tests for inference pipeline."""
import sys
import os
import numpy as np
import cv2

sys.path.insert(0,
os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.registry as registry
from src.detector import detect_faces
from src.inference import run_inference, _embedding_model, _preprocess

# Redirect registry to test file
registry.REGISTRY_PATH = "tests/test_registry.npz"

EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral",
"sad", "surprise"]

# --- Setup: register a face from test image ---
frame = cv2.imread("tests/test_face.jpg")
faces = detect_faces(frame)
assert len(faces) > 0, "No face detected — check test image"

crop = faces[0]["raw_crop"]
batch = np.expand_dims(_preprocess(crop.astype("float32")), axis=0)
embedding = _embedding_model.predict(batch, verbose=0)[0]
registry.register("Test Person", embedding)


# --- Test 1: known face returns full result ---
result = run_inference(crop)
assert result["identity"] == "Test Person", f"Expected match, got: {result}"
assert isinstance(result["liveness"], bool), "Liveness should be bool"
assert result["emotion"] in EMOTION_LABELS, f"Unexpected emotion: {result['emotion']}"
assert 0.0 <= result["distance"] <= 1.0, "Distance out of expected range"
print(f"full pipeline: PASSED — identity={result['identity']}, liveness={result['liveness']}, emotion={result['emotion']}")


# --- Test 2: unregistered face returns None ---
registry.names.clear()
registry.embeddings.clear()
result = run_inference(crop)
assert result["identity"] is None, "Expected no match on empty registry"
print("gate logic: PASSED")


# --- Cleanup ---
if os.path.exists("tests/test_registry.npz"):
    os.remove("tests/test_registry.npz")

print("All inference tests passed.")