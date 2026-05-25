"""Tests for display module."""

import sys
import os
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
"..")))

from src.detector import detect_faces
from src.display import annotate_frame

frame = cv2.imread("tests/test_face.jpg")
faces = detect_faces(frame)

assert len(faces) > 0, "No face detected — check test image"

# Test 1: matched, live, with emotion
faces_matched = [{"box": faces[0]["box"], "result": {"identity": "Test Person", "liveness": True, "emotion": "happy"}}]
output = annotate_frame(frame.copy(), faces_matched)
cv2.imwrite("tests/display_matched.jpg", output)
print("matched + live: PASSED — check tests/display_matched.jpg")

# Test 2: matched but spoofed
faces_spoof = [{"box": faces[0]["box"], "result": {"identity": "Test Person", "liveness": False, "emotion": "happy"}}]
output = annotate_frame(frame.copy(), faces_spoof)
cv2.imwrite("tests/display_spoof.jpg", output)
print("matched + spoof: PASSED — check tests/display_spoof.jpg")

# Test 3: unknown face
faces_unknown = [{"box": faces[0]["box"], "result": {"identity": None, "liveness": None, "emotion": None}}]
output = annotate_frame(frame.copy(), faces_unknown)
cv2.imwrite("tests/display_unknown.jpg", output)
print("unknown: PASSED — check tests/display_unknown.jpg")