import cv2
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.detector import detect_faces

frame = cv2.imread("tests/test_detector_input.png")
faces = detect_faces(frame)

print(f"Found {len(faces)} face(s)")
for f in faces:
    x, y, w, h = f["box"]
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imwrite("tests/test_output.jpg", frame)