"""Smoke test for pipeline"""

import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import Pipeline

pipeline = Pipeline()
pipeline.start()

# Give the pipeline time to capture and process at least one frame
time.sleep(8)

frame = pipeline.get_frame()
assert frame is not None, "No frame received — check webcam is connected"
assert isinstance(frame, bytes), "Frame should be JPEG bytes"
assert len(frame) > 0, "Frame is empty"

print(f"smoke test: PASSED — received frame ({len(frame)} bytes)")

pipeline.stop()