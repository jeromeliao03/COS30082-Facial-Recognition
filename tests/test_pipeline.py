import sys
import os
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import Pipeline

pipeline = Pipeline()
pipeline.start()

print("Press Q to quit")

while True:
    jpeg = pipeline.get_frame()
    if jpeg:
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        cv2.imshow("Pipeline Output", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

pipeline.stop()
cv2.destroyAllWindows()