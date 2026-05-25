"""
Pipeline Module: orchestration pipeline that powers webcam loop
Produces annotated JPEG frames consumed by FastAPI stream.
Operates on background thread, FastAPI will be primary
"""

import cv2
import threading
import queue
import yaml

from src.detector import detect_faces
from src.inference import run_inference
from src.display import annotate_frame


with open("config.yaml") as f:
    config = yaml.safe_load(f)

class Pipeline:
    """
    Webcam capture and inference loop, on background thread
    """

    def __init__(self, max_queue=2):
        self._camera_index = config["camera_index"]
        self._queue = queue.Queue(maxsize=max_queue)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        """Start background capture thread"""
        self._thread.start()

    def stop(self):
        """Sigterm thread"""
        self._stop_event.set()

    def get_frame(self):
        """Return the latest JPEG bytes"""
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None
        
    def _run(self):
        """Main loop - runs on background thread."""
        cap = cv2.VideoCapture(self._camera_index)
        try:
          while not self._stop_event.is_set():
              try:
                  ret, frame = cap.read()
                  if not ret or frame is None or frame.size == 0:
                      continue

                  faces = detect_faces(frame)

                  annotated_faces = []
                  for face in faces:
                      try:
                          result = run_inference(face["raw_crop"])
                      except Exception as e:
                          print(f"inference error: {e}")
                          continue
                      annotated_faces.append({"box": face["box"], "result": result})

                  rendered = annotate_frame(frame, annotated_faces)

                  _, jpeg = cv2.imencode(".jpg", rendered)

                  if self._queue.full():
                      try:
                          self._queue.get_nowait()
                      except queue.Empty:
                          pass

                  self._queue.put(jpeg.tobytes())

              except Exception as e:
                  # Log but never kill the thread
                  print(f"pipeline loop error: {e}")
                  continue
        finally:
            cap.release()