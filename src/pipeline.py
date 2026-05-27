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
from src.inference import _liveness_cache


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
        self._frame_count = 0
        self._last_results = []

    def start(self):
        """Start background capture thread"""
        self._thread.start()

    def stop(self):
        """Sigterm thread"""
        self._stop_event.set()

    def get_frame(self):
        """Return the latest JPEG bytes, or None if nothing available."""
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None
        
    def get_crop(self):
        """Return raw crop of first detected face, or None."""
        if not self._last_results:
            return None
        return self._last_results[0].get("crop")

    def peek_frame(self):
        """Return latest JPEG bytes without consuming from queue."""
        try:
            jpeg = self._queue.get_nowait()
            self._queue.put(jpeg)
            return jpeg
        except queue.Empty:
            return None
        
    def status(self):
        """Return (as a flat object) status components for frontend panel"""
        faces = self._last_results
        if not faces:
            return {"face_detected": False, "name": None, "emotion": None, "liveness": None}

        first = faces[0]["result"]
        liveness_str = None
        if first.get("liveness") is True:
            liveness_str = "Live"
        elif first.get("liveness") is False:
            liveness_str = "Spoof"

        return {
            "face_detected": True,
            "name": first.get("identity") or "Unknown",
            "emotion": first.get("emotion"),
            "liveness": liveness_str,
        }

    def _run(self):
        """Main loop - runs on background thread."""
        cap = cv2.VideoCapture(self._camera_index)
        try:
            while not self._stop_event.is_set():
                try:
                    ret, frame = cap.read()
                    if not ret or frame is None or frame.size == 0:
                        continue

                    self._frame_count += 1
                    faces = detect_faces(frame)

                    # Run inference every 3rd frame, reuse cached results in between
                    if faces and self._frame_count % 3 == 0:
                        annotated_faces = []
                        for face in faces:
                            try:
                                result = run_inference(face["raw_crop"])
                            except Exception as e:
                                print(f"inference error: {e}")
                                continue
                            annotated_faces.append({"box": face["box"], "result": result, "crop": face["raw_crop"]})
                        self._last_results = annotated_faces

                    elif faces:
                        # Reuse last inference results, update box n crop
                        self._last_results = [
                            {"box": faces[i]["box"], "result": self._last_results[i]["result"], "crop": faces[i]["raw_crop"]}
                            for i in range(min(len(faces), len(self._last_results)))
                        ]

                    else:
                        # No faces detected — clear cached results
                        self._last_results = []
                        _liveness_cache.clear()

                    rendered = annotate_frame(frame, self._last_results)
                    _, jpeg = cv2.imencode(".jpg", rendered)

                    if self._queue.full():
                        try:
                            self._queue.get_nowait()
                        except queue.Empty:
                            pass

                    self._queue.put(jpeg.tobytes())

                except Exception as e:
                    print(f"pipeline loop error: {e}")
                    continue
        finally:
            cap.release()
