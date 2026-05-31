from config import LOG_PATH
import csv
from datetime import datetime

class EmotionLogger:
    def __init__(self):
        self.file_path = LOG_PATH

        # create file if not exists
        try:
            with open(self.file_path, "x", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["time", "emotion"])
        except FileExistsError:
            pass

    def log(self, emotion):
        now = datetime.now().strftime("%H:%M:%S")

        with open(self.file_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([now, emotion])