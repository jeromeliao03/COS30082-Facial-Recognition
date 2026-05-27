import csv
from datetime import datetime

class EmotionLogger:

    def __init__(self, file_path="emotion_log.csv"):
        self.file_path = file_path

    def log(self, label):
        current_time = datetime.now().strftime("%H:%M:%S")

        with open(self.file_path, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([current_time, label])

        print(f"Logged: {label}")