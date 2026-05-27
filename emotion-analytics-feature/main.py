from datetime import datetime
import csv
import time

emotions = ["happy", "sad", "neutral", "surprise"]

for label in emotions:

    current_time = datetime.now().strftime("%H:%M:%S")

    with open('emotion_log.csv', 'a', newline='') as file:
        writer = csv.writer(file)

        writer.writerow([current_time, label])

    print(f"Logged: {label}")

    time.sleep(1)