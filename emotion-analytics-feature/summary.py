import pandas as pd
import matplotlib.pyplot as plt

def generate_report(file_path="emotion_log.csv"):
    df = pd.read_csv(file_path, header=None)

    df.columns = ["time", "emotion"]

    print(df.head())
    print("Columns:", df.columns)

    counts = df["emotion"].value_counts()

    counts.plot(kind="bar")
    plt.title("Emotion Distribution")
    plt.xlabel("Emotion")
    plt.ylabel("Count")

    plt.show()

if __name__ == "__main__":
    generate_report()