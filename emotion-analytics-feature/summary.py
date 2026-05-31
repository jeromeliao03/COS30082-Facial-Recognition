import pandas as pd
import matplotlib.pyplot as plt
from config import LOG_PATH

def generate_report():
    df = pd.read_csv(LOG_PATH, header=None)
    df.columns = ["time", "emotion"]

    print(df.head())

    counts = df["emotion"].value_counts()

    counts.plot(kind="bar")
    plt.title("Emotion Distribution")
    plt.xlabel("Emotion")
    plt.ylabel("Count")

    plt.show()

if __name__ == "__main__":
    generate_report()