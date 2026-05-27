import pandas as pd
import matplotlib.pyplot as plt

# load csv
df = pd.read_csv("emotion_log.csv", header=None)

df.columns = ["time", "emotion"]

# count emotions
counts = df["emotion"].value_counts()

print(counts)

# graph
counts.plot(kind='bar')

plt.xlabel("Emotion")
plt.ylabel("Count")
plt.title("Emotion Distribution")

plt.show()