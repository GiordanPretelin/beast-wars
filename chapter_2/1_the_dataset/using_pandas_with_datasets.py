import pandas as pd
from datasets import load_dataset

emotions = load_dataset("emotion")

emotions.set_format(type="pandas")
df = emotions["train"][:]
# using the int2str() method of the label feature to create a new column in our DataFrame with the corresponding label names


def label_int2str(row):
    return emotions["train"].features["label"].int2str(row)


df["label_name"] = df["label"].apply(label_int2str)

print(df.head())
