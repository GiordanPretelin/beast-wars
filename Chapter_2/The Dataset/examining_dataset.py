import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset

emotions = load_dataset("emotion")

emotions.set_format(type="pandas")
df = emotions["train"][:]
# using the int2str() method of the label feature to create a new column in our DataFrame with the corresponding label names


def label_int2str(row):
    return emotions["train"].features["label"].int2str(row)


df["label_name"] = df["label"].apply(label_int2str)

df["label_name"].value_counts(ascending=True).plot.barh()
plt.title("Frequency of Classes")
plt.show()
'''
There are several ways to deal with imbalanced data, including:
* Randomly oversample the minority class.
* Randomly undersample the majority class.
* Gather more labeled data from the underrepresented classes.
To keep things simple in this chapter, we’ll work with the raw, unbalanced class frequencies. If you want to learn more about these sampling techniques, we recommend checking out the Imbalanced-learn library. Just make sure that you don’t apply sampling methods before creating your train/test splits, or you’ll get plenty of leakage between them!
'''
