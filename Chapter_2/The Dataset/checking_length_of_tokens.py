
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
# checking length of the tweets in the dataset

df["Words Per Tweet"] = df["text"].str.split().apply(len)
df.boxplot("Words Per Tweet", by="label_name",
           grid=False, showfliers=False, color="black")
plt.suptitle("")
plt.xlabel("")
plt.show()

'''
For applications using DistilBERT, the maximum context size is 512 tokens, which amounts to a few paragraphs of text.

From the plot we see that for each emotion, most tweets are around 15 words long and the longest tweets are well below DistilBERT’s maximum context size. Texts that are longer than a model’s context size need to be truncated, which can lead to a loss in performance if the truncated text contains crucial information; in this case, it looks like that won’t be an issue.
'''
