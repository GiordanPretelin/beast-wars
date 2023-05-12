import pandas as pd

# the simplest tokenization scheme is per character
text = "Tokenizing text is a core task of NLP."
tokenized_text = list(text)
print(tokenized_text)

"""This model expects each character to be converted to an integer, a process
sometimes called numericalization. One simple way to do this is by encoding
each unique token (which are characters in this case) with a unique integer:"""
token2idx = {ch: idx for idx, ch in enumerate(sorted(set(tokenized_text)))}
print(token2idx)

input_ids = [token2idx[token] for token in tokenized_text]
print(input_ids)

# converting input_ids to a 2d tensor of one-hot vectors
# Example first
categorical_df = pd.DataFrame(
    {"Name": ["Bumblebee", "Optimus Prime", "Megatron"], "Label ID": [0, 1, 2]}
)
# regular dataframe
print(categorical_df)

print(pd.get_dummies(categorical_df["Name"]))

# Applying PyTorch one-hot encodings
import torch
import torch.nn.functional as F

input_ids = torch.tensor(input_ids)
one_hot_encodings = F.one_hot(input_ids, num_classes=len(token2idx))
print(one_hot_encodings.shape)
print(f"Token: {tokenized_text[0]}")
print(f"Tensor index: {input_ids[0]}")
print(f"One-hot: {one_hot_encodings[0]}")
