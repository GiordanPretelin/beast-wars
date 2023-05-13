from datasets import load_dataset

emotions = load_dataset("emotion")

print(emotions)

train_ds = emotions["train"]
train_ds

print(train_ds)
# showing it proves like an array or list
print(len(train_ds))
# accesing an example by its index, each row is a dictionary where keys are columns
print(train_ds[0])
# showing column names
print(train_ds.column_names)
# showing data types
print(train_ds.features)
# accesing rows with a slice
print(train_ds[:5])
# the dict values are lists and columns can be filtered by name
print(train_ds["text"][:5])
