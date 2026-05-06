import pandas as pd


# check split of samples lean vs myogym, ignore gym exercises

PATH = "./golden_dataset_v9.csv"

df = pd.read_csv(PATH)

print("CHECK")
seq_labels = df.groupby("ID")[["parent", "child"]].first()

print("Sequence counts:")
print(seq_labels.value_counts())

print("\nSequence proportions:")
print(seq_labels.value_counts(normalize=True))