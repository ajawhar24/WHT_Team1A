import pandas as pd
import numpy as np

INPUT_PATH = "./golden_dataset_random_v10_10Hz.csv"
OUTPUT_PATH = "./golden_dataset_v10_10Hz.csv"

df = pd.read_csv(INPUT_PATH)

# 1. get unique block IDs
unique_ids = df["ID"].unique()

# 2. shuffle them
np.random.seed(42)  # optional (reproducibility)
shuffled_ids = np.random.permutation(unique_ids)

# 3. rebuild dataframe in new order
shuffled_df = pd.concat(
    [df[df["ID"] == i] for i in shuffled_ids],
    ignore_index=True
)

# save
shuffled_df.to_csv(OUTPUT_PATH, index=False)

print("Saved:", OUTPUT_PATH)
print("Original rows:", len(df))
print("Shuffled rows:", len(shuffled_df))