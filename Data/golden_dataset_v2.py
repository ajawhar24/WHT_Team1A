import pandas as pd
import glob

input_csv = "data/*v2*.csv"
output_file = "golden_dataset_v2.csv"

cols = [
    "x_accel",
    "y_accel",
    "z_accel",
    "x_gyro",
    "y_gyro",
    "z_gyro",
    "excercise",
    "participant"
]
all_files = glob.glob(input_csv)

if not all_files:
    raise FileNotFoundError(f"No files matched {input_csv}!")

dfs = []

for f in all_files:
    if f.endswith("MyoGym_v2_2.csv"):
        df = pd.read_csv(f, header=None, names=cols)
    else:
        df = pd.read_csv(f, low_memory=False)

    df = df[cols]
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)

# print(combined_df.columns)
# print(combined_df.shape)
# print(combined_df.head())
# print(combined_df.sample(5))

combined_df.to_csv(output_file, index=False)

print(f"Successfully wrote {output_file} from {len(all_files)} files!")