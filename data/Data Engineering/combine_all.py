import pandas as pd

MYOGYM_PATH = "./MyData/Final/MyoGym_40Hz_final.csv"
# GYMEX_PATH = "./MyData/GymExercises_v4.csv"
LEAN_PATH = "./MyData/Final/Lean_40Hz_final.csv"
OURDATA_PATH = "./MyData/Final/OurData_40Hz_final.csv"
OUTPUT_PATH = "./golden_dataset_random_v10_40Hz.csv"

myogym = pd.read_csv(MYOGYM_PATH)
# gymex = pd.read_csv(GYMEX_PATH)
lean = pd.read_csv(LEAN_PATH)
ours = pd.read_csv(OURDATA_PATH)

required_columns = [
    "x_accel",
    "y_accel",
    "z_accel",
    "x_gyro",
    "y_gyro",
    "z_gyro",
    "parent",
    "child",
    "participant",
    "set",
    "ID",
    "dataset",
]

def remap_ids_sequentially(df, start_id):
    df = df.copy()

    # get unique sequence IDs
    old_ids = sorted(df["ID"].unique())

    # map old -> new sequential IDs
    id_map = {old_id: new_id for new_id, old_id in enumerate(old_ids, start=start_id)}

    df["ID"] = df["ID"].map(id_map)

    next_start_id = start_id + len(old_ids)
    return df, next_start_id

myogym = myogym[required_columns].copy()
# gymex = gymex[required_columns].copy()
lean = lean[required_columns].copy()
ours = ours[required_columns].copy()

print("MyoGym unique IDs:", myogym["ID"].nunique())
print("Lean unique IDs:", lean["ID"].nunique())
print("Ours unique IDs:", ours["ID"].nunique())

# # offset GymEx IDs so they stay unique after merge
# id_offset = myogym["ID"].max()
# lean["ID"] = lean["ID"] + id_offset

# # offset Ours IDs so they stay unique after merge
# id_offset = lean["ID"].max()
# ours["ID"] = ours["ID"] + id_offset

# # optional: also offset participant IDs if you want both datasets to never overlap
# # comment this out if you WANT shared participant ranges
# participant_offset = recgym["participant"].max()
# myogym["participant"] = myogym["participant"] + participant_offset

# --- REMAP IDS ---
next_id = 1

myogym, next_id = remap_ids_sequentially(myogym, next_id)
lean, next_id = remap_ids_sequentially(lean, next_id)
ours, next_id = remap_ids_sequentially(ours, next_id)

# --- MERGE ---
# merged_df = pd.concat([myogym, lean, ours], ignore_index=True)
# merged_df = merged_df[required_columns]

merged_df = pd.concat([myogym, lean, ours], ignore_index=True)

merged_df = merged_df[
    [
        "x_accel",
        "y_accel",
        "z_accel",
        "x_gyro",
        "y_gyro",
        "z_gyro",
        "parent",
        "child",
        "participant",
        "set",
        "ID",
        "dataset",
    ]
]

# print(merged_df.duplicated(subset=["ID"]).sum())
# print(merged_df.duplicated(subset=["dataset", "participant", "set"]).sum())

# print("Merged unique IDs:", merged_df["ID"].nunique())

# --- AFTER CHECK ---
print("\nAfter merge:")
print("Shape:", merged_df.shape)
print("Datasets:", merged_df["dataset"].value_counts().to_dict())

expected = (
    myogym["ID"].nunique()
    + lean["ID"].nunique()
    + ours["ID"].nunique()
)

actual = merged_df["ID"].nunique()

print("Expected unique IDs:", expected)
print("Actual unique IDs:", actual)

# optional: inspect distribution
print("\nRows per ID (sample):")
print(merged_df.groupby("ID").size().head())

print("\nLabel split:")
print(merged_df[["parent", "child"]].value_counts().sort_index())

print(
    merged_df.groupby("ID")[["dataset"]].nunique().value_counts()
)

print("IDs spanning multiple datasets:",
      (merged_df.groupby("ID")["dataset"].nunique() > 1).sum())
merged_df.to_csv(OUTPUT_PATH, index=False)

# print("Saved:", OUTPUT_PATH)
# print("Shape:", merged_df.shape)
# print("Datasets:", merged_df["dataset"].value_counts().to_dict())
# print("Unique IDs:", merged_df["ID"].nunique())
# print("Unique participants:", sorted(merged_df["participant"].unique()))
# print("Label pairs:")
# print(merged_df[["parent", "child"]].drop_duplicates().sort_values(["parent", "child"]))

# golden = pd.read_csv("golden_dataset_v4.csv")
# id_offset = golden["ID"].max()

# print(id_offset)
# --- 8. Check balance ---
print("CHECK")
seq_labels = merged_df.groupby("ID")[["parent", "child"]].first()

print("Sequence counts:")
print(seq_labels.value_counts())

print("\nSequence proportions:")
print(seq_labels.value_counts(normalize=True))

# # downsampling lean

# import numpy as np

# # --- 1. Get one row per sequence (ID) ---
# seq_labels = (
#     merged_df.groupby("ID")[["parent", "child", "dataset"]]
#     .first()
#     .reset_index()
# )

# # --- 2. Split groups ---
# group_13 = seq_labels[(seq_labels["parent"] == 1) & (seq_labels["child"] == 3)]
# group_55 = seq_labels[(seq_labels["parent"] == 5) & (seq_labels["child"] == 5)]

# # --- 3. Separate Lean vs others ---
# lean_13 = group_13[group_13["dataset"] == "lean"]
# other_13 = group_13[group_13["dataset"] != "lean"]

# # --- 4. Compute target ---
# target = len(group_55)

# # we MUST keep all (5,5) and all non-Lean (1,3)
# remaining_needed = target - len(other_13)

# print("Total (5,5) sequences:", target)
# print("Non-Lean (1,3) sequences kept:", len(other_13))
# print("Lean (1,3) sequences to keep:", remaining_needed)

# # --- 5. Sample Lean sequences ---
# lean_ids = lean_13["ID"].values
# np.random.shuffle(lean_ids)

# selected_lean_ids = lean_ids[:remaining_needed]

# # --- 6. Build final ID set ---
# keep_ids = set(group_55["ID"]) \
#          | set(other_13["ID"]) \
#          | set(selected_lean_ids)

# # --- 7. Filter dataset ---
# balanced_df = merged_df[merged_df["ID"].isin(keep_ids)]

# # --- 8. Check balance ---
# print("\nSequence-level balance:")
# print(
#     balanced_df.groupby("ID")[["parent", "child"]]
#     .first()
#     .value_counts(normalize=True)
# )

# print("\nRow-level balance:")
# print(
#     balanced_df[["parent", "child"]]
#     .value_counts(normalize=True)
# )

# print("\nLabel split:")
# print(balanced_df[["parent", "child"]].value_counts().sort_index())

# # row-level
# print(balanced_df[["parent", "child"]].value_counts(normalize=True))

# # sequence-level (important)
# print(
#     balanced_df.groupby("ID")[["parent", "child"]]
#     .first()
#     .value_counts(normalize=True)
# )