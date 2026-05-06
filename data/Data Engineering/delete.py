import pandas as pd

# INPUT_PATH = "./golden_dataset_random_v4.csv"
# OUTPUT_PATH = "./golden_dataset_random_v5.csv"

INPUT_PATH = "./MyData/DiffFreq/OurData_v5_10Hz.csv"
OUTPUT_PATH = "./MyData/DiffFreq/OurData_v5_10Hz_children.csv"

# INPUT_PATH = "MyData/Lean_v3.csv"
# OUTPUT_PATH = "MyData/MyoGym_v5.csv"

# INPUT_PATH = "MyData/MyOurData_v5_unshuffled.csv.csv"
# OUTPUT_PATH = "MyData/MyOurData_v7.csv"

df = pd.read_csv(INPUT_PATH)

# remove null and unwanted rows
df = df[
    ((df["parent"] == 1) & (df["child"] == 3)) |
    ((df["parent"] == 5) & (df["child"] == 5))
].copy()

# # build mapping
# mapping = {
#     old: new
#     for new, old in enumerate(df["ID"].unique(), start=1)
# }

# # apply mapping
# df["new_ID"] = df["ID"].map(mapping)

# # preview mapping
# print(pd.DataFrame(list(mapping.items()), columns=["old_ID", "new_ID"]).head(20))

# # 🔥 replace old IDs
# df["ID"] = df["new_ID"]
# df = df.drop(columns=["new_ID"])

print(df[["parent", "child"]].value_counts())

df.to_csv(OUTPUT_PATH, index=False)

# print("Saved:", OUTPUT_PATH)
print("Shape:", df.shape)
# print("Remaining label pairs:")
# print(df[["parent", "child"]].drop_duplicates())
# print("Remaining IDs:", df["ID"].nunique())