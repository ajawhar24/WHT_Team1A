import pandas as pd

# CSV_PATH = "./MyData/GymExc_v4.csv"
# OUTPUT_PATH = "./MyData/GymExercises_v4.csv"

# CSV_PATH = "./MyData/RecGym_v3.csv"
# OUTPUT_PATH = "./MyData/RecGym_v4.csv"

# CSV_PATH = "./MyData/DiffFreq/OurData_v5_10Hz_children.csv"
# OUTPUT_PATH = "./MyData/Final/OurData_10Hz_final.csv"


# CSV_PATH = "./MyData/Lean_trimmed (1).csv"
# OUTPUT_PATH = "./MyData/Lean_trimmed_v3.csv"


# CSV_PATH = "./MyData/OurData_v5_unshuffled.csv"
# OUTPUT_PATH = "./MyData/MyOurData_v5_unshuffled.csv"

# CSV_PATH = "./golden_dataset_v10.csv"
# CSV_PATH = "./MyData/RecGym_v4.csv"

CSV_PATH = "./golden_dataset_v10_10Hz.csv"

df = pd.read_csv(CSV_PATH)

# df["dataset"] = "ourdata"

# df["participant"] = df["participant"] + 10

# df["parent"] = df["parent"].replace({2: 5})
print("Unique participants:", sorted(df["participant"].unique()))

print(df[["parent", "child"]].drop_duplicates().sort_values(["parent", "child"]))

print(df[["participant", "ID"]].drop_duplicates())

print(
    df[["participant", "ID"]]
    .drop_duplicates()
    .sort_values(["participant", "ID"])
)
# df.to_csv(OUTPUT_PATH, index=False)

# flip
# y_accel
# x_gyro
# z_gyro