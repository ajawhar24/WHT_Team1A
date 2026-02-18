import pandas as pd
import re

CSV_PATH = "WHT_Team1A/Data/RecGym.csv"

# load in csv and clean any data that needs
df = pd.read_csv(CSV_PATH)

df = df.drop(columns=["Subject", "Position", "Session", "C_1"])
df.columns = [
    "x_accel",
    "y_accel",
    "z_accel",
    "x_gyro",
    "y_gyro",
    "z_gyro",
    "excercise"
]

# print(df.dtypes)

df_armcurl = df[df["excercise"] == "ArmCurl"]

df_benchpress = df[df["excercise"] == "BenchPress"]

df_armcurl["excercise"] = df_armcurl["excercise"].str.replace(r"([a-z])([A-Z])", r"\1_\2", regex=True).str.lower()
df_benchpress["excercise"] = df_benchpress["excercise"].str.replace(r"([a-z])([A-Z])", r"\1_\2", regex=True).str.lower()

combined_df = pd.concat([df_armcurl, df_benchpress], ignore_index=True)

# print(df_armcurl.sample(5))
# print(df_benchpress.sample(5))
# print(combined_df.sample(5))

combined_df.to_csv("RecGym_augmented.csv", index=False)

