import pandas as pd

CSV_PATH = "./MyData/DiffFreq/MyoGym_40Hz_v1.csv"
OUTPUT_PATH = "./MyData/Final/MyoGym_40Hz_final.csv"

df = pd.read_csv(CSV_PATH)

# flip
# y_accel
# x_gyro
# z_gyro

df["y_accel"] = df["y_accel"] * -1
df["x_gyro"] = df["x_gyro"] * -1
df["z_gyro"] = df["z_gyro"] * -1


df.to_csv(OUTPUT_PATH, index=False)

