import pandas as pd


full = pd.read_csv('261224_AIDBC_W7_5_S1_R12-2024-12-26_16-19-51.csv')
df = full

df = df.iloc[150: len(full) - 150]
downsampled_rows = []
downsampled_rows.append(df.iloc[::5])
df = pd.concat(downsampled_rows, ignore_index=True)

columns_to_keep = ['wristMotion_accelerationX','wristMotion_accelerationY','wristMotion_accelerationZ','wristMotion_rotationRateX','wristMotion_rotationRateY','wristMotion_rotationRateZ']
df = df[columns_to_keep]

df.columns = ['x_accel','y_accel','z_accel','x_gyro','y_gyro','z_gyro']

excercise = []
for i in range(len(df)):
    excercise.append('arm_curl')

df['excercise'] = excercise


df.to_csv('GymExc.csv', index=False)
