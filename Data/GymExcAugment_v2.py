import pandas as pd
from scipy import signal


columns_to_keep = ['wristMotion_accelerationX','wristMotion_accelerationY','wristMotion_accelerationZ','wristMotion_rotationRateX','wristMotion_rotationRateY','wristMotion_rotationRateZ']

files = [
    "C:\\Users\\scruf\\Desktop\\WHT\\011224_PREC_W7_5_S1_R10-2024-12-01_08-18-54.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\011224_PREC_W7_5_S2_R10-2024-12-01_08-23-27.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\021224_30DBP_W10_S1_R6-2024-12-02_14-42-26.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\021224_30DBP_W12_5_S2_R5-2024-12-02_14-44-20.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\021224_30DBP_W17_5_S3_R18-2024-12-02_14-47-29.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\021224_30DBP_W17_5_S4_R14-2024-12-02_14-53-23.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\040125_30DBP_W7_5_S1_R12-2025-01-04_14-51-59.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\040125_30DBP_W12_5_S2_R8-2025-01-04_14-53-47.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\040125_30DBP_W17_5_S3_R12-2025-01-04_14-57-26.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\040125_30DBP_W17_5_S4_R10-2025-01-04_15-02-06.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\120125_30BP_W30_S1_R12-2025-01-12_07-40-23.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\120125_30BP_W30_S2_R8-2025-01-12_07-46-50.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\130125_PREC_W7_5_S1_R10-2025-01-13_16-56-48.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\231124_30DBP_W17_5_S1_R16-2024-11-23_12-21-42.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\231124_30DBP_W17_5_S2_R16-2024-11-23_12-27-44.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\231124_45DBP_W12_5_S1_R16-2024-11-23_12-34-11.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\231124_45DBP_W12_5_S2_R16-2024-11-23_12-39-26.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\241124_PREC_W7_5_S1_R14-2024-11-24_08-16-33.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\241124_PREC_W7_5_S2_R14-2024-11-24_08-22-36.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\261124_30DBP_W10_S1_R8-2024-11-26_13-18-24.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\261124_30DBP_W17_5_S2_17-2024-11-26_13-22-15.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\261124_30DBP_W17_5_S2_R12-2024-11-26_13-27-55.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\261124_45DBP_W12_5_S1_R12-2024-11-26_13-39-13.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\261224_AIDBC_W7_5_S1_R12-2024-12-26_16-19-51.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\271124_PREC_W7_5_S1_R9-2024-11-27_15-30-47.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\271124_PREC_W7_5_S2_R11-2024-11-27_15-36-09.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\291124_30DBP_W7_5_S1_R8-2024-11-29_14-34-27.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\291124_30DBP_W7_5_S2_R8-2024-11-29_14-49-58.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\291124_30DBP_W17_5_S1_R16-2024-11-29_14-41-39.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\291124_45DBP_W12_5_S1_R12-2024-11-29_14-55-21.csv"
]

excercise_participant = [
    ('1b','8'),
    ('1b','8'),
    ('2e','8'),
    ('2e','8'),
    ('2e','8'),
    ('2e','8'),
    ('2e','8'),
    ('2e','8'),
    ('2e','8'),
    ('2e','8'),
    ('2b','8'),
    ('2b','8'),
    ('1b','8'),
    ('2e','8'),
    ('2e','8'),
    ('2e','9'),
    ('2e','9'),
    ('1b','8'),
    ('1b','8'),
    ('2e','8'),
    ('2e','8'),
    ('2e','8'),
    ('2e','9'),
    ('1a','8'),
    ('1b','8'),
    ('1b','8'),
    ('2e','8'),
    ('2e','8'),
    ('2e','8'),
    ('2e','9'),
]

all = []
for i,f in enumerate(files):
    df = pd.read_csv(f)
    df = df[columns_to_keep].iloc[150:-150]
    df_downsampled = pd.DataFrame(
        {col: signal.decimate(df[col].values.astype(float), 5) for col in df.columns}
    )

    excercise = []
    participant = []
    for j in range(len(df_downsampled)):
        excercise.append(excercise_participant[i][0])
        participant.append(excercise_participant[i][1])
    df_downsampled['excercise'] = excercise
    df_downsampled['participant'] = participant

    all.append(df_downsampled)

full = pd.concat(all, ignore_index=True)


full.columns = ['x_accel','y_accel','z_accel','x_gyro','y_gyro','z_gyro','excercise','participant']


full.to_csv('GymExc_v2.csv', index=False)
