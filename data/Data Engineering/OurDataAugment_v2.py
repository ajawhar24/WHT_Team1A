import pandas as pd
from scipy import signal
import numpy as np

idOffset = 20
participantOffset = 14

columns_to_keep = ['accelerationX','accelerationY','accelerationZ','rotationRateX','rotationRateY','rotationRateZ']

files = [
    "C:\\Users\\scruf\\Desktop\\WHT\\Collected_Data_Abdullah\\A_BB_AC.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\Collected_Data_Abdullah\\A_BB_BP.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\Collected_Data_Abdullah\\A_BB_DBP.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\Collected_Data_Abdullah\\A_BB_IBP.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\Collected_Data_Abdullah\\A_BB_PC.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\Collected_Data_Abdullah\\A_BB_RAC.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\Collected_Data_Abdullah\\A_DB_AC.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\Collected_Data_Abdullah\\A_DB_BP.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\Collected_Data_Abdullah\\A_DB_DBP.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\Collected_Data_Abdullah\\A_DB_IBP.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\Collected_Data_Constance\\C_BB_AC.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\Collected_Data_Constance\\C_BB_BP.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\Collected_Data_Constance\\C_BB_DBP.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\Collected_Data_Constance\\C_BB_IBP.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\Collected_Data_Constance\\C_BB_PC.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\Collected_Data_Constance\\C_BB_RAC.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\Collected_Data_Constance\\C_DB_AC.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\Collected_Data_Constance\\C_DB_BP.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\Collected_Data_Constance\\C_DB_DBP.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\Collected_Data_Constance\\C_DB_IBP.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\Collected_Data_Ishita\\I_BB_AC.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\Collected_Data_Ishita\\I_BB_BP.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\Collected_Data_Ishita\\I_BB_DBP.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\Collected_Data_Ishita\\I_BB_IBP.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\Collected_Data_Ishita\\I_BB_PC.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\Collected_Data_Ishita\\I_BB_RAC.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\Collected_Data_Ishita\\I_DB_BP.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\Collected_Data_Ishita\\I_DB_DBP.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\Collected_Data_Ishita\\I_DB_IBP.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\Collected_Data_Julian\\J_BB_AC.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\Collected_Data_Julian\\J_BB_BP.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\Collected_Data_Julian\\J_BB_DBP.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\Collected_Data_Julian\\J_BB_IBP.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\Collected_Data_Julian\\J_BB_PC.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\Collected_Data_Julian\\J_BB_RAC.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\Collected_Data_Julian\\J_DB_AC.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\Collected_Data_Julian\\J_DB_BP.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\Collected_Data_Julian\\J_DB_DBP.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\Collected_Data_Julian\\J_DB_IBP.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\Collected_Data_Stephen\\S_BB_AC.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\Collected_Data_Stephen\\S_BB_BP.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\Collected_Data_Stephen\\S_BB_DBP.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\Collected_Data_Stephen\\S_BB_IBP.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\Collected_Data_Stephen\\S_BB_PC.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\Collected_Data_Stephen\\S_BB_RAC.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\Collected_Data_Stephen\\S_DB_AC.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\Collected_Data_Stephen\\S_DB_BP.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\Collected_Data_Stephen\\S_DB_DBP.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\Collected_Data_Stephen\\S_DB_IBP.csv"
]

# parent, child, participant, ID, start1, end1, start2, end2
tailData = [
    (1,3,1,1,550,3838,5359,8528),
    (5,5,1,3,612,2794,4704,6966),
    (5,7,1,5,711,2557,4382,6245),
    (5,6,1,7,786,4184,6190,9264),
    (1,2,1,9,633,3456,5066,8051),
    (1,4,1,11,688,3690,5305,8249),
    (1,1,1,13,1329,4307,5623,9069),
    (5,8,1,15,795,6092,7270,12256),
    (5,9,1,17,571,4297,6015,9538),
    (5,10,1,19,880,5641,6869,11322),
    (1,3,2,21,373,2659,4469,6838),
    (5,5,2,23,340,2239,3981,5892),
    (5,7,2,25,1156,2607,4180,6165),
    (5,6,2,27,530,2484,4523,6528),
    (1,2,2,29,212,2205,3792,5589),
    (1,4,2,31,350,3038,4182,6928),
    (1,1,2,33,231,3397,4832,9080),
    (5,8,2,35,286,2221,3690,5654),
    (5,9,2,37,599,3131,4910,7140),
    (5,10,2,39,359,2591,4077,6071),
    (1,3,3,41,505,3349,5339,8368),
    (5,5,3,43,345,2399,4235,6638),
    (5,7,3,45,337,2985,6875,10632),
    (5,6,3,47,495,3202,5371,9062),
    (1,2,3,49,316,2874,4517,7205),
    (1,4,3,51,355,2881,4497,7785),
    (5,8,3,53,463,2619,4433,6676),
    (5,9,3,55,443,2620,3983,6496),
    (5,10,3,57,307,2423,3873,6330),
    (1,3,4,59,388,2829,4480,6686),
    (5,5,4,61,1017,3410,5262,7193),
    (5,7,4,63,680,2915,4973,7023),
    (5,6,4,65,536,2712,4472,6602),
    (1,2,4,67,394,2780,4151,6628),
    (1,4,4,69,306,2436,4169,6563),
    (1,1,4,71,373,3101,4464,7043),
    (5,8,4,73,1043,3179,4711,6677),
    (5,9,4,75,753,3097,4843,7067),
    (5,10,4,77,371,2502,4301,6315),
    (1,3,5,79,498,3129,4511,7883),
    (5,5,5,81,948,3310,4899,7237),
    (5,7,5,83,648,3022,4799,7181),
    (5,6,5,85,684,3257,5014,7501),
    (1,2,5,87,358,4534,5846,9889),
    (1,4,5,89,484,4018,5408,8592),
    (1,1,5,91,355,4074,5294,8753),
    (5,8,5,93,456,3411,4836,7301),
    (5,9,5,95,436,3426,5096,7905),
    (5,10,5,97,300,3340,4578,7034),
]

all = []
for i,f in enumerate(files):
    df = pd.read_csv(f)
    df = df[columns_to_keep]

    decimated = {}
    for col in df.columns[:6]:
        vals = df[col].values.astype(np.float64)
        mean = np.mean(vals)
        vals_centered = vals - mean                        
        resampled = signal.resample_poly(vals_centered, 1, 5, padtype='line')
        decimated[col] = resampled + mean

    df_downsampled = pd.DataFrame(decimated)

    parent = []
    child = []
    participant = []
    setNum = []
    id = []
    ax, ay, az, gx, gy, gz = [], [], [], [], [], []
    df_downsampled_noNull = pd.DataFrame()
    for j in range(len(df_downsampled)):
        # df_downsampled.iloc[j, 0] = df_downsampled.iloc[j, 0]*9.81
        # df_downsampled.iloc[j, 1] = df_downsampled.iloc[j, 1]*9.81
        # df_downsampled.iloc[j, 2] = df_downsampled.iloc[j, 2]*9.81
        df_downsampled.iloc[j, 3] = df_downsampled.iloc[j, 3]*180/np.pi
        df_downsampled.iloc[j, 4] = df_downsampled.iloc[j, 4]*180/np.pi
        df_downsampled.iloc[j, 5] = df_downsampled.iloc[j, 5]*180/np.pi
        if i in (10,11,12,13,14,15,16,17,18,19): # Constance is a lefty
            df_downsampled.iloc[j, 0] = -df_downsampled.iloc[j, 0]
            df_downsampled.iloc[j, 4] = -df_downsampled.iloc[j, 4]
            df_downsampled.iloc[j, 5] = -df_downsampled.iloc[j, 5]
        if i in (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,39,40,41,42,43,44,45,46,47,48): # Abdullah's watch is rotated 180 degrees
            df_downsampled.iloc[j, 0] = -df_downsampled.iloc[j, 0]
            df_downsampled.iloc[j, 1] = -df_downsampled.iloc[j, 1]
            df_downsampled.iloc[j, 3] = -df_downsampled.iloc[j, 3]
            df_downsampled.iloc[j, 4] = -df_downsampled.iloc[j, 4]
        if not(j < tailData[i][4]//5 or tailData[i][5]//5 < j < tailData[i][6]//5 or j > tailData[i][7]//5):
            ax.append(df_downsampled.iloc[j, 0])
            ay.append(df_downsampled.iloc[j, 1])
            az.append(df_downsampled.iloc[j, 2])
            gx.append(df_downsampled.iloc[j, 3])
            gy.append(df_downsampled.iloc[j, 4])
            gz.append(df_downsampled.iloc[j, 5])

            parent.append(tailData[i][0])
            child.append(tailData[i][1])
            if j < (tailData[i][5]//5 + tailData[i][6]//5) // 2:
                setNum.append(1)
                id.append(tailData[i][3]+idOffset)
            else:
                setNum.append(2)
                id.append(tailData[i][3]+1+idOffset)
            participant.append(tailData[i][2]+participantOffset)
    df_downsampled_noNull['x_accel'] = ax
    df_downsampled_noNull['y_accel'] = ay
    df_downsampled_noNull['z_accel'] = az
    df_downsampled_noNull['x_gyro'] = gx
    df_downsampled_noNull['y_gyro'] = gy
    df_downsampled_noNull['z_gyro'] = gz

    df_downsampled_noNull['parent'] = parent
    df_downsampled_noNull['child'] = child
    df_downsampled_noNull['participant'] = participant
    df_downsampled_noNull['set'] = setNum
    df_downsampled_noNull['ID'] = id

    all.append(df_downsampled_noNull)

full = pd.concat(all, ignore_index=True)



# id_col = full.columns[-1]
# unique_ids = full[id_col].unique()
# np.random.shuffle(unique_ids)
# final = pd.concat([full[full[id_col] == id] for id in unique_ids]).reset_index(drop=True)
# final.columns = ['x_accel','y_accel','z_accel','x_gyro','y_gyro','z_gyro','parent', 'child','participant', 'set', 'ID']

# final.to_csv('OurData_v5.csv', index=False)
full.to_csv('OurData_v5_unshuffled.csv', index=False)
