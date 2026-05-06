import pandas as pd
from scipy import signal
import numpy as np

idOffset = 0
participantOffset = 0

columns_to_keep = ['DMUAccelX','DMUAccelY','DMUAccelZ','DMRotX','DMRotY','DMRotZ']

files = [
    "C:\\Users\\scruf\\Desktop\\WHT\\BicepCurl-Good-0.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\BicepCurl-Good-1.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\BicepCurl-Good-2.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\BicepCurl-Good-3.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\BicepCurl-Good-4.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\BicepCurl-Good-5.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\BicepCurl-Good-6.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\BicepCurl-Good-7.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\BicepCurl-Good-8.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\BicepCurl-Good-9.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\BicepCurl-Test-Good-0.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\BicepCurl-Good-0.csv",
    "C:\\Users\\scruf\\Desktop\\WHT\\BicepCurl-Test-RealWorld-0.csv"
]

# parent, child, participant, ID, start1, end1, start2, end2
tailData = [
    (1,3,1,1,1,204,1572),
    (1,3,1,2,2,1405,2999),
    (1,3,1,3,3,1610,3091),
    (1,3,1,4,4,1393,2891),
    (1,3,1,5,5,1580,3037),
    (1,3,1,6,6,1340,2815),
    (1,3,1,7,7,1490,3066),
    (1,3,1,8,8,1670,3307),
    (1,3,1,9,9,1281,2893),
    (1,3,1,10,10,1426,3109),
    (1,3,2,1,11,286,1664),
    (1,3,3,1,12,2010,3484),
    (1,3,4,1,13,437,1519)
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
        resampled = signal.resample_poly(vals_centered, 1, 2, padtype='line')
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
        if i not in (10,11,12): # Right Hand
            df_downsampled.iloc[j, 0] = -df_downsampled.iloc[j, 0]
            df_downsampled.iloc[j, 4] = -df_downsampled.iloc[j, 4]
            df_downsampled.iloc[j, 5] = -df_downsampled.iloc[j, 5]
        if not(j < tailData[i][5]//2 or tailData[i][6]//2 < j):
            ax.append(df_downsampled.iloc[j, 0])
            ay.append(df_downsampled.iloc[j, 1])
            az.append(df_downsampled.iloc[j, 2])
            gx.append(df_downsampled.iloc[j, 3])
            gy.append(df_downsampled.iloc[j, 4])
            gz.append(df_downsampled.iloc[j, 5])

            parent.append(tailData[i][0])
            child.append(tailData[i][1])
            setNum.append(tailData[i][3])
            participant.append(tailData[i][2]+participantOffset)
            id.append(tailData[i][4]+idOffset)

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



full.to_csv('Lean_trimmed.csv', index=False)
