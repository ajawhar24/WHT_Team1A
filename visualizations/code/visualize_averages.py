import pandas as pd
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt

def showPeaks(df, col):
    peaks, props = find_peaks(df[col], prominence=0)
    top10 = np.argsort(props["prominences"])[-10:]
    peaks = np.sort(peaks[top10])

    plt.plot(df[col])
    plt.plot(peaks, df[col].iloc[peaks], 'x')
    #plt.title(f'{int(df["parent"].iloc[0])} - {int(df["child"].iloc[0])} - {int(df["participant"].iloc[0])} - {int(df["ID"].iloc[0])} - {col}')
    plt.title(f'{int(df["parent"].iloc[0])} - {int(df["child"].iloc[0])} - {int(df["participant"].iloc[0])} - {int(df["ID"].iloc[0])} - {col} - {df["dataset"].iloc[0]}')
    plt.show()

def getSets(arr):
    sets = []
    for i in range(1, len(arr)):
        if arr[i] != arr[i-1]:
            sets.append(i)
    return [0] + sets + [len(arr)+1]

def plot_average_reps(all_reps, labels=None, title=None, common_length=100):
    colors = ['blue', 'orange', 'green']
    fig, axes = plt.subplots(1, len(all_reps), figsize=(5 * len(all_reps), 4), sharey=True)

    if labels is None:
        labels = [f"Dataset {i+1}" for i in range(len(all_reps))]

    for ax, reps, label, color in zip(axes, all_reps, labels, colors):
        # print(f"{label}: {len(reps)} reps found")  # debug
        if len(reps) == 0:
            ax.set_title(f"{label} (no data)")
            continue

        resampled = []
        for rep in reps:
            if len(rep) < 2:  # skip reps too short to interpolate
                continue
            x_old = np.linspace(0, 1, len(rep))
            x_new = np.linspace(0, 1, common_length)
            f = interp1d(x_old, rep, kind='linear')
            resampled.append(f(x_new))

        if len(resampled) == 0:
            ax.set_title(f"{label} (no valid reps)")
            continue

        resampled = np.array(resampled)
        mean = np.mean(resampled, axis=0)
        std = np.std(resampled, axis=0)

        x = np.linspace(0, 100, common_length)
        ax.plot(x, mean, color=color, label=label)
        ax.fill_between(x, mean - std, mean + std, alpha=0.3, color=color)
        ax.set_title(label)
        ax.set_xlabel('% of rep')

    if 'accel' in title:
        axes[0].set_ylabel('acceleration [m/s^2]')
    else:
        axes[0].set_ylabel('angular velocity [deg/s]')

    if title:
        fig.suptitle(title, fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        plt.tight_layout()

    plt.savefig(f'{title}.png', dpi=300, bbox_inches='tight')  
    plt.show()

def getReps1(dfs):
    listListArray = []
    for df in dfs:
        listArray = []
        sets = getSets(df)
        for i in range(len(sets)-1):
            set_df = df[sets[i]:sets[i+1]]
            peaks, props = find_peaks(set_df, prominence=0)
            top10 = np.argsort(props["prominences"])[-10:]
            peaks = np.sort(peaks[top10])
            total = 0
            for j in range(len(peaks)-1):
                total += peaks[j+1] - peaks[j]
            average = total / (len(peaks)-1)
            slices = []
            for j in range(len(peaks)-1):
                if average * 0.75 < peaks[j+1] - peaks[j] < average * 1.25:
                    slices.append(j)

            for j in slices:
                listArray.append(set_df[peaks[j]:peaks[j+1]])
        listListArray.append(listArray)
    return listListArray
                
def getReps(dfs, col):
    listListArray = []
    for df in dfs:
        listArray = []
        sets = getSets(df["ID"].values)
        # print(f"sets: {sets}")
        for i in range(len(sets)-1):
            set_arr = df[col].values[sets[i]:sets[i+1]]
            # print(f"set {i} length: {len(set_arr)}")
            peaks, props = find_peaks(set_arr, prominence=0)
            # print(f"peaks found: {len(peaks)}")
            if len(peaks) < 2:
                # print("not enough peaks, skipping")
                continue
            top10 = np.argsort(props["prominences"])[-10:]
            peaks = np.sort(peaks[top10])
            total = 0
            for j in range(len(peaks)-1):
                total += peaks[j+1] - peaks[j]
            average = total / (len(peaks)-1)
            slices = []
            for j in range(len(peaks)-1):
                if average * 0.6 < peaks[j+1] - peaks[j] < average * 1.4:
                    slices.append(j)
            # print(f"valid slices: {slices}")
            for j in slices:
                rep = set_arr[peaks[j]:peaks[j+1]]
                # print(f"rep length: {len(rep)}")
                listArray.append(rep)
        # print(f"total reps found: {len(listArray)}")
        listListArray.append(listArray)
    return listListArray


def main():
    df1 = pd.read_csv('golden_dataset_v10.csv')
    ours = pd.read_csv('OurData_v5.csv')
    df1['x_accel'] = 9.81 * df1['x_accel']
    df1['y_accel'] = 9.81 * df1['y_accel']
    df1['z_accel'] = 9.81 * df1['z_accel']
    ours['x_accel'] = 9.81 * ours['x_accel']
    ours['y_accel'] = 9.81 * ours['y_accel']
    ours['z_accel'] = 9.81 * ours['z_accel']

    myo = df1[df1['dataset'] == 'myogym']
    rec = df1[df1['dataset'] == 'lean']
    ours_arm = ours[ours['child'] == 3]
    ours_bench = ours[ours['child'] == 5]
    myo_arm = myo[myo['child'] == 3]
    myo_bench = myo[myo['child'] == 5]
    rec_arm = rec[rec['child'] == 3]
    rec_bench = rec[rec['child'] == 5]


    # Scale RecGym:

    # accelScale = 0
    # for col in ['x_accel', 'y_accel', 'z_accel']:
    #     accelScale += np.std(myo_bench[col]) / (3*np.std(rec_bench[col]))
    # gyroScale = 0
    # for col in ['x_gyro', 'y_gyro', 'z_gyro']:
    #     gyroScale += np.std(myo_bench[col]) / (3*np.std(rec_bench[col]))
    # for col in range(0,3):
    #     for row in range(len(rec_bench)):
    #         rec_bench.iloc[row, col] = (rec_bench.iloc[row, col] - 0.5) * accelScale
    # for col in range(3,6):
    #     for row in range(len(rec_bench)):
    #         rec_bench.iloc[row, col] = (rec_bench.iloc[row, col] - 0.5) * gyroScale


    # Switch MyoGym hands:

    # for row in range(len(myo_arm)):
    #     myo_arm.iloc[row, 0] = -myo_arm.iloc[row, 0]
    #     myo_arm.iloc[row, 4] = -myo_arm.iloc[row, 4]
    #     myo_arm.iloc[row, 5] = -myo_arm.iloc[row, 5]
    # for row in range(len(myo_bench)):
    #     myo_bench.iloc[row, 0] = -myo_bench.iloc[row, 0]
    #     myo_bench.iloc[row, 4] = -myo_bench.iloc[row, 4]
    #     myo_bench.iloc[row, 5] = -myo_bench.iloc[row, 5]


    # Shift MyoGym:

    # shiftX = 0
    # shiftY = 0
    # shiftZ = 0
    # shiftX += (np.mean(myo_arm['x_accel']) + np.mean(myo_bench['x_accel'])) / 2
    # shiftY += (np.mean(myo_arm['y_accel']) + np.mean(myo_bench['y_accel'])) / 2
    # shiftZ += (np.mean(myo_arm['z_accel']) + np.mean(myo_bench['z_accel'])) / 2
    # for row in range(len(myo_arm)):
    #     myo_arm.iloc[row, 0] -= shiftX
    #     myo_arm.iloc[row, 1] -= shiftY
    #     myo_arm.iloc[row, 2] -= shiftZ
    # for row in range(len(myo_bench)):
    #     myo_bench.iloc[row, 0] -= shiftX
    #     myo_bench.iloc[row, 1] -= shiftY
    #     myo_bench.iloc[row, 2] -= shiftZ


    # Just one participant:

    # ours_arm = ours_arm[ours_arm['participant'] == 22]
    # ours_bench = ours_bench[ours_bench['participant'] == 22]
    # myo_arm = myo_arm[myo_arm['participant'] == 1]
    # myo_bench = myo_bench[myo_bench['participant'] == 1]
    # rec_arm = rec_arm[rec_arm['participant'] == 12]
    # rec_bench = rec_bench[rec_bench['participant'] == 12]
    


    arm_reps_x_accel = getReps([ours_arm, myo_arm, rec_arm], 'x_accel')
    bench_reps_x_accel = getReps([ours_bench, myo_bench, rec_bench], 'x_accel')
    arm_reps_y_accel = getReps([ours_arm, myo_arm, rec_arm], 'y_accel')
    bench_reps_y_accel = getReps([ours_bench, myo_bench, rec_bench], 'y_accel')
    arm_reps_z_accel = getReps([ours_arm, myo_arm, rec_arm], 'z_accel')
    bench_reps_z_accel = getReps([ours_bench, myo_bench, rec_bench], 'z_accel')
    arm_reps_x_gyro = getReps([ours_arm, myo_arm, rec_arm], 'x_gyro')
    bench_reps_x_gyro = getReps([ours_bench, myo_bench, rec_bench], 'x_gyro')
    arm_reps_y_gyro = getReps([ours_arm, myo_arm, rec_arm], 'y_gyro')
    bench_reps_y_gyro = getReps([ours_bench, myo_bench, rec_bench], 'y_gyro')
    arm_reps_z_gyro = getReps([ours_arm, myo_arm, rec_arm], 'z_gyro')
    bench_reps_z_gyro = getReps([ours_bench, myo_bench, rec_bench], 'z_gyro')

    plot_average_reps(arm_reps_x_accel, title='Arm Curl for x_accel', labels=['OurData', 'MyoGym', 'Lean'])
    plot_average_reps(arm_reps_y_accel, title='Arm Curl for y_accel', labels=['OurData', 'MyoGym', 'Lean'])
    plot_average_reps(arm_reps_z_accel, title='Arm Curl for z_accel', labels=['OurData', 'MyoGym', 'Lean'])
    plot_average_reps(arm_reps_x_gyro, title='Arm Curl for x_gyro', labels=['OurData', 'MyoGym', 'Lean'])
    plot_average_reps(arm_reps_y_gyro, title='Arm Curl for y_gyro', labels=['OurData', 'MyoGym', 'Lean'])
    plot_average_reps(arm_reps_z_gyro, title='Arm Curl for z_gyro', labels=['OurData', 'MyoGym', 'Lean'])
    plot_average_reps(bench_reps_x_accel, title='Bench Press for x_accel', labels=['OurData', 'MyoGym', 'Lean'])
    plot_average_reps(bench_reps_y_accel, title='Bench Press for y_accel', labels=['OurData', 'MyoGym', 'Lean'])
    plot_average_reps(bench_reps_z_accel, title='Bench Press for z_accel', labels=['OurData', 'MyoGym', 'Lean'])
    plot_average_reps(bench_reps_x_gyro, title='Bench Press for x_gyro', labels=['OurData', 'MyoGym', 'Lean'])
    plot_average_reps(bench_reps_y_gyro, title='Bench Press for y_gyro', labels=['OurData', 'MyoGym', 'Lean'])
    plot_average_reps(bench_reps_z_gyro, title='Bench Press for z_gyro', labels=['OurData', 'MyoGym', 'Lean'])










main()

# df = pd.read_csv('OurData_v2.csv')
# df = pd.read_csv('golden_dataset_v5.csv')
# df = df[df['dataset'] == 'recgym']
# sets = getSets(df['ID'].values)
# for i in range (len(sets)-1):
#     showPeaks(df.iloc[sets[i]:sets[i+1]].reset_index(drop=True), 'x_accel')
#     #showPeaks(df.iloc[sets[i]:sets[i+1]].reset_index(drop=True), 'y_accel')
#     #showPeaks(df.iloc[sets[i]:sets[i+1]].reset_index(drop=True), 'z_accel')
#     showPeaks(df.iloc[sets[i]:sets[i+1]].reset_index(drop=True), 'x_gyro')
#     #showPeaks(df.iloc[sets[i]:sets[i+1]].reset_index(drop=True), 'y_gyro')
#     #showPeaks(df.iloc[sets[i]:sets[i+1]].reset_index(drop=True), 'z_gyro')