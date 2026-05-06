import pandas as pd
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt

def getSets(arr):
    sets = []
    for i in range(1, len(arr)):
        if arr[i] != arr[i-1]:
            sets.append(i)
    return [0] + sets + [len(arr)+1]

def resample_rep(rep, common_length):
    if len(rep) < 2:
        return None
    x_old = np.linspace(0, 1, len(rep))
    x_new = np.linspace(0, 1, common_length)
    f = interp1d(x_old, rep, kind='linear')
    return f(x_new)

def plot_average_reps(all_reps, labels=None, title=None, common_length=100):
    colors = ['blue', 'orange', 'green']
    fig, axes = plt.subplots(1, len(all_reps), figsize=(5 * len(all_reps), 4), sharey=True)

    if labels is None:
        labels = [f"Dataset {i+1}" for i in range(len(all_reps))]

    for ax, reps, label, color in zip(axes, all_reps, labels, colors):
        if len(reps) == 0:
            ax.set_title(f"{label} (no data)")
            continue

        resampled = []
        for rep in reps:
            r = resample_rep(rep, common_length)
            if r is not None:
                resampled.append(r)

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

def getReps(dfs, col):
    listListArray = []
    for df in dfs:
        listArray = []
        sets = getSets(df["ID"].values)
        for i in range(len(sets)-1):
            set_arr = df[col].values[sets[i]:sets[i+1]]
            peaks, props = find_peaks(set_arr, prominence=0)
            if len(peaks) < 2:
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
            for j in slices:
                rep = set_arr[peaks[j]:peaks[j+1]]
                listArray.append(rep)
        listListArray.append(listArray)
    return listListArray

def getReps_by_participant(dfs, col):
    """
    Return a list (one per dataset) of dicts mapping participant -> list of reps (raw arrays).
    """
    datasets_participant_reps = []
    for df in dfs:
        participant_reps = {}
        # split by contiguous ID blocks (sets) to isolate sessions
        sets = getSets(df["ID"].values)
        for i in range(len(sets)-1):
            block = df.iloc[sets[i]:sets[i+1]].reset_index(drop=True)
            if block.empty:
                continue
            # group by participant within this block
            for participant, p_df in block.groupby('participant'):
                arr = p_df[col].values
                peaks, props = find_peaks(arr, prominence=0)
                if len(peaks) < 2:
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
                for j in slices:
                    rep = arr[peaks[j]:peaks[j+1]]
                    participant_reps.setdefault(participant, []).append(rep)
        datasets_participant_reps.append(participant_reps)
    return datasets_participant_reps

def plot_participant_means(datasets_participant_reps, labels=None, title=None, common_length=100):
    """
    datasets_participant_reps: list (per dataset) of dicts {participant_id: [rep_arrays,...]}
    """
    colors = ['blue', 'orange', 'green']
    fig, axes = plt.subplots(1, len(datasets_participant_reps), figsize=(5 * len(datasets_participant_reps), 4), sharey=True)

    if labels is None:
        labels = [f"Dataset {i+1}" for i in range(len(datasets_participant_reps))]

    for ax, participant_dict, label, color in zip(axes, datasets_participant_reps, labels, colors):
        if not participant_dict:
            ax.set_title(f"{label} (no data)")
            continue

        x = np.linspace(0, 100, common_length)
        # choose a colormap for participants to ensure distinct lines
        cmap = plt.get_cmap('tab20')
        participants = sorted(participant_dict.keys())
        for idx, participant in enumerate(participants):
            reps = participant_dict[participant]
            resampled = []
            for rep in reps:
                r = resample_rep(rep, common_length)
                if r is not None:
                    resampled.append(r)
            if len(resampled) == 0:
                continue
            resampled = np.array(resampled)
            mean = np.mean(resampled, axis=0)
            # plot participant mean with thinner line and some transparency
            ax.plot(x, mean, color=cmap(idx % 20), linewidth=1.5, alpha=0.9, label=f"P{participant}")
        ax.set_title(label)
        ax.set_xlabel('% of rep')
        # show legend if not too many participants
        if len(participants) <= 10:
            ax.legend(fontsize='small', loc='best')

    if 'accel' in title:
        axes[0].set_ylabel('acceleration [m/s^2]')
    else:
        axes[0].set_ylabel('angular velocity [deg/s]')

    if title:
        fig.suptitle(title, fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        plt.tight_layout()

    plt.savefig(f'{title}_participant_means.png', dpi=300, bbox_inches='tight')
    plt.show()

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

    # compute reps (existing mean+std plots)
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

    # compute per-participant reps
    arm_participant_x_accel = getReps_by_participant([ours_arm, myo_arm, rec_arm], 'x_accel')
    bench_participant_x_accel = getReps_by_participant([ours_bench, myo_bench, rec_bench], 'x_accel')
    arm_participant_y_accel = getReps_by_participant([ours_arm, myo_arm, rec_arm], 'y_accel')
    bench_participant_y_accel = getReps_by_participant([ours_bench, myo_bench, rec_bench], 'y_accel')
    arm_participant_z_accel = getReps_by_participant([ours_arm, myo_arm, rec_arm], 'z_accel')
    bench_participant_z_accel = getReps_by_participant([ours_bench, myo_bench, rec_bench], 'z_accel')
    arm_participant_x_gyro = getReps_by_participant([ours_arm, myo_arm, rec_arm], 'x_gyro')
    bench_participant_x_gyro = getReps_by_participant([ours_bench, myo_bench, rec_bench], 'x_gyro')
    arm_participant_y_gyro = getReps_by_participant([ours_arm, myo_arm, rec_arm], 'y_gyro')
    bench_participant_y_gyro = getReps_by_participant([ours_bench, myo_bench, rec_bench], 'y_gyro')
    arm_participant_z_gyro = getReps_by_participant([ours_arm, myo_arm, rec_arm], 'z_gyro')
    bench_participant_z_gyro = getReps_by_participant([ours_bench, myo_bench, rec_bench], 'z_gyro')

    # original mean+std plots
    # plot_average_reps(arm_reps_x_accel, title='Arm Curl for x_accel', labels=['OurData', 'MyoGym', 'Lean'])
    # plot_average_reps(arm_reps_y_accel, title='Arm Curl for y_accel', labels=['OurData', 'MyoGym', 'Lean'])
    # plot_average_reps(arm_reps_z_accel, title='Arm Curl for z_accel', labels=['OurData', 'MyoGym', 'Lean'])
    # plot_average_reps(arm_reps_x_gyro, title='Arm Curl for x_gyro', labels=['OurData', 'MyoGym', 'Lean'])
    # plot_average_reps(arm_reps_y_gyro, title='Arm Curl for y_gyro', labels=['OurData', 'MyoGym', 'Lean'])
    # plot_average_reps(arm_reps_z_gyro, title='Arm Curl for z_gyro', labels=['OurData', 'MyoGym', 'Lean'])
    # plot_average_reps(bench_reps_x_accel, title='Bench Press for x_accel', labels=['OurData', 'MyoGym', 'Lean'])
    # plot_average_reps(bench_reps_y_accel, title='Bench Press for y_accel', labels=['OurData', 'MyoGym', 'Lean'])
    # plot_average_reps(bench_reps_z_accel, title='Bench Press for z_accel', labels=['OurData', 'MyoGym', 'Lean'])
    # plot_average_reps(bench_reps_x_gyro, title='Bench Press for x_gyro', labels=['OurData', 'MyoGym', 'Lean'])
    # plot_average_reps(bench_reps_y_gyro, title='Bench Press for y_gyro', labels=['OurData', 'MyoGym', 'Lean'])
    # plot_average_reps(bench_reps_z_gyro, title='Bench Press for z_gyro', labels=['OurData', 'MyoGym', 'Lean'])

    # new per-participant mean overlay plots
    plot_participant_means(arm_participant_x_accel, title='Arm Curl x_accel Participant Means', labels=['OurData', 'MyoGym', 'Lean'])
    plot_participant_means(arm_participant_y_accel, title='Arm Curl y_accel Participant Means', labels=['OurData', 'MyoGym', 'Lean'])
    plot_participant_means(arm_participant_z_accel, title='Arm Curl z_accel Participant Means', labels=['OurData', 'MyoGym', 'Lean'])
    plot_participant_means(arm_participant_x_gyro, title='Arm Curl x_gyro Participant Means', labels=['OurData', 'MyoGym', 'Lean'])
    plot_participant_means(arm_participant_y_gyro, title='Arm Curl y_gyro Participant Means', labels=['OurData', 'MyoGym', 'Lean'])
    plot_participant_means(arm_participant_z_gyro, title='Arm Curl z_gyro Participant Means', labels=['OurData', 'MyoGym', 'Lean'])
    plot_participant_means(bench_participant_x_accel, title='Bench Press x_accel Participant Means', labels=['OurData', 'MyoGym', 'Lean'])
    plot_participant_means(bench_participant_y_accel, title='Bench Press y_accel Participant Means', labels=['OurData', 'MyoGym', 'Lean'])
    plot_participant_means(bench_participant_z_accel, title='Bench Press z_accel Participant Means', labels=['OurData', 'MyoGym', 'Lean'])
    plot_participant_means(bench_participant_x_gyro, title='Bench Press x_gyro Participant Means', labels=['OurData', 'MyoGym', 'Lean'])
    plot_participant_means(bench_participant_y_gyro, title='Bench Press y_gyro Participant Means', labels=['OurData', 'MyoGym', 'Lean'])
    plot_participant_means(bench_participant_z_gyro, title='Bench Press z_gyro Participant Means', labels=['OurData', 'MyoGym', 'Lean'])

if __name__ == "__main__":
    main()