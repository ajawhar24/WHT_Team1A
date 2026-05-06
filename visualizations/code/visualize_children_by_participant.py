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


def getReps_by_participant(df, col):
    """
    For a DataFrame (already filtered to one child class), return a dict mapping
    participant -> list of rep arrays.
    """
    participant_reps = {}
    sets = getSets(df["ID"].values)
    for i in range(len(sets) - 1):
        block = df.iloc[sets[i]:sets[i+1]].reset_index(drop=True)
        if block.empty:
            continue
        for participant, p_df in block.groupby('participant'):
            arr = p_df[col].values
            peaks, props = find_peaks(arr, prominence=0)
            if len(peaks) < 2:
                continue
            top10 = np.argsort(props["prominences"])[-10:]
            peaks = np.sort(peaks[top10])
            total = sum(peaks[j+1] - peaks[j] for j in range(len(peaks) - 1))
            average = total / (len(peaks) - 1)
            for j in range(len(peaks) - 1):
                if average * 0.6 < peaks[j+1] - peaks[j] < average * 1.4:
                    participant_reps.setdefault(participant, []).append(arr[peaks[j]:peaks[j+1]])
    return participant_reps


def plot_participants_by_child(parent_df, parent_label, col, common_length=100):
    """
    One subplot per child class within this parent group.
    Each subplot overlays one mean line per participant.

    parent_df    : DataFrame filtered to a single parent
    parent_label : string used in the figure title and saved filename
    col          : signal column name
    """
    children = sorted(parent_df['child'].unique())
    n = len(children)
    if n == 0:
        print(f"No children found for {parent_label}")
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    cmap = plt.get_cmap('tab20')
    x = np.linspace(0, 100, common_length)

    for ax, child in zip(axes, children):
        child_df = parent_df[parent_df['child'] == child]
        participant_reps = getReps_by_participant(child_df, col)

        if not participant_reps:
            ax.set_title(f"Child {child} (no data)")
            continue

        participants = sorted(participant_reps.keys())
        for idx, participant in enumerate(participants):
            reps = participant_reps[participant]
            resampled = [r for rep in reps
                         if (r := resample_rep(rep, common_length)) is not None]
            if not resampled:
                continue
            resampled = np.array(resampled)
            mean = np.mean(resampled, axis=0)
            ax.plot(x, mean, color=cmap(idx % 20), linewidth=1.5,
                    alpha=0.9, label=f"P{participant}")

        ax.set_title(f"Child {child}")
        ax.set_xlabel('% of rep')
        if len(participants) <= 10:
            ax.legend(fontsize='small', loc='best')

    if 'accel' in col:
        axes[0].set_ylabel('acceleration [m/s²]')
    else:
        axes[0].set_ylabel('angular velocity [deg/s]')

    fig.suptitle(f'{parent_label} — {col}', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    fname = f'ourdata_{parent_label.lower().replace(" ", "_")}_{col}_by_child.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"Saved: {fname}")
    plt.show()


def main():
    ours = pd.read_csv('OurData_v5.csv')

    for axis in ('x', 'y', 'z'):
        ours[f'{axis}_accel'] = 9.81 * ours[f'{axis}_accel']

    parent1 = ours[ours['parent'] == 1]
    parent5 = ours[ours['parent'] == 5]

    cols = ['x_accel', 'y_accel', 'z_accel',
            'x_gyro',  'y_gyro',  'z_gyro']

    for col in cols:
        plot_participants_by_child(parent1, 'Arm Curl', col)
        plot_participants_by_child(parent5, 'Bench Press', col)


main()