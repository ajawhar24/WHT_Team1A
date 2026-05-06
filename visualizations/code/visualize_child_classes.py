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


def getReps_by_child(df, col):
    """
    For a single parent-filtered DataFrame, return a dict mapping
    child -> list of rep arrays.
    """
    child_reps = {}
    sets = getSets(df["ID"].values)
    for i in range(len(sets) - 1):
        block = df.iloc[sets[i]:sets[i+1]].reset_index(drop=True)
        if block.empty:
            continue
        for child, c_df in block.groupby('child'):
            arr = c_df[col].values
            peaks, props = find_peaks(arr, prominence=0)
            if len(peaks) < 2:
                continue
            top10 = np.argsort(props["prominences"])[-10:]
            peaks = np.sort(peaks[top10])
            total = sum(peaks[j+1] - peaks[j] for j in range(len(peaks) - 1))
            average = total / (len(peaks) - 1)
            for j in range(len(peaks) - 1):
                if average * 0.6 < peaks[j+1] - peaks[j] < average * 1.4:
                    child_reps.setdefault(child, []).append(arr[peaks[j]:peaks[j+1]])
    return child_reps


def plot_children_overlay(child_reps_by_parent, parent_labels, col, common_length=100):
    """
    Two subplots (one per parent exercise), each overlaying one mean line per child.

    child_reps_by_parent : list of dicts {child_id: [rep_arrays,...]}, one per parent
    parent_labels        : list of subplot titles, one per parent
    col                  : signal column name (for y-axis label and file name)
    """
    fig, axes = plt.subplots(1, len(child_reps_by_parent),
                             figsize=(5 * len(child_reps_by_parent), 4),
                             sharey=True)

    cmap = plt.get_cmap('tab20')
    x = np.linspace(0, 100, common_length)

    for ax, child_reps, parent_label in zip(axes, child_reps_by_parent, parent_labels):
        if not child_reps:
            ax.set_title(f"{parent_label} (no data)")
            continue

        children = sorted(child_reps.keys())
        for idx, child in enumerate(children):
            reps = child_reps[child]
            resampled = [r for rep in reps
                         if (r := resample_rep(rep, common_length)) is not None]
            if not resampled:
                continue
            resampled = np.array(resampled)
            mean = np.mean(resampled, axis=0)
            ax.plot(x, mean, color=cmap(idx % 20), linewidth=1.5,
                    alpha=0.9, label=f"Child {child}")

        ax.set_title(parent_label)
        ax.set_xlabel('% of rep')
        if len(children) <= 10:
            ax.legend(fontsize='small', loc='best')

    if 'accel' in col:
        axes[0].set_ylabel('acceleration [m/s²]')
    else:
        axes[0].set_ylabel('angular velocity [deg/s]')

    fig.suptitle(f'OurData — {col}', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    fname = f'ourdata_{col}_children_overlay.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"Saved: {fname}")
    plt.show()


def main():
    ours = pd.read_csv('OurData_v5.csv')

    for axis in ('x', 'y', 'z'):
        ours[f'{axis}_accel'] = 9.81 * ours[f'{axis}_accel']

    parent1 = ours[ours['parent'] == 1]
    parent5 = ours[ours['parent'] == 5]

    parent_labels = ['Parent 1 (Arm Curl)', 'Parent 5 (Bench Press)']

    cols = ['x_accel', 'y_accel', 'z_accel',
            'x_gyro',  'y_gyro',  'z_gyro']

    for col in cols:
        child_reps_p1 = getReps_by_child(parent1, col)
        child_reps_p5 = getReps_by_child(parent5, col)
        plot_children_overlay(
            child_reps_by_parent=[child_reps_p1, child_reps_p5],
            parent_labels=parent_labels,
            col=col,
        )


main()