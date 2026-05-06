import numpy as np
import scipy.stats as stats
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler


def to_categorical(y, num_classes):
    return np.eye(num_classes, dtype=np.float32)[y.astype(int)]


def load_data_LOSO(data_path, dataset, test_subject, sensor="imu"):
    """
    Load data and perform a Leave-One-Subject-Out split.

    All rows where participant == test_subject become the test set.
    All remaining rows become the training set.
    Windowing is applied independently to each split so that no window
    ever straddles the train/test boundary.

    Parameters
    ----------
    data_path    : str   – path to the CSV file
    dataset      : str   – dataset name used to select the label map
    test_subject : int   – participant ID to hold out as the test fold
    sensor       : str   – 'imu' | 'cap' | 'combine'

    Returns
    -------
    X_train, y_train, X_test, y_test
    """

    # ------------------------------------------------------------------ #
    # 1. Load and label-map
    # ------------------------------------------------------------------ #
    df = pd.read_csv(data_path)

    if dataset == 'GymCap':
        label_map = {
            "Adductor": 1, "ArmCurl": 2, "BenchPress": 3, "LegCurl": 4,
            "LegPress": 5, "Null": 6, "Riding": 7, "RopeSkipping": 8,
            "Running": 9, "Squat": 10, "StairClimber": 11, "Walking": 12
        }
    elif dataset == 'Golden':
        label_map = {
             0:0,
            1:1,
            5:2 }
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    df['participant'] = df['participant'].astype(int)
    df['parent'] = df['parent'].map(label_map)
    df = df.dropna(subset=['parent'])   # drop unmapped labels BEFORE casting
    df['parent'] = df['parent'].astype(int)

    # ------------------------------------------------------------------ #
    # 2. Feature columns
    # ------------------------------------------------------------------ #
    if sensor == "cap":
        feature_cols = ["C_1"]
    elif sensor == "imu":
        feature_cols = ["x_accel", "y_accel", "z_accel","x_gyro", "y_gyro", "z_gyro"]
    elif sensor == "combine":
        feature_cols = ["x_accel", "y_accel", "z_accel", "x_gyro", "y_gyro", "z_gyro", "C_1"]
    else:
        raise ValueError(f"Unknown sensor type: {sensor}")

    label_col = "parent"

    # Drop any rows where sensor readings are NaN — these cause NaN loss
    rows_before = len(df)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    rows_after = len(df)
    if rows_before != rows_after:
        print(f"  Dropped {rows_before - rows_after} rows with NaN sensor values")

    # ------------------------------------------------------------------ #
    # 3. LOSO split  (kept in temporal order within each subject)
    # ------------------------------------------------------------------ #
    train_df = df[df['participant'] != test_subject].reset_index(drop=True)
    test_df  = df[df['participant'] == test_subject].reset_index(drop=True)


    print(f"\n[LOSO] Test subject: {test_subject}")
    print(f"  Train rows: {len(train_df)} | Test rows: {len(test_df)}")

    # ------------------------------------------------------------------ #
    # 4. Vectorised windowing
    # ------------------------------------------------------------------ #
    def create_windows(df_split, feat_cols, lbl_col, window_size=40):
        """
        Window the data for every subject inside df_split independently so
        that windows never span subject boundaries.
        """
        X_list, y_list = [], []
        for pid, grp in df_split.groupby('participant', sort=True):

            grp = grp.reset_index(drop=True)
            n_samples = len(grp)
            n_windows = n_samples // window_size
            if n_windows == 0:
                print(f"  Warning: participant {pid} has fewer than {window_size} "
                      f"rows – skipping.")
                continue
            cutoff = n_windows * window_size

            x_raw = grp[feat_cols].iloc[:cutoff].to_numpy()
            X_windowed = x_raw.reshape(n_windows, window_size, len(feat_cols))

            y_raw = grp[lbl_col].iloc[:cutoff].to_numpy()
            y_reshaped = y_raw.reshape(n_windows, window_size)
            y_windowed, _ = stats.mode(y_reshaped, axis=1, keepdims=False)
            y_windowed = y_windowed.flatten().astype(int) - 1
            valid = y_windowed != -1
            X_windowed = X_windowed[valid]
            y_windowed = y_windowed[valid]
            X_list.append(X_windowed)
            y_list.append(y_windowed)

        X_out = np.concatenate(X_list, axis=0)
        y_out = np.concatenate(y_list, axis=0)
        return X_out, y_out

    X_train, y_train = create_windows(train_df, feature_cols, label_col)
    X_test,  y_test  = create_windows(test_df,  feature_cols, label_col)

    print(f"  X_train: {X_train.shape}  |  X_test: {X_test.shape}")

    # Class distribution in training fold
    unique_vals, counts = np.unique(y_train, return_counts=True)
    for v, c in zip(unique_vals, counts):
        print(f"  Train class {v}: {c} samples")

    return X_train, y_train, X_test, y_test


def get_data(data_path, dataset, test_subject, sensor="imu", shuffle_train=True):
    """
    Wrapper that loads LOSO data, optionally shuffles the training set,
    reshapes arrays for the model, and one-hot encodes the labels.

    Input shape required by Post_Fusion: (N, 1, time_steps, channels)

    Returns
    -------
    X_train, y_train, y_train_onehot, X_test, y_test, y_test_onehot
    """
    X_train, y_train, X_test, y_test = load_data_LOSO(
        data_path, dataset, test_subject, sensor
    )

    if shuffle_train:
        X_train, y_train = shuffle(X_train, y_train, random_state=42)

    # ------------------------------------------------------------------ #
    # Normalise: fit scaler on train windows, apply to both splits.
    # Reshape to 2D for scaler, then back to 3D.
    # ------------------------------------------------------------------ #
    N_tr, T, N_ch = X_train.shape
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, N_ch)).reshape(N_tr, T, N_ch)

    N_te, T, N_ch = X_test.shape
    X_test = scaler.transform(X_test.reshape(-1, N_ch)).reshape(N_te, T, N_ch)

    # X is currently (N, window_size, n_channels) → (N, 1, window_size, n_channels)
    X_train = X_train.reshape(N_tr, 1, T, N_ch).astype(np.float32)
    y_train_onehot = to_categorical(y_train, num_classes=2)

    X_test = X_test.reshape(N_te, 1, T, N_ch).astype(np.float32)
    y_test_onehot = to_categorical(y_test, num_classes=2)

    return X_train, y_train, y_train_onehot, X_test, y_test, y_test_onehot


def get_unique_subjects(data_path):
    """Return a sorted list of unique participant IDs from the CSV."""
    df = pd.read_csv(data_path, usecols=['participant'])
    return sorted(df['participant'].unique().tolist())


def load_new_dataset(data_path, dataset, sensor="imu", window_size=40):
    """
    Load and window all subjects from a new CSV for ensemble inference.
    No train/test split is performed — the entire dataset is returned.
    Normalisation is fit on all windows in the new dataset.

    Returns
    -------
    X            : (N, 1, window_size, n_channels)  float32
    y            : (N,)  int
    subject_ids  : (N,)  – participant ID for each window
    """
    df = pd.read_csv(data_path)

    if dataset == 'GymCap':
        label_map = {
            "Adductor": 1, "ArmCurl": 2, "BenchPress": 3, "LegCurl": 4,
            "LegPress": 5, "Null": 6, "Riding": 7, "RopeSkipping": 8,
            "Running": 9, "Squat": 10, "StairClimber": 11, "Walking": 12
        }
    elif dataset == 'Golden':
        label_map = {0: 0, 1: 1, 5: 2}
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    df['participant'] = df['participant'].astype(int)
    df['parent'] = df['parent'].map(label_map)
    df = df.dropna(subset=['parent'])
    df['parent'] = df['parent'].astype(int)

    if sensor == "cap":
        feature_cols = ["C_1"]
    elif sensor == "imu":
        feature_cols = ["x_accel", "y_accel", "z_accel","x_gyro", "y_gyro", "z_gyro"]
    elif sensor == "combine":
        feature_cols = ["x_accel", "y_accel", "z_accel", "x_gyro", "y_gyro", "z_gyro", "C_1"]
    else:
        raise ValueError(f"Unknown sensor type: {sensor}")

    rows_before = len(df)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    if rows_before != len(df):
        print(f"  Dropped {rows_before - len(df)} rows with NaN sensor values")

    X_list, y_list, sub_list = [], [], []

    for pid, grp in df.groupby('participant', sort=True):
        grp = grp.reset_index(drop=True)
        n_windows = len(grp) // window_size
        if n_windows == 0:
            print(f"  Warning: participant {pid} has fewer than {window_size} rows – skipping.")
            continue
        cutoff = n_windows * window_size

        x_raw = grp[feature_cols].iloc[:cutoff].to_numpy()
        X_windowed = x_raw.reshape(n_windows, window_size, len(feature_cols))

        y_raw = grp['parent'].iloc[:cutoff].to_numpy()
        y_reshaped = y_raw.reshape(n_windows, window_size)
        y_windowed, _ = stats.mode(y_reshaped, axis=1, keepdims=False)
        y_windowed = y_windowed.flatten().astype(int) - 1

        valid = y_windowed != -1
        X_list.append(X_windowed[valid])
        y_list.append(y_windowed[valid])
        sub_list.append(np.full(valid.sum(), pid))

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    subject_ids = np.concatenate(sub_list, axis=0)

    # Fit scaler on all new-dataset windows (no training data involved)
    N, T, C = X.shape
    scaler = StandardScaler()
    X = scaler.fit_transform(X.reshape(-1, C)).reshape(N, T, C)
    X = X.reshape(N, 1, T, C).astype(np.float32)

    print(f"  New dataset loaded: {N} windows, {len(np.unique(subject_ids))} subjects")
    unique_vals, counts = np.unique(y, return_counts=True)
    for v, c in zip(unique_vals, counts):
        print(f"  Class {v}: {c} windows")

    return X, y, subject_ids
