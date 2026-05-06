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
   
    #df = df[df['dataset'] != 'ourdata']
    #print("training data: ", df['dataset'].unique())
    print(df['parent'].unique())
    if dataset == 'GymCap':
        label_map = {
            "Adductor": 1, "ArmCurl": 2, "BenchPress": 3, "LegCurl": 4,
            "LegPress": 5, "Null": 6, "Riding": 7, "RopeSkipping": 8,
            "Running": 9, "Squat": 10, "StairClimber": 11, "Walking": 12
        }
    elif dataset == 'Golden':
        # label_map = {
        #      0:0,
        #     3:1,
        #     5:2 }
        label_map = {
             1:1,
            2:1,
            3:1,
            4:1,
            5:2,
            6:2,
            7:2,
            8:2,
            9:2,
            10:2}
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    df['participant'] = df['participant'].astype(int)
    print(df["parent"].unique())
    print(df["child"].unique())
    df['parent'] = df['child'].map(label_map)
    print(df["parent"].unique())
    df = df.dropna(subset=['parent'])   # drop unmapped labels BEFORE casting
    df['parent'] = df['parent'].astype(int)
    print(df["parent"].unique())

    # ------------------------------------------------------------------ #
    # 2. Feature columns
    # ------------------------------------------------------------------ #
    if sensor == "cap":
        feature_cols = ["C_1"]
    elif sensor == "imu":
        feature_cols = ["x_accel", "y_accel", "z_accel", "x_gyro", "y_gyro", "z_gyro"]
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
    y_train_onehot = to_categorical(y_train, num_classes=10)

    X_test = X_test.reshape(N_te, 1, T, N_ch).astype(np.float32)
    y_test_onehot = to_categorical(y_test, num_classes=10)

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
    y            : (N,)  int  – parent class index per window
    subject_ids  : (N,)  – participant ID for each window
    y_child      : (N,)  int or None  – child class index per window (Golden only)
    """
    df = pd.read_csv(data_path)

    if dataset == 'GymCap':
        label_map = {
            "Adductor": 1, "ArmCurl": 2, "BenchPress": 3, "LegCurl": 4,
            "LegPress": 5, "Null": 6, "Riding": 7, "RopeSkipping": 8,
            "Running": 9, "Squat": 10, "StairClimber": 11, "Walking": 12
        }
    elif dataset == 'Golden':
        label_map = {
             1:1,
            2:1,
            3:1,
            4:1,
            5:2,
            6:2,
            7:2,
            8:2,
            9:2,
            10:2}
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    df['participant'] = df['participant'].astype(int)
    print(df['child'].unique())

    # Save raw child IDs before parent-label mapping (used for child-class tracking)
    df['child_raw'] = df['child'].copy()

    df['child'] = df['child'].map(label_map)
    df = df.dropna(subset=['child'])
    df['child'] = df['child'].astype(int)
    print(df['child'].unique())

    # Map raw child IDs to consecutive child class indices (Golden only)
    track_child = dataset == 'Golden'
    if track_child:
        df['child_idx'] = df['child_raw'].map(GOLDEN_CHILD_CLASS_MAP)
        df = df.dropna(subset=['child_idx']).reset_index(drop=True)
        df['child_idx'] = df['child_idx'].astype(int)

    if sensor == "cap":
        feature_cols = ["C_1"]
    elif sensor == "imu":
        feature_cols = ["x_accel", "y_accel", "z_accel", "x_gyro", "y_gyro", "z_gyro"]
    elif sensor == "combine":
        feature_cols = ["x_accel", "y_accel", "z_accel", "x_gyro", "y_gyro", "z_gyro", "C_1"]
    else:
        raise ValueError(f"Unknown sensor type: {sensor}")

    rows_before = len(df)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    if rows_before != len(df):
        print(f"  Dropped {rows_before - len(df)} rows with NaN sensor values")

    X_list, y_list, sub_list, yc_list = [], [], [], []

    for pid, grp in df.groupby('participant', sort=True):
        grp = grp.reset_index(drop=True)
        n_windows = len(grp) // window_size
        if n_windows == 0:
            print(f"  Warning: participant {pid} has fewer than {window_size} rows – skipping.")
            continue
        cutoff = n_windows * window_size

        x_raw = grp[feature_cols].iloc[:cutoff].to_numpy()
        X_windowed = x_raw.reshape(n_windows, window_size, len(feature_cols))

        y_raw = grp['child'].iloc[:cutoff].to_numpy()
        y_reshaped = y_raw.reshape(n_windows, window_size)
        y_windowed, _ = stats.mode(y_reshaped, axis=1, keepdims=False)
        y_windowed = y_windowed.flatten().astype(int) - 1

        valid = y_windowed != -1
        X_list.append(X_windowed[valid])
        y_list.append(y_windowed[valid])
        sub_list.append(np.full(valid.sum(), pid))

        if track_child:
            yc_raw = grp['child_idx'].iloc[:cutoff].to_numpy()
            yc_reshaped = yc_raw.reshape(n_windows, window_size)
            yc_windowed, _ = stats.mode(yc_reshaped, axis=1, keepdims=False)
            yc_windowed = yc_windowed.flatten().astype(int)
            yc_list.append(yc_windowed[valid])

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    subject_ids = np.concatenate(sub_list, axis=0)
    y_child = np.concatenate(yc_list, axis=0) if track_child else None

    # Fit scaler on all new-dataset windows (no training data involved)
    N, T, C = X.shape
    scaler = StandardScaler()
    X = scaler.fit_transform(X.reshape(-1, C)).reshape(N, T, C)
    X = X.reshape(N, 1, T, C).astype(np.float32)

    print(f"  New dataset loaded: {N} windows, {len(np.unique(subject_ids))} subjects")
    unique_vals, counts = np.unique(y, return_counts=True)
    for v, c in zip(unique_vals, counts):
        print(f"  Class {v}: {c} windows")

    return X, y, subject_ids, y_child


# ======================================================================
# Child-class label maps (used by load_new_dataset and test_general)
# ======================================================================
# Raw child-ID in OurData → consecutive child class index (0-based)
GOLDEN_CHILD_CLASS_MAP = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9}
# Child class index → parent class index (0=arm curl, 1=bench press)
GOLDEN_CHILD_TO_PARENT = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1}
GOLDEN_CHILD_LABELS = [
    '(1,1)', '(1,2)', '(1,3)', '(1,4)',
    '(5,5)', '(5,6)', '(5,7)', '(5,8)',
    '(5,9)', '(5,10)'
]
GOLDEN_PARENT_LABELS = ['arm curl', 'bench press']

# ======================================================================
# All 10 (parent, child) pairs in OurData → consecutive class index
# ======================================================================
OURDATA_PAIR_TO_CLASS = {
    (1, 1): 0, (1, 2): 1, (1, 3): 2, (1, 4): 3,
    (5, 5): 4, (5, 6): 5, (5, 7): 6, (5, 8): 7, (5, 9): 8, (5, 10): 9,
}
OURDATA_10CLASS_LABELS = [
    '(1,1)', '(1,2)', '(1,3)', '(1,4)',
    '(5,5)', '(5,6)', '(5,7)', '(5,8)', '(5,9)', '(5,10)',
]
# 10-class index → parent class index (0=arm curl, 1=bench press)
OURDATA_10CLASS_TO_PARENT = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1}

# ======================================================================
# Fine-tuning constants for the progressive ID-based approach
# ======================================================================
# (parent, child) pairs the model was trained on — excluded from child-class data
PARENT_CLASS_PAIRS_GOLDEN = {(1, 3), (5, 5)}
# parent column value → class index  (same as the model's output)
PARENT_LABEL_MAP_GOLDEN   = {1: 0, 5: 1}
# Participants in OurData: 15 is held out for evaluation, 16-19 for fine-tuning
CHILD_EVAL_PARTICIPANTS       = [15]
CHILD_FINETUNE_PARTICIPANTS   = [16, 17, 18, 19]


def load_child_class_data(data_path, dataset="Golden", window_size=40, sensor="imu"):
    """
    Load child-class exercise data from OurData, segmenting per unique exercise ID.

    Excludes the parent-class training pairs (e.g. (1,3) and (5,5)) so that only
    the unseen child-class exercises are returned.  Labels are assigned based on
    the *parent* column (same mapping as during model training: 1→0, 5→1), so the
    model can be fine-tuned with its original 2-class output head unchanged.

    Each window is tagged with its (parent, child) pair, unique exercise ID, and
    participant number so the caller can split and track results flexibly.

    Parameters
    ----------
    data_path   : str  – path to the child-class CSV (e.g. OurData_v5.csv)
    dataset     : str  – 'Golden'
    window_size : int
    sensor      : str  – 'imu' | 'cap' | 'combine'

    Returns
    -------
    X            : (N, 1, T, C) float32  – normalised windows
    y            : (N,) int64            – parent class label (0=arm curl, 1=bench press)
    pairs        : (N, 2) int64          – [parent, child] raw IDs per window
    ids          : (N,) int64            – exercise-set ID per window
    participants : (N,) int64            – participant number per window
    child_pairs  : list of (parent, child) tuples  – all unique pairs in the data
    parent_label_map : dict used for label assignment
    """
    if dataset != 'Golden':
        raise ValueError(f"load_child_class_data not implemented for dataset '{dataset}'")

    parent_class_pairs = PARENT_CLASS_PAIRS_GOLDEN
    parent_label_map   = PARENT_LABEL_MAP_GOLDEN

    if sensor == "imu":
        feature_cols = ["x_accel", "y_accel", "z_accel", "x_gyro", "y_gyro", "z_gyro"]
    elif sensor == "cap":
        feature_cols = ["C_1"]
    elif sensor == "combine":
        feature_cols = ["x_accel", "y_accel", "z_accel", "x_gyro", "y_gyro", "z_gyro", "C_1"]
    else:
        raise ValueError(f"Unknown sensor: {sensor}")

    df = pd.read_csv(data_path)
    df['participant'] = df['participant'].astype(int)
    df['parent']      = df['parent'].astype(int)
    df['child']       = df['child'].astype(int)
    df['ID']          = df['ID'].astype(int)

    # Drop parent-class training pairs  (model already knows these)
    pair_mask = pd.Series(False, index=df.index)
    for p, c in parent_class_pairs:
        pair_mask |= (df['parent'] == p) & (df['child'] == c)
    df = df[~pair_mask].reset_index(drop=True)

    # Drop rows whose parent value has no label mapping (e.g. rest class)
    df = df[df['parent'].isin(parent_label_map)].reset_index(drop=True)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    # Collect unique (parent, child) pairs present in the filtered data
    child_pairs = sorted(
        df.groupby(['parent', 'child']).size().reset_index()
          .apply(lambda r: (int(r['parent']), int(r['child'])), axis=1)
          .tolist()
    )

    print(f"\n[Child-class data] {len(df):,} rows | "
          f"{df['ID'].nunique()} IDs | "
          f"{df['participant'].nunique()} participants")
    print(f"  Child-class pairs : {child_pairs}")
    print(f"  Participants      : {sorted(df['participant'].unique().tolist())}")

    # Segment per unique exercise ID (preserves temporal integrity)
    X_list, y_list, pair_list, id_list, part_list = [], [], [], [], []
    for uid in sorted(df['ID'].unique()):
        sub = df[df['ID'] == uid].reset_index(drop=True)
        if len(sub) < window_size:
            continue

        parent_val  = int(sub['parent'].iloc[0])
        child_val   = int(sub['child'].iloc[0])
        label       = parent_label_map[parent_val]
        participant = int(sub['participant'].iloc[0])

        n_windows = len(sub) // window_size
        cutoff    = n_windows * window_size

        x_raw      = sub[feature_cols].iloc[:cutoff].to_numpy()
        X_windowed = x_raw.reshape(n_windows, window_size, len(feature_cols))

        X_list.append(X_windowed)
        y_list.extend([label]      * n_windows)
        pair_list.extend([(parent_val, child_val)] * n_windows)
        id_list.extend([uid]        * n_windows)
        part_list.extend([participant] * n_windows)

    X            = np.concatenate(X_list, axis=0)          # (N, T, C)
    y            = np.array(y_list,    dtype=np.int64)
    pairs        = np.array(pair_list, dtype=np.int64)      # (N, 2)
    ids          = np.array(id_list,   dtype=np.int64)
    participants = np.array(part_list, dtype=np.int64)

    # Normalise across all windows (single global scaler)
    N, T, C = X.shape
    scaler  = StandardScaler()
    X       = scaler.fit_transform(X.reshape(-1, C)).reshape(N, T, C)
    X       = X.reshape(N, 1, T, C).astype(np.float32)

    print(f"  Segmented : {N} windows total")
    print(f"    Label 0 (arm curl)    : {(y == 0).sum()}")
    print(f"    Label 1 (bench press) : {(y == 1).sum()}")

    return X, y, pairs, ids, participants, child_pairs, parent_label_map


def load_ourdata_all(data_path, dataset="Golden", window_size=40, sensor="imu"):
    """
    Load ALL OurData windows (all (parent, child) pairs, all participants),
    including the parent-class training pairs.  Used for within-participant
    training/testing where the model is built fresh per subject.

    Labels are assigned by the parent column: {1 → 0 (arm curl), 5 → 1 (bench press)}.
    Windows are NOT normalised here — the caller normalises per-participant
    after splitting into train/test.

    Parameters
    ----------
    data_path   : str
    dataset     : str  – 'Golden'
    window_size : int
    sensor      : str  – 'imu' | 'cap' | 'combine'

    Returns
    -------
    X            : (N, T, C) float32  – raw, un-normalised
    y            : (N,) int64         – 0=arm curl, 1=bench press  (parent class)
    y_child      : (N,) int64         – 0-9 child-class index (see OURDATA_PAIR_TO_CLASS)
    ids          : (N,) int64         – exercise-set ID per window
    participants : (N,) int64         – participant number per window
    pairs_arr    : (N, 2) int64       – (parent, child) values per window
    all_pairs    : sorted list of (parent, child) tuples present in the data
    """
    if dataset != 'Golden':
        raise ValueError(f"load_ourdata_all not implemented for dataset '{dataset}'")

    parent_label_map = PARENT_LABEL_MAP_GOLDEN   # {1: 0, 5: 1}

    if sensor == "imu":
        feature_cols = ["x_accel", "y_accel", "z_accel", "x_gyro", "y_gyro", "z_gyro"]
    elif sensor == "cap":
        feature_cols = ["C_1"]
    elif sensor == "combine":
        feature_cols = ["x_accel", "y_accel", "z_accel", "x_gyro", "y_gyro", "z_gyro", "C_1"]
    else:
        raise ValueError(f"Unknown sensor: {sensor}")

    df = pd.read_csv(data_path)
    df['participant'] = df['participant'].astype(int)
    df['parent']      = df['parent'].astype(int)
    df['child']       = df['child'].astype(int)
    df['ID']          = df['ID'].astype(int)

    df = df[df['parent'].isin(parent_label_map)].reset_index(drop=True)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    all_pairs = sorted(
        df.groupby(['parent', 'child']).size().reset_index()
          .apply(lambda r: (int(r['parent']), int(r['child'])), axis=1).tolist()
    )

    X_list, y_list, yc_list, id_list, part_list, pair_list = [], [], [], [], [], []
    for uid in sorted(df['ID'].unique()):
        sub = df[df['ID'] == uid].reset_index(drop=True)
        if len(sub) < window_size:
            continue

        parent_val  = int(sub['parent'].iloc[0])
        child_val   = int(sub['child'].iloc[0])
        label       = parent_label_map[parent_val]
        child_label = OURDATA_PAIR_TO_CLASS.get((parent_val, child_val), -1)
        participant = int(sub['participant'].iloc[0])

        n_windows = len(sub) // window_size
        cutoff    = n_windows * window_size

        x_raw = sub[feature_cols].iloc[:cutoff].to_numpy()
        X_list.append(x_raw.reshape(n_windows, window_size, len(feature_cols)))
        y_list.extend([label]        * n_windows)
        yc_list.extend([child_label] * n_windows)
        id_list.extend([uid]         * n_windows)
        part_list.extend([participant] * n_windows)
        pair_list.extend([(parent_val, child_val)] * n_windows)

    X            = np.concatenate(X_list, axis=0).astype(np.float32)
    y            = np.array(y_list,    dtype=np.int64)
    y_child      = np.array(yc_list,   dtype=np.int64)
    ids          = np.array(id_list,   dtype=np.int64)
    participants = np.array(part_list, dtype=np.int64)
    pairs_arr    = np.array(pair_list, dtype=np.int64)

    print(f"\n[OurData all] {len(X)} windows  |  "
          f"{len(all_pairs)} pairs  |  "
          f"{len(np.unique(participants))} participants")
    print(f"  Label 0 (arm curl)    : {(y==0).sum()}")
    print(f"  Label 1 (bench press) : {(y==1).sum()}")

    return X, y, y_child, ids, participants, pairs_arr, all_pairs


def load_golden_for_forgetting(data_path, window_size=40, sensor="imu"):
    """
    Load ALL golden-dataset windows (both parent classes, all subjects) as a
    flat normalised array for use as a catastrophic-forgetting check.

    The golden dataset uses child column values {3, 5} for the two exercises;
    all other child values (including 0) are dropped.

    Returns
    -------
    X : (N, 1, T, C) float32
    y : (N,) int64  – 0 = arm curl (child=3), 1 = bench press (child=5)
    """
    label_map = {3: 0, 5: 1}    # golden child IDs → class indices

    if sensor == "imu":
        feature_cols = ["x_accel", "y_accel", "z_accel", "x_gyro", "y_gyro", "z_gyro"]
    elif sensor == "cap":
        feature_cols = ["C_1"]
    elif sensor == "combine":
        feature_cols = ["x_accel", "y_accel", "z_accel", "x_gyro", "y_gyro", "z_gyro", "C_1"]
    else:
        raise ValueError(f"Unknown sensor: {sensor}")

    df = pd.read_csv(data_path)
    df['participant'] = df['participant'].astype(int)
    df['label']       = df['child'].map(label_map)
    df = df.dropna(subset=['label']).reset_index(drop=True)
    df['label'] = df['label'].astype(int)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    X_list, y_list = [], []
    for _, grp in df.groupby('participant', sort=True):
        grp = grp.reset_index(drop=True)
        n_windows = len(grp) // window_size
        if n_windows == 0:
            continue
        cutoff = n_windows * window_size

        x_raw      = grp[feature_cols].iloc[:cutoff].to_numpy()
        X_windowed = x_raw.reshape(n_windows, window_size, len(feature_cols))

        y_raw      = grp['label'].iloc[:cutoff].to_numpy()
        y_reshaped = y_raw.reshape(n_windows, window_size)
        y_win, _   = stats.mode(y_reshaped, axis=1, keepdims=False)
        y_win      = y_win.flatten().astype(int)

        X_list.append(X_windowed)
        y_list.append(y_win)

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    N, T, C = X.shape
    scaler  = StandardScaler()
    X       = scaler.fit_transform(X.reshape(-1, C)).reshape(N, T, C)
    X       = X.reshape(N, 1, T, C).astype(np.float32)

    print(f"[Golden forgetting data] {N} windows  "
          f"(class 0: {(y==0).sum()}  class 1: {(y==1).sum()})")
    return X, y
