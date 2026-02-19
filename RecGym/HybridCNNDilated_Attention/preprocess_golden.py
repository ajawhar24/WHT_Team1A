
import numpy as np
import scipy.io as sio
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pandas as pd
import tqdm
import scipy.stats as stats

# We need the following function to load and preprocess the High Gamma Dataset
# from preprocess_HGD import load_HGD_data

#%%

def load_data_Gym_SingleSubject(data_path, dataset, sensor="imu", test_size=0.2):
    # 1. Load Data
    df = pd.read_csv(data_path)
    
    # Handle Label Mapping
    if dataset == 'GymCap':
        label_map = {
            "Adductor": 1, "ArmCurl": 2, "BenchPress": 3, "LegCurl": 4, "LegPress": 5, "Null": 6, 
            "Riding": 7, "RopeSkipping": 8, "Running": 9, "Squat": 10, "StairClimber": 11, "Walking": 12
        }
    elif dataset == 'Golden': # Based on your first snippet
        label_map = {
            "bench_press": 1, "arm_curl": 2,
        }
        
    # Map labels and drop rows that don't match our label map (safety check)
    df['excercise'] = df['excercise'].map(label_map)
    df = df.dropna(subset=['excercise']) 
    
    # 2. Select Feature Columns
    if sensor == "cap":
        feature_cols = ["C_1"]
    elif sensor == "imu":
        feature_cols = ["x_accel", "y_accel", "z_accel", "x_gyro", "y_gyro", "z_gyro"]
    elif sensor == "combine":
        feature_cols = ["A_x", "A_y", "A_z", "G_x", "G_y", "G_z", "C_1"]

    label_col = "excercise" 

    # 3. Vectorized Windowing Function
    def create_windows(df, feat_cols, lbl_col, window_size=80):
        n_samples = len(df)
        n_windows = n_samples // window_size
        cutoff = n_windows * window_size
        
        # Process Features (X)
        x_raw = df[feat_cols].iloc[:cutoff].to_numpy()
        X_windowed = x_raw.reshape(n_windows, window_size, len(feat_cols))
        
        # Process Labels (y)
        y_raw = df[lbl_col].iloc[:cutoff].to_numpy()
        y_reshaped = y_raw.reshape(n_windows, window_size)
        
        # Get mode (most common label) for the window
        y_windowed, _ = stats.mode(y_reshaped, axis=1, keepdims=False)
        y_windowed = y_windowed.flatten()
        
        # Adjust to 0-indexed labels
        y_windowed = y_windowed - 1
        
        return X_windowed, y_windowed

    # 4. Create Windows from the ENTIRE dataset
    X_all, y_all = create_windows(df, feature_cols, label_col)

    print(f"Total Windows Created: {X_all.shape[0]}")

    # 5. Split into Train and Test
    # This randomly shuffles the windows and splits them
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=test_size, random_state=42, stratify=y_all
    )

    print("X_train shape:", X_train.shape)
    print("X_test shape: ", X_test.shape)

    return X_train, y_train, X_test, y_test


def get_data(path, dataset, sensor="imu"):
    # Note: 'subject' argument is removed as it's no longer needed
    
    # Load and split the single subject data
    X_train, y_train, X_test, y_test = load_data_Gym_SingleSubject(path, dataset, sensor)

    # Prepare training data (Reshape for CNN input: N, 1, Channels, Time)
    # Note: Your original code used (N, 1, Ch, T). 
    # Standard Keras usually prefers (N, T, Ch) or (N, T, Ch, 1), 
    # but I will keep your specific format:
    
    N_tr, T, N_ch = X_train.shape 
    # Transpose to match your original (N, 1, Ch, T) format if that's what your model expects
    # Original: X_train was (Windows, 80, 6) -> Reshape to (Windows, 1, 6, 80)
    X_train = X_train.transpose(0, 2, 1).reshape(N_tr, 1, N_ch, T)
    y_train_onehot = to_categorical(y_train)

    # Prepare testing data 
    N_te, T, N_ch = X_test.shape 
    X_test = X_test.transpose(0, 2, 1).reshape(N_te, 1, N_ch, T)
    y_test_onehot = to_categorical(y_test)

    return X_train, y_train, y_train_onehot, X_test, y_test, y_test_onehot