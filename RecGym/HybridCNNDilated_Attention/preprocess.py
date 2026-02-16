
import numpy as np
import scipy.io as sio
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import pandas as pd
import tqdm
import scipy.stats as stats

# We need the following function to load and preprocess the High Gamma Dataset
# from preprocess_HGD import load_HGD_data

#%%

def most_common(lst):
    return max(set(lst), key=lst.count)


def fold(n, data):
    index =[x for x in range(1, 11)]
    index.pop(n-1)
    train = data.loc[index, :]
    test = data.loc[[n], :]
    return train, test


def load_data_Gym(data_path, subject, dataset, sensor="imu"):
    # 1. Load Data
    data_session = pd.read_csv(data_path)
    print(data_session["Subject"].unique())
    data_session = data_session.set_index("Subject")
    
    label_map = {
            "Adductor": 1, "ArmCurl": 2, "BenchPress": 3, "LegCurl": 4, "LegPress": 5, "Null": 6, 
            "Riding": 7, "RopeSkipping": 8, "Running": 9, "Squat": 10, "StairClimber": 11, "Walking": 12
        }
    data_session['Workout'] = data_session['Workout'].map(label_map)
    print(f"Fold (Test Subject) = {subject}")
    train, test = fold(subject, data_session)
    print(data_session["Workout"].unique())
    
    # 2. Select Feature Columns based on sensor type
    # We define the specific column names to ensure we don't grab "Position" or "Session"
    if sensor == "cap":
        feature_cols = ["C_1"]
    elif sensor == "imu":
        feature_cols = ["A_x", "A_y", "A_z", "G_x", "G_y", "G_z"]
    elif sensor == "combine":
        feature_cols = ["A_x", "A_y", "A_z", "G_x", "G_y", "G_z", "C_1"]
    
    # Define the label column name
    label_col = "Workout" 
    
    # 3. Helper function to process the dataframe (Vectorized)
    def process_split(df, feat_cols, lbl_col, window_size=80):
        # Calculate how many full windows fit in the data
        n_samples = len(df)
        n_windows = n_samples // window_size
        cutoff = n_windows * window_size
        
        # A. Process Features (X)
        # Drop extra rows that don't fit a full window
        x_raw = df[feat_cols].iloc[:cutoff].to_numpy()
        
        # Reshape: (Total_Rows, Channels) -> (Windows, Window_Size, Channels)
        # This replaces the entire first for-loop
        X_windowed = x_raw.reshape(n_windows, window_size, len(feat_cols))
        
        # B. Process Labels (y)
        y_raw = df[lbl_col].iloc[:cutoff].to_numpy()
        
        # Reshape to (Windows, Window_Size) to analyze each window
        y_reshaped = y_raw.reshape(n_windows, window_size)
        
        # Find the most common label (mode) for each window
        # This replaces the 'most_common' loop
        y_windowed, _ = stats.mode(y_reshaped, axis=1, keepdims=False)
        
        # Flatten to 1D array
        y_windowed = y_windowed.flatten()
        
        # Adjust labels to be 0-indexed (assuming your CSV has 1-based labels)
        y_windowed = y_windowed - 1
        
        return X_windowed, y_windowed

    # 4. Apply processing
    X_train, y_train = process_split(train, feature_cols, label_col)
    X_test, y_test = process_split(test, feature_cols, label_col)

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape: ", X_test.shape)
    print("y_test shape: ", y_test.shape)
    
    # Check class distribution
    unique_values, counts = np.unique(y_train, return_counts=True)
    for value, count in zip(unique_values, counts):
        print(f"Class {value}: {count} samples")

    return X_train, y_train, X_test, y_test


#%%
def get_data(path, subject, dataset,isShuffle = True):

    X_train, y_train, X_test, y_test = load_data_Gym(path, subject, dataset, sensor="imu")   ## sensor="combine"  "imu"  "cap"

    # shuffle the data 
    if isShuffle:
        X_train, y_train = shuffle(X_train, y_train,random_state=42)
        #X_test, y_test = shuffle(X_test, y_test,random_state=42)

    # Prepare training data     
    N_tr, N_ch, T = X_train.shape 
    X_train = X_train.reshape(N_tr, 1, N_ch, T)
    y_train_onehot = to_categorical(y_train)
    # Prepare testing data 
    N_tr, N_ch, T = X_test.shape 
    X_test = X_test.reshape(N_tr, 1, N_ch, T)
    y_test_onehot = to_categorical(y_test)

    return X_train, y_train, y_train_onehot, X_test, y_test, y_test_onehot

