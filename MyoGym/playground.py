from scipy.io import loadmat
import pandas as pd
import os
data = loadmat('data/gym_data/gym_data1.mat')
# print(type(data))
# print(data.keys())
# print(data['data1_wrist'].shape)
#print(data['data1_labels'])
recgym_data = pd.read_csv('../RecGym/data/RecGym.csv')
print(recgym_data['Subject'].unique().shape)
columns_to_drop = ['Subject', 'Position','Session', 'C_1']
recgym_data = recgym_data.drop(columns_to_drop, axis=1)
print(recgym_data['Workout'].shape)
directory = 'data/gym_data'
filename = 'recgym.csv'
# os.makedirs(directory, exist_ok=True)
# full_path = os.path.join(directory, filename)

# recgym_data.to_csv(full_path, index=False)