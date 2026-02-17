from scipy.io import loadmat

data = loadmat('data/gym_data/gym_data1.mat')
print(type(data))
print(data.keys())
print(data['data1_wrist'].shape)
print(data['data1_labels'])