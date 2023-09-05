import numpy as np

def datasets(file_path='./data/', train=True):
    if train:
        mode = 'train'
    else:
        mode = 'test'
    
    X = np.load(file_path + mode + '_data.npy')
    y = np.load(file_path + mode + '_label.npy').reshape(-1)
    return X, y