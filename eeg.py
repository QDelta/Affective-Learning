import random as rand
import numpy as np
from torch.utils.data import Dataset
import torch
import scipy.io as sio

# 15 * 3394 * 310
EEG_DATA = sio.loadmat('SEED-III/EEG_X.mat')['X'][0]
# 15 * 3394 * 1
EEG_LABEL = sio.loadmat('SEED-III/EEG_Y.mat')['Y'][0]

DOMAIN_NUM = 15
CLASS_NUM = 3
INPUT_DIM = 310

class EEGDataset(Dataset):
    def __init__(self, dom_for_test):
        pass

def split_data(dom_for_test):
    test_data = []
    for i, vec in enumerate(EEG_DATA[dom_for_test]):
        test_data.append((torch.from_numpy(vec).float(), torch.tensor(EEG_LABEL[dom_for_test][i][0] + 1)))
    train_data = []
    for t in range(DOMAIN_NUM):
        if t == dom_for_test:
            continue
        for i, vec in enumerate(EEG_DATA[t]):
            train_data.append((torch.from_numpy(vec).float(), (torch.tensor(EEG_LABEL[t][i][0] + 1), torch.tensor(t))))
    rand.shuffle(test_data)
    rand.shuffle(train_data)

    return train_data, test_data

def split_data_for_svm(dom_for_test):
    test_data = []
    test_label = []
    for i, vec in enumerate(EEG_DATA[dom_for_test]):
        test_data.append(vec)
        test_label.append(EEG_LABEL[dom_for_test][i][0] + 1)
    train_data = []
    train_label = []
    for t in range(DOMAIN_NUM):
        if t == dom_for_test:
            continue
        for i, vec in enumerate(EEG_DATA[t]):
            train_data.append(vec)
            train_label.append(EEG_LABEL[t][i][0] + 1)
    return (train_data, train_label, test_data, test_label)
