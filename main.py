import scipy.io as sio
import torch
import numpy as np
import sklearn.svm as svm
from dann import EEGDANN

# 15 * 3394 * 310
EEG_DATA = sio.loadmat('SEED-III/EEG_X.mat')['X'][0]
# 15 * 3394 * 1
EEG_LABEL = sio.loadmat('SEED-III/EEG_Y.mat')['Y'][0]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = EEGDANN().to(device)
