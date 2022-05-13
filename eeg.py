from torch.utils.data import Dataset
import scipy.io as sio
import numpy as np
from os.path import join

# 15 * 3394 * 310
EEG_DATA = sio.loadmat(join('SEED-III', 'EEG_X.mat'))['X'][0]
# 15 * 3394 * 1
EEG_LABEL = sio.loadmat(join('SEED-III', 'EEG_Y.mat'))['Y'][0]

DOMAIN_NUM = 15
SAMPLE_PER_DOMAIN = 3394
INPUT_DIM = 310
CLASS_NUM = 3

def split_data(dom_for_test):
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
    return (np.array(train_data, dtype=np.float64),
            np.array(train_label, dtype=np.int64),
            np.array(test_data, dtype=np.float64),
            np.array(test_label, dtype=np.int64))

class EEGDataset(Dataset):
    def __init__(
        self,
        dom_for_test: int,
        train: bool = True,
        transform = None
    ) -> None:
        self.transform = transform
        self.dom_for_test = dom_for_test
        if train:
            self.data, self.label, _, _ = split_data(dom_for_test)
        else:
            _, _, self.data, self.label = split_data(dom_for_test)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> tuple:
        data = self.data[idx]
        label = self.label[idx]

        if self.transform:
            data = self.transform(data)

        dom = idx // SAMPLE_PER_DOMAIN
        if dom >= self.dom_for_test:
            dom += 1

        return data, label, np.int64(dom)