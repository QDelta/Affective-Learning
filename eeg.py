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

def split_data(target_dom):
    target_data = []
    target_label = []
    for i, vec in enumerate(EEG_DATA[target_dom]):
        target_data.append(vec)
        target_label.append(EEG_LABEL[target_dom][i][0] + 1)
    source_data = []
    source_label = []
    for t in range(DOMAIN_NUM):
        if t == target_dom:
            continue
        for i, vec in enumerate(EEG_DATA[t]):
            source_data.append(vec)
            source_label.append(EEG_LABEL[t][i][0] + 1)
    return (np.array(source_data, dtype=np.float64),
            np.array(source_label, dtype=np.int64),
            np.array(target_data, dtype=np.float64),
            np.array(target_label, dtype=np.int64))

class EEGDataset(Dataset):
    def __init__(
        self,
        target_dom: int,
        source: bool = True,
        transform = None
    ) -> None:
        self.transform = transform
        self.target_dom = target_dom
        self.source = source
        if source:
            self.data, self.label, _, _ = split_data(target_dom)
        else:
            _, _, self.data, self.label = split_data(target_dom)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> tuple:
        data = self.data[idx]
        label = self.label[idx]

        if self.transform:
            data = self.transform(data)

        # if self.source:
        #     dom = idx // SAMPLE_PER_DOMAIN
        #     if dom >= self.target_dom:
        #         dom += 1
        # else:
        #     dom = self.target_dom

        return data, label, np.float32(0 if self.source else 1)