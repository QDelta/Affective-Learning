from torch.utils.data import Dataset
import scipy.io as sio

# 15 * 3394 * 310
EEG_DATA = sio.loadmat('SEED-III/EEG_X.mat')['X'][0]
# 15 * 3394 * 1
EEG_LABEL = sio.loadmat('SEED-III/EEG_Y.mat')['Y'][0]

DOMAIN_NUM = 15
SAMPLE_PER_DOMAIN = 3394
INPUT_DIM = 310
CLASS_NUM = 3

class EEGDataset(Dataset):
    def __init__(
        self,
        dom_for_test: int,
        train: bool = True,
        transform = None
    ) -> None:
        self.dom_for_test = dom_for_test
        self.train = train
        self.transform = transform

    def __len__(self) -> int:
        if self.train:
            return (DOMAIN_NUM - 1) * SAMPLE_PER_DOMAIN
        else:
            return SAMPLE_PER_DOMAIN

    def __getitem__(self, idx) -> tuple:
        dom = self.dom_for_test
        if self.train:
            dom = idx // SAMPLE_PER_DOMAIN
            if dom == self.dom_for_test:
                dom += 1
            idx %= SAMPLE_PER_DOMAIN

        data = EEG_DATA[dom][idx]
        label = EEG_LABEL[dom][idx][0] + 1  # [-1, 1] -> [0, 2]

        if self.transform:
            data = self.transform(data)

        return data, label, dom

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
