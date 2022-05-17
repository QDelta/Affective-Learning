from torch import nn
from kgrad import KGradF
from eeg import CLASS_NUM, INPUT_DIM

FEATURE_DIM = 128

def feature_extractor():
    return nn.Sequential(
        nn.Linear(INPUT_DIM, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(True),
        nn.Linear(256, FEATURE_DIM),
        nn.BatchNorm1d(FEATURE_DIM),
        nn.Dropout(),
        nn.ReLU()
    )

def label_classifier():
    return nn.Sequential(
        nn.Linear(FEATURE_DIM, 64),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(64, 32),
        nn.BatchNorm1d(32),
        nn.ReLU(True),
        nn.Linear(32, CLASS_NUM)
    )

def domain_classifier():
    return nn.Sequential(
        nn.Linear(FEATURE_DIM, 32),
        nn.BatchNorm1d(32),
        nn.ReLU(True),
        nn.Linear(32, 1),
        nn.Sigmoid()
    )

class EEGDANN(nn.Module):
    def __init__(self):
        super(EEGDANN, self).__init__()
        self.feat_extr = feature_extractor()
        self.label_classify = label_classifier()
        self.domain_classify = domain_classifier()

    def forward(self, input, label_lambda=1.0, dom_lambda=-1.0):
        feature = self.feat_extr(input)
        label_pred = self.label_classify(KGradF.apply(feature, label_lambda))
        domain_pred = self.domain_classify(KGradF.apply(feature, dom_lambda))
        return label_pred, domain_pred