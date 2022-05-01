from torch import nn
from kgrad import KGrad
from eeg import DOMAIN_NUM, CLASS_NUM, INPUT_DIM

FEATURE_DIM = 256

def feature_extractor():
    return nn.Sequential(
        nn.Linear(INPUT_DIM, 512),
        nn.ReLU(inplace=True),
        nn.Linear(512, 512),
        nn.ReLU(inplace=True),
        nn.Linear(512, FEATURE_DIM),
    )

def label_classifier():
    return nn.Sequential(
        nn.Linear(FEATURE_DIM, 256),
        nn.ReLU(inplace=True),
        nn.Linear(256, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, CLASS_NUM)
    )

def domain_classifier():
    return  nn.Sequential(
        nn.Linear(FEATURE_DIM, 256),
        nn.ReLU(inplace=True),
        nn.Linear(256, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, DOMAIN_NUM)
    )

class EEGDANN(nn.Module):
    def __init__(self):
        super(EEGDANN, self).__init__()
        self.feat_extr = feature_extractor()
        self.label_classify = label_classifier()
        self.domain_classify = domain_classifier()

    def forward(self, input, label_lambda=1.0, dom_lambda=-1.0):
        label_kgrad = KGrad(label_lambda)
        dom_kgrad = KGrad(dom_lambda)
        feature = self.feat_extr(input)
        label_pred = self.label_classify(label_kgrad(feature))
        domain_pred = self.domain_classify(dom_kgrad(feature))
        return label_pred, domain_pred

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.feat_extr = feature_extractor
        self.label_classify = label_classifier()

    def forward(self, input):
        feature = self.feat_extr(input)
        label_pred = self.label_classify(feature)
        return label_pred