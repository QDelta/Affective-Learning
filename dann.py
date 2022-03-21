from torch import nn
from revgrad import RevGrad
from eeg import DOMAIN_NUM, CLASS_NUM, INPUT_DIM

class EEGDANN(nn.Module):
    def __init__(self, lambda_=1.0):
        super(EEGDANN, self).__init__()
        self.revgrad = RevGrad(lambda_)
        self.feat_extr = nn.Sequential(
            nn.Linear(INPUT_DIM, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        self.label_classify = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, CLASS_NUM)
        )
        self.domain_classify = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, DOMAIN_NUM)
        )

    def forward(self, input):
        feature = self.feat_extr(input)
        label_pred = self.label_classify(feature)
        domain_pred = self.domain_classify(self.revgrad(feature))
        return label_pred, domain_pred
